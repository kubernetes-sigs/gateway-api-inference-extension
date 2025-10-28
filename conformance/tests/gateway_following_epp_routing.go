/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tests

import (
	"fmt"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"golang.org/x/sync/errgroup"
	"k8s.io/apimachinery/pkg/types"
	gwhttp "sigs.k8s.io/gateway-api/conformance/utils/http"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/resources"
	k8sutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/test"
)

func init() {
	ConformanceTests = append(ConformanceTests, GatewayFollowingEPPRouting)
}

// GatewayFollowingEPPRouting defines the test case for verifying gateway should send traffic to an endpoint in the list returned by EPP.
var GatewayFollowingEPPRouting = suite.ConformanceTest{
	ShortName:   "GatewayFollowingEPPRouting",
	Description: "Inference gateway should send traffic to an endpoint in the list returned by EPP",
	Manifests:   []string{"tests/gateway_following_epp_routing.yaml"},
	Features: []features.FeatureName{
		features.FeatureName("SupportInferencePool"),
		features.SupportGateway,
	},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			hostname            = "primary.example.com"
			path                = "/primary-gateway-test"
			appPodBackendPrefix = "primary-inference-model-server"
		)

		httpRouteNN := types.NamespacedName{Name: "httproute-for-primary-gw", Namespace: resources.AppBackendNamespace}
		gatewayNN := resources.PrimaryGatewayNN
		poolNN := resources.PrimaryInferencePoolNN
		backendPodLabels := map[string]string{"app": resources.PrimaryModelServerAppLabel}

		t.Log("Verifying HTTPRoute and InferencePool are accepted and the Gateway has an address.")
		k8sutils.HTTPRouteMustBeAcceptedAndResolved(t, s.Client, s.TimeoutConfig, httpRouteNN, gatewayNN)
		k8sutils.InferencePoolMustBeAcceptedByParent(t, s.Client, poolNN, gatewayNN)
		gwAddr := k8sutils.GetGatewayEndpoint(t, s.Client, s.TimeoutConfig, gatewayNN)

		t.Logf("Fetching backend pods with labels: %v", backendPodLabels)
		pods, err := k8sutils.GetPodsWithLabel(t, s.Client, resources.AppBackendNamespace, backendPodLabels, s.TimeoutConfig)
		require.NoError(t, err, "Failed to get backend pods")
		require.Len(t, pods, resources.ModelServerPodReplicas, "Expected to find %d backend pods, but found %d.", resources.ModelServerPodReplicas, len(pods))

		podIPs := make([]string, len(pods))
		podNames := make([]string, len(pods))
		for i, pod := range pods {
			podIPs[i] = pod.Status.PodIP
			podNames[i] = pod.Name
		}

		requestBody := `{
            "model": "conformance-fake-model",
            "prompt": "Write as if you were a critic: San Francisco"
        }`

		for i := 0; i < len(pods); i++ {
			// Send an initial request targeting a single pod and wait for it to be successful to ensure the Gateway and EPP
			// are functioning correctly before running the main test cases.
			gwhttp.MakeRequestAndExpectEventuallyConsistentResponse(
				t,
				s.RoundTripper,
				s.TimeoutConfig,
				gwAddr,
				gwhttp.ExpectedResponse{
					Request: gwhttp.Request{
						Host:   hostname,
						Path:   path,
						Method: http.MethodPost,
						Body:   requestBody,
						Headers: map[string]string{
							test.HeaderTestEppEndPointSelectionKey: podIPs[i],
						},
					},
					Response: gwhttp.Response{
						StatusCodes: []int{http.StatusOK},
					},
					Backend:   podNames[i],
					Namespace: resources.AppBackendNamespace,
				},
			)
		}

		testCases := []struct {
			name                                  string
			podIPsToBeReturnedByEPP               []string
			expectAllRequestsRoutedWithinPodNames []string
		}{
			{
				name:                                  "should route traffic to a single designated pod",
				podIPsToBeReturnedByEPP:               []string{podIPs[2]},
				expectAllRequestsRoutedWithinPodNames: []string{podNames[2]},
			},
			{
				name:                                  "should route traffic to two designated pods",
				podIPsToBeReturnedByEPP:               []string{podIPs[0], podIPs[1]},
				expectAllRequestsRoutedWithinPodNames: []string{podNames[0], podNames[1]},
			},
			{
				name:                                  "should route traffic to all available pods",
				podIPsToBeReturnedByEPP:               []string{podIPs[0], podIPs[1], podIPs[2]},
				expectAllRequestsRoutedWithinPodNames: []string{podNames[0], podNames[1], podNames[2]},
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				eppHeaderValue := strings.Join(tc.podIPsToBeReturnedByEPP, ",")
				headers := map[string]string{
					test.HeaderTestEppEndPointSelectionKey: eppHeaderValue,
				}

				t.Logf("Sending request to %s with EPP header '%s: %s'", gwAddr, test.HeaderTestEppEndPointSelectionKey, eppHeaderValue)
				t.Logf("Expecting traffic to be routed to pod: %v", tc.expectAllRequestsRoutedWithinPodNames)

				assertTrafficOnlyReachesToExpectedPods(t, s, gwAddr, gwhttp.ExpectedResponse{
					Request: gwhttp.Request{
						Host:    hostname,
						Path:    path,
						Method:  http.MethodPost,
						Body:    requestBody,
						Headers: headers,
					},
					Response: gwhttp.Response{
						StatusCode: http.StatusOK,
					},
					Backend:   appPodBackendPrefix,
					Namespace: resources.AppBackendNamespace,
				}, tc.expectAllRequestsRoutedWithinPodNames)
			})
		}
	},
}

func assertTrafficOnlyReachesToExpectedPods(t *testing.T, suite *suite.ConformanceTestSuite, gwAddr string, expected gwhttp.ExpectedResponse, expectedPodNames []string) {
	t.Helper()
	const (
		concurrentRequests = 10
		totalRequests      = 100
	)
	var (
		roundTripper = suite.RoundTripper
		g            errgroup.Group
		req          = gwhttp.MakeRequest(t, &expected, gwAddr, "HTTP", "http")
	)
	g.SetLimit(concurrentRequests)
	for i := 0; i < totalRequests; i++ {
		g.Go(func() error {
			cReq, cRes, err := roundTripper.CaptureRoundTrip(req)
			if err != nil {
				return fmt.Errorf("failed to roundtrip request: %w", err)
			}
			if err := gwhttp.CompareRoundTrip(t, &req, cReq, cRes, expected); err != nil {
				return fmt.Errorf("response expectation failed for request: %w", err)
			}

			if slices.Contains(expectedPodNames, cReq.Pod) {
				return nil
			}
			return fmt.Errorf("request was handled by an unexpected pod %q", cReq.Pod)
		})
	}
	if err := g.Wait(); err != nil {
		t.Fatalf("Not all the requests are sent to the expectedPods successfully, err: %v", err)
	}
	t.Logf("Traffic successfully reached only to expected pods: %v", expectedPodNames)
}
