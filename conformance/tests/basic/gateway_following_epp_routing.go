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

package basic

import (
	"fmt"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"golang.org/x/sync/errgroup"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	k8sutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
	"sigs.k8s.io/gateway-api-inference-extension/conformance/utils/traffic"
	trafficutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/traffic"
	gwhttp "sigs.k8s.io/gateway-api/conformance/utils/http"
)

func init() {
	// Register the GatewayFollowingEPPRouting test case with the conformance suite.
	// This ensures it will be discovered and run by the test runner.
	tests.ConformanceTests = append(tests.ConformanceTests, GatewayFollowingEPPRouting)
}

// GatewayFollowingEPPRouting defines the test case for verifying gateway should send traffic to an endpoint in the list returned by EPP.
var GatewayFollowingEPPRouting = suite.ConformanceTest{
	ShortName:   "GatewayFollowingEPPRouting",
	Description: "Inference gateway should send traffic to an endpoint in the list returned by EPP",
	Manifests:   []string{"tests/basic/gateway_following_epp_routing.yaml"},
	Features: []features.FeatureName{
		features.FeatureName("SupportInferencePool"),
		features.SupportGateway,
	},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			appBackendNamespace = "gateway-conformance-app-backend"
			infraNamespace      = "gateway-conformance-infra"
			hostname            = "primary.example.com"
			path                = "/primary-gateway-test"
			expectedPodReplicas = 3
			// eppSelectionHeaderName is the custom header used by the testing-EPP service
			// to determine which endpoint to select.
			eppSelectionHeaderName = "test-epp-endpoint-selection"
			appPodBackendPrefix    = "primary-inference-model-server"
		)

		httpRouteNN := types.NamespacedName{Name: "httproute-for-primary-gw", Namespace: appBackendNamespace}
		gatewayNN := types.NamespacedName{Name: "conformance-primary-gateway", Namespace: infraNamespace}
		poolNN := types.NamespacedName{Name: "primary-inference-pool", Namespace: appBackendNamespace}
		backendPodLabels := map[string]string{"app": "primary-inference-model-server"}

		t.Log("Verifying HTTPRoute and InferencePool are accepted and the Gateway has an address.")
		k8sutils.HTTPRouteMustBeAcceptedAndResolved(t, s.Client, s.TimeoutConfig, httpRouteNN, gatewayNN)
		k8sutils.InferencePoolMustBeAcceptedByParent(t, s.Client, poolNN)
		gwAddr := k8sutils.GetGatewayEndpoint(t, s.Client, s.TimeoutConfig, gatewayNN)

		t.Logf("Fetching backend pods with labels: %v", backendPodLabels)
		pods, err := k8sutils.GetPodsWithLabel(t, s.Client, appBackendNamespace, backendPodLabels)
		require.NoError(t, err, "Failed to get backend pods")
		require.Len(t, pods, expectedPodReplicas, "Expected to find %d backend pods, but found %d.", expectedPodReplicas, len(pods))

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
			trafficutils.MakeRequestWithRequestParamAndExpectSuccess(
				t,
				s.RoundTripper,
				s.TimeoutConfig,
				gwAddr,
				trafficutils.Request{
					Host:      hostname,
					Path:      path,
					Headers:   map[string]string{eppSelectionHeaderName: podIPs[i]},
					Method:    http.MethodPost,
					Body:      requestBody,
					Backend:   podNames[i],
					Namespace: appBackendNamespace,
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
				headers := map[string]string{eppSelectionHeaderName: eppHeaderValue}

				t.Logf("Sending request to %s with EPP header '%s: %s'", gwAddr, eppSelectionHeaderName, eppHeaderValue)
				t.Logf("Expecting traffic to be routed to pod: %v", tc.expectAllRequestsRoutedWithinPodNames)

				assertTrafficOnlyReachesToExpectedPods(t, s, gwAddr, gwhttp.ExpectedResponse{
					Request: gwhttp.Request{
						Host:    hostname,
						Path:    path,
						Method:  http.MethodPost,
						Headers: headers,
					},
					Response: gwhttp.Response{
						StatusCode: http.StatusOK,
					},
					Backend:   appPodBackendPrefix,
					Namespace: appBackendNamespace,
				}, requestBody, tc.expectAllRequestsRoutedWithinPodNames)
			})
		}
	},
}

func assertTrafficOnlyReachesToExpectedPods(t *testing.T, suite *suite.ConformanceTestSuite, gwAddr string, expected gwhttp.ExpectedResponse, requestBody string, expectedPodNames []string) {
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
			cReq, cRes, err := traffic.MakeCallRoundTripper(t, roundTripper, &traffic.RequestWithBody{Request: req, Body: strings.NewReader(requestBody)})
			if err != nil {
				return fmt.Errorf("failed to roundtrip request: %w", err)
			}
			if err := gwhttp.CompareRequest(t, &req, cReq, cRes, expected); err != nil {
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
