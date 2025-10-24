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
	ConformanceTests = append(ConformanceTests, GatewayFollowingEPPRoutingWithDataParallelism)
}

// GatewayFollowingEPPRoutingWithDataParallelism verifies that with multiple targetPorts (ranks)
// the gateway still routes *only* to pods returned by EPP. Rank/port fan-out is exercised by load.
var GatewayFollowingEPPRoutingWithDataParallelism = suite.ConformanceTest{
	ShortName:   "GatewayFollowingEPPRoutingWithDataParallelism",
	Description: "Inference gateway should restrict traffic to EPP-selected pods while EPP balances across multiple targetPorts (DP ranks)",
	Manifests:   []string{"tests/gateway_following_epp_routing_dp.yaml"},
	Features: []features.FeatureName{
		features.FeatureName("SupportInferencePool"),
		features.SupportGateway,
	},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			hostname            = "primary.example.com"
			path                = "/primary-gateway-dp-test"
			appPodBackendPrefix = "dp-inference-model-server" // Must match the app label in the backend pods
		)

		httpRouteNN := types.NamespacedName{Name: "httproute-for-primary-gw-dp", Namespace: resources.AppBackendNamespace}
		gatewayNN := resources.PrimaryGatewayNN
		poolNN := types.NamespacedName{Name: "dp-inference-pool", Namespace: resources.AppBackendNamespace}
		backendPodLabels := map[string]string{"app": "dp-inference-model-server"}

		t.Log("Verifying HTTPRoute and InferencePool are accepted and the Gateway has an address.")
		k8sutils.HTTPRouteMustBeAcceptedAndResolved(t, s.Client, s.TimeoutConfig, httpRouteNN, gatewayNN)
		k8sutils.InferencePoolMustBeAcceptedByParent(t, s.Client, poolNN, gatewayNN)
		gwAddr := k8sutils.GetGatewayEndpoint(t, s.Client, s.TimeoutConfig, gatewayNN)

		t.Logf("Fetching DP backend pods with labels: %v", backendPodLabels)
		pods, err := k8sutils.GetPodsWithLabel(t, s.Client, resources.AppBackendNamespace, backendPodLabels, s.TimeoutConfig)
		require.NoError(t, err, "Failed to get DP backend pods")
		require.Len(t, pods, 3, "Expected to find 3 DP backend pods, found %d", len(pods))

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

		// Smoke: single-pod pin to ensure header filter works before main cases.
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
						test.HeaderTestEppEndPointSelectionKey: podIPs[0],
					},
				},
				Response: gwhttp.Response{StatusCodes: []int{http.StatusOK}},
				Backend:   podNames[0],
				Namespace: resources.AppBackendNamespace,
			},
		)

		testCases := []struct {
			name                                  string
			podIPsToBeReturnedByEPP               []string
			expectAllRequestsRoutedWithinPodNames []string
		}{
			{
				name:                                  "DP routes only to one designated pod (multiple ranks under the hood)",
				podIPsToBeReturnedByEPP:               []string{podIPs[1]},
				expectAllRequestsRoutedWithinPodNames: []string{podNames[1]},
			},
			{
				name:                                  "DP routes only to two designated pods",
				podIPsToBeReturnedByEPP:               []string{podIPs[0], podIPs[2]},
				expectAllRequestsRoutedWithinPodNames: []string{podNames[0], podNames[2]},
			},
			{
				name:                                  "DP routes only to all pods (EPP returns all; ranks balanced internally)",
				podIPsToBeReturnedByEPP:               []string{podIPs[0], podIPs[1], podIPs[2]},
				expectAllRequestsRoutedWithinPodNames: []string{podNames[0], podNames[1], podNames[2]},
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				eppHeaderValue := strings.Join(tc.podIPsToBeReturnedByEPP, ",")
				headers := map[string]string{test.HeaderTestEppEndPointSelectionKey: eppHeaderValue}

				assertTrafficOnlyReachesToExpectedPodsDP(
					t, s, gwAddr,
					gwhttp.ExpectedResponse{
						Request: gwhttp.Request{
							Host:    hostname,
							Path:    path,
							Method:  http.MethodPost,
							Body:    requestBody,
							Headers: headers,
						},
						Response: gwhttp.Response{StatusCode: http.StatusOK},
						Backend:   appPodBackendPrefix,
						Namespace: resources.AppBackendNamespace,
					},
					tc.expectAllRequestsRoutedWithinPodNames,
				)
			})
		}
	},
}

func assertTrafficOnlyReachesToExpectedPodsDP(t *testing.T, suite *suite.ConformanceTestSuite, gwAddr string, expected gwhttp.ExpectedResponse, expectedPodNames []string) {
	t.Helper()
	const (
		concurrency   = 20
		totalRequests = 200
	)
	var (
		rt = suite.RoundTripper
		g  errgroup.Group
		r  = gwhttp.MakeRequest(t, &expected, gwAddr, "HTTP", "http")
	)
	g.SetLimit(concurrency)

	for i := 0; i < totalRequests; i++ {
		g.Go(func() error {
			cReq, cRes, err := rt.CaptureRoundTrip(r)
			if err != nil {
				return fmt.Errorf("failed roundtrip: %w", err)
			}
			if err := gwhttp.CompareRoundTrip(t, &r, cReq, cRes, expected); err != nil {
				return fmt.Errorf("expectation failed: %w", err)
			}
			// Enforce no leakage to non-selected pods (ports/ranks are internal).
			if !slices.Contains(expectedPodNames, cReq.Pod) {
				return fmt.Errorf("unexpected pod %q (expected one of %v)", cReq.Pod, expectedPodNames)
			}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		t.Fatalf("Requests were not confined to expected pods (DP), err: %v", err)
	}
	t.Logf("DP traffic successfully restricted to expected pods: %v", expectedPodNames)
}
