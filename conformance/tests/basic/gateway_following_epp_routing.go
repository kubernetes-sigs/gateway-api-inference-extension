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
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	k8sutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
	trafficutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/traffic"
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
		)

		httpRouteNN := types.NamespacedName{Name: "httproute-for-primary-gw", Namespace: appBackendNamespace}
		gatewayNN := types.NamespacedName{Name: "conformance-gateway", Namespace: infraNamespace}
		poolNN := types.NamespacedName{Name: "normal-gateway-pool", Namespace: appBackendNamespace}
		backendPodLabels := map[string]string{"app": "infra-backend"}

		t.Log("Verifying HTTPRoute and InferencePool are accepted and the Gateway has an address.")
		k8sutils.HTTPRouteMustBeAcceptedAndResolved(t, s.Client, s.TimeoutConfig, httpRouteNN, gatewayNN)
		k8sutils.InferencePoolMustBeAcceptedByParent(t, s.Client, poolNN)
		gwAddr := k8sutils.GetGatewayEndpoint(t, s.Client, s.TimeoutConfig, gatewayNN)

		t.Logf("Fetching backend pods with labels: %v", backendPodLabels)
		pods, err := k8sutils.GetPodsWithLabel(t, s.Client, appBackendNamespace, backendPodLabels)
		require.NoError(t, err, "Failed to get backend pods")
		require.Len(t, pods, expectedPodReplicas, "Expected to find %d backend pods, but found %d.", expectedPodReplicas, len(pods))

		podIPs := make([]string, len(pods))
		for i, pod := range pods {
			podIPs[i] = pod.Status.PodIP
		}

		requestBody := `{
            "model": "conformance-fake-model",
            "prompt": "Write as if you were a critic: San Francisco"
        }`

		testCases := []struct {
			name                    string
			podOrder                []string
			expectedBackendPodIndex int
		}{
			{
				name:                    fmt.Sprintf("should route to first pod in list: %s", pods[0].Name),
				podOrder:                []string{podIPs[0], podIPs[1], podIPs[2]},
				expectedBackendPodIndex: 0,
			},
			{
				name:                    fmt.Sprintf("should route to new first pod after reordering: %s", pods[2].Name),
				podOrder:                []string{podIPs[2], podIPs[1], podIPs[0]},
				expectedBackendPodIndex: 2,
			},
		}

		s.TimeoutConfig.MaxTimeToConsistency = 200 * time.Second
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				eppHeaderValue := strings.Join(tc.podOrder, ",")
				headers := map[string]string{eppSelectionHeaderName: eppHeaderValue}
				expectedBackendPod := pods[tc.expectedBackendPodIndex]

				t.Logf("Sending request to %s with EPP header '%s: %s'", gwAddr, eppSelectionHeaderName, eppHeaderValue)
				t.Logf("Expecting traffic to be routed to pod %s (%s)", expectedBackendPod.Name, expectedBackendPod.Status.PodIP)

				trafficutils.MakeRequestAndExpectSuccessV2(
					t,
					s.RoundTripper,
					s.TimeoutConfig,
					gwAddr,
					trafficutils.Request{
						Host:      hostname,
						Path:      path,
						Headers:   headers,
						Method:    http.MethodPost,
						Body:      requestBody,
						Backend:   expectedBackendPod.Name,
						Namespace: appBackendNamespace,
					},
				)
			})
		}
	},
}
