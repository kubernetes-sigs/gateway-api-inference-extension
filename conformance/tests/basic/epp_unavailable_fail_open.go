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
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	k8sutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
	trafficutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/traffic"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, EppUnAvailableFailOpen)
}

var EppUnAvailableFailOpen = suite.ConformanceTest{
	ShortName:   "EppUnAvailableFailOpen",
	Description: "Inference gateway should send traffic to backends even when the EPP is unavailable (fail-open)",
	Manifests:   []string{"tests/basic/epp_unavailable_fail_open.yaml"},
	Features: []features.FeatureName{
		features.FeatureName("SupportInferencePool"),
		features.SupportGateway,
	},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		// Group constants for better readability, a common Go practice.
		const (
			appBackendNamespace    = "gateway-conformance-app-backend"
			infraNamespace         = "gateway-conformance-infra"
			hostname               = "secondary.example.com"
			path                   = "/failopen-pool-test"
			expectedPodReplicas    = 3
			eppSelectionHeaderName = "test-epp-endpoint-selection"
			appPodBackendPrefix    = "secondary-inference-model-server"
			requestBody            = `{
                "model": "conformance-fake-model",
                "prompt": "Write as if you were a critic: San Francisco"
            }`
		)

		httpRouteNN := types.NamespacedName{Name: "httproute-for-failopen-pool-gw", Namespace: appBackendNamespace}
		gatewayNN := types.NamespacedName{Name: "conformance-secondary-gateway", Namespace: infraNamespace}
		poolNN := types.NamespacedName{Name: "secondary-inference-pool", Namespace: appBackendNamespace}
		eppDeploymentNN := types.NamespacedName{Name: "secondary-app-endpoint-picker", Namespace: appBackendNamespace}
		backendPodLabels := map[string]string{"app": "secondary-inference-model-server"}

		k8sutils.HTTPRouteMustBeAcceptedAndResolved(t, s.Client, s.TimeoutConfig, httpRouteNN, gatewayNN)
		k8sutils.InferencePoolMustBeAcceptedByParent(t, s.Client, poolNN)
		gwAddr := k8sutils.GetGatewayEndpoint(t, s.Client, s.TimeoutConfig, gatewayNN)

		pods, err := k8sutils.GetPodsWithLabel(t, s.Client, appBackendNamespace, backendPodLabels)
		require.NoError(t, err, "Failed to get backend pods")
		require.Len(t, pods, expectedPodReplicas, "Expected to find %d backend pod, but found %d.", expectedPodReplicas, len(pods))

		targetPodIP := pods[0].Status.PodIP
		t.Run("Phase 1: Verify baseline connectivity with EPP available", func(t *testing.T) {
			t.Log("Sending request to ensure the Gateway and EPP are working correctly...")
			trafficutils.MakeRequestWithRequestParamAndExpectSuccess(
				t,
				s.RoundTripper,
				s.TimeoutConfig,
				gwAddr,
				trafficutils.Request{
					Host:      hostname,
					Path:      path,
					Headers:   map[string]string{eppSelectionHeaderName: targetPodIP},
					Method:    http.MethodPost,
					Body:      requestBody,
					Backend:   pods[0].Name, // Make sure the request is from the targetPod when the EPP is alive.
					Namespace: appBackendNamespace,
				},
			)
		})

		t.Run("Phase 2: Verify fail-open behavior after EPP becomes unavailable", func(t *testing.T) {
			t.Log("Simulating an EPP failure by deleting its deployment...")
			deleteErr := k8sutils.DeleteDeployment(t, s.Client, s.TimeoutConfig, eppDeploymentNN)
			require.NoError(t, deleteErr, "Failed to delete the EPP deployment")

			t.Log("Sending request again, expecting success to verify fail-open...")
			trafficutils.MakeRequestWithRequestParamAndExpectSuccess(
				t,
				s.RoundTripper,
				s.TimeoutConfig,
				gwAddr,
				trafficutils.Request{
					Host:      hostname,
					Path:      path,
					Headers:   map[string]string{eppSelectionHeaderName: targetPodIP},
					Method:    http.MethodPost,
					Body:      requestBody,
					Backend:   appPodBackendPrefix, // Only checks the prefix since the EPP is not alive and the response can return from any Pod.
					Namespace: appBackendNamespace,
				},
			)
		})
	},
}
