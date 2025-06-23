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
	gwhttp "sigs.k8s.io/gateway-api/conformance/utils/http"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	k8sutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
	"sigs.k8s.io/gateway-api-inference-extension/conformance/utils/traffic"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, HTTPRouteMultipleGatewaysDifferentPools)
}

var HTTPRouteMultipleGatewaysDifferentPools = suite.ConformanceTest{
	ShortName:   "HTTPRouteMultipleGatewaysDifferentPools",
	Description: "Validates two HTTPRoutes on different Gateways successfully referencing different InferencePools and routes traffic accordingly.",
	Manifests:   []string{"tests/basic/httproute_multiple_gateways_different_pools.yaml"},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			appBackendNamespace = "gateway-conformance-app-backend"
			infraNamespace      = "gateway-conformance-infra"
			backendAppLabelKey  = "app"

			primaryGatewayName    = "conformance-primary-gateway"
			routeForPrimaryGWName = "route-for-primary-gateway"
			primaryPoolName       = "primary-inference-pool"
			primaryBackendLabel   = "primary-inference-model-server"
			primaryRoutePath      = "/test-primary-gateway"
			primaryRouteHostname  = "primary.example.com"

			secondaryGatewayName    = "conformance-secondary-gateway"
			routeForSecondaryGWName = "route-for-secondary-gateway"
			secondaryPoolName       = "secondary-inference-pool"
			secondaryBackendLabel   = "secondary-inference-model-server"
			secondaryRoutePath      = "/test-secondary-gateway"
			secondaryRouteHostname  = "secondary.example.com"
		)

		routeForPrimaryGWNN := types.NamespacedName{Name: routeForPrimaryGWName, Namespace: appBackendNamespace}
		routeForSecondaryGWNN := types.NamespacedName{Name: routeForSecondaryGWName, Namespace: appBackendNamespace}
		primaryPoolNN := types.NamespacedName{Name: primaryPoolName, Namespace: appBackendNamespace}
		secondaryPoolNN := types.NamespacedName{Name: secondaryPoolName, Namespace: appBackendNamespace}
		primaryGatewayNN := types.NamespacedName{Name: primaryGatewayName, Namespace: infraNamespace}
		secondaryGatewayNN := types.NamespacedName{Name: secondaryGatewayName, Namespace: infraNamespace}

		t.Run("Primary HTTPRoute, InferencePool, and Gateway path: verify status and traffic", func(t *testing.T) {
			k8sutils.HTTPRouteAndInferencePoolMustBeAcceptedAndRouteAccepted(
				t,
				s.Client,
				routeForPrimaryGWNN,
				primaryGatewayNN,
				primaryPoolNN,
			)

			primaryGwAddr := k8sutils.GetGatewayEndpoint(t, s.Client, s.TimeoutConfig, primaryGatewayNN)
			primaryBackendLabels := map[string]string{backendAppLabelKey: primaryBackendLabel}
			primaryPods, err := k8sutils.GetPodsWithLabel(t, s.Client, appBackendNamespace, primaryBackendLabels)
			require.NoError(t, err, "Failed to get pods for primary backend")
			require.NotEmpty(t, primaryPods, "No pods found for primary backend")

			var primaryPodNames []string
			for _, pod := range primaryPods {
				primaryPodNames = append(primaryPodNames, pod.Name)
			}

			traffic.AssertTrafficOnlyReachesToExpectedPods(t, s.RoundTripper, primaryGwAddr, gwhttp.ExpectedResponse{
				Request: gwhttp.Request{
					Path:   primaryRoutePath,
					Host:   primaryRouteHostname,
					Method: http.MethodGet,
				},
				Response: gwhttp.Response{
					StatusCode: http.StatusOK,
				},
				Namespace: appBackendNamespace,
			}, "", primaryPodNames)
		})

		t.Run("Secondary HTTPRoute, InferencePool, and Gateway path: verify status and traffic", func(t *testing.T) {
			k8sutils.HTTPRouteAndInferencePoolMustBeAcceptedAndRouteAccepted(
				t,
				s.Client,
				routeForSecondaryGWNN,
				secondaryGatewayNN,
				secondaryPoolNN,
			)

			secondaryGwAddr := k8sutils.GetGatewayEndpoint(t, s.Client, s.TimeoutConfig, secondaryGatewayNN)
			secondaryBackendLabels := map[string]string{backendAppLabelKey: secondaryBackendLabel}
			secondaryPods, err := k8sutils.GetPodsWithLabel(t, s.Client, appBackendNamespace, secondaryBackendLabels)
			require.NoError(t, err, "Failed to get pods for secondary backend")
			require.NotEmpty(t, secondaryPods, "No pods found for secondary backend")

			var secondaryPodNames []string
			for _, pod := range secondaryPods {
				secondaryPodNames = append(secondaryPodNames, pod.Name)
			}

			traffic.AssertTrafficOnlyReachesToExpectedPods(t, s.RoundTripper, secondaryGwAddr, gwhttp.ExpectedResponse{
				Request: gwhttp.Request{
					Path:   secondaryRoutePath,
					Host:   secondaryRouteHostname,
					Method: http.MethodGet,
				},
				Response: gwhttp.Response{
					StatusCode: http.StatusOK,
				},
				Namespace: appBackendNamespace,
			}, "", secondaryPodNames)
		})
	},
}
