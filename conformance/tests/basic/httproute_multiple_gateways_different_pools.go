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

	"k8s.io/apimachinery/pkg/types"
	gwhttp "sigs.k8s.io/gateway-api/conformance/utils/http"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	k8utils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
	trafficutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/traffic"
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
			appBackendNamespace            = "gateway-conformance-app-backend"
			infraNamespace                 = "gateway-conformance-infra"
			primaryGatewayName             = "conformance-gateway"
			secondaryGatewayName           = "conformance-secondary-gateway"
			routeForPrimaryGWName          = "route-for-primary-gateway"
			routeForSecondaryGWName        = "route-for-secondary-gateway"
			primaryPoolName                = "primary-pool"
			secondaryPoolName              = "secondary-pool"
			primaryBackendDeploymentName   = "multi-gw-primary-backend-deployment"
			secondaryBackendDeploymentName = "multi-gw-secondary-backend-deployment"
			primaryRouteHostname           = "primary.example.com"
			primaryRoutePath               = "/test-primary-gateway"
			secondaryRouteHostname         = "secondary.example.com"
			secondaryRoutePath             = "/test-secondary-gateway"
		)

		routeForPrimaryGWNN := types.NamespacedName{Name: routeForPrimaryGWName, Namespace: appBackendNamespace}
		routeForSecondaryGWNN := types.NamespacedName{Name: routeForSecondaryGWName, Namespace: appBackendNamespace}
		primaryPoolNN := types.NamespacedName{Name: primaryPoolName, Namespace: appBackendNamespace}
		secondaryPoolNN := types.NamespacedName{Name: secondaryPoolName, Namespace: appBackendNamespace}
		primaryGatewayNN := types.NamespacedName{Name: primaryGatewayName, Namespace: infraNamespace}
		secondaryGatewayNN := types.NamespacedName{Name: secondaryGatewayName, Namespace: infraNamespace}

		t.Run("Primary HTTPRoute, InferencePool, and Gateway path: verify status and traffic", func(t *testing.T) {
			k8utils.HTTPRouteAndInferencePoolMustBeAcceptedAndRouteAccepted(
				t,
				s.Client,
				routeForPrimaryGWNN,
				primaryGatewayNN,
				primaryPoolNN,
			)
			t.Logf("Primary path: HTTPRoute %s -> Gateway %s -> InferencePool %s status verified.",
				routeForPrimaryGWNN.String(), primaryGatewayNN.String(), primaryPoolNN.String())
			t.Logf("Fetching Gateway Endpoint for Primary Gateway...")
			primaryGwAddr := k8utils.GetGatewayEndpoint(t, s.Client, s.TimeoutConfig, primaryGatewayNN)

			t.Logf("Testing traffic to Primary Gateway: %s, expecting pod to start with: %s",
				primaryGwAddr, primaryBackendDeploymentName)

			primaryExpectedResponse := trafficutils.BuildExpectedHTTPResponse(
				primaryRouteHostname,
				primaryRoutePath,
				http.StatusOK,
				primaryBackendDeploymentName,
				appBackendNamespace,
			)

			gwhttp.MakeRequestAndExpectEventuallyConsistentResponse(
				t,
				s.RoundTripper,
				s.TimeoutConfig,
				primaryGwAddr,
				primaryExpectedResponse,
			)
			t.Logf("Successfully routed traffic via Primary Gateway, response from pod starting with %s", primaryBackendDeploymentName)
		})

		t.Run("Secondary HTTPRoute, InferencePool, and Gateway path: verify status and traffic", func(t *testing.T) {
			k8utils.HTTPRouteAndInferencePoolMustBeAcceptedAndRouteAccepted(
				t,
				s.Client,
				routeForSecondaryGWNN,
				secondaryGatewayNN,
				secondaryPoolNN,
			)
			t.Logf("Secondary path: HTTPRoute %s -> Gateway %s -> InferencePool %s status verified.",
				routeForSecondaryGWNN.String(), secondaryGatewayNN.String(), secondaryPoolNN.String())

			t.Logf("Fetching Gateway Endpoint for Secondary Gateway...")
			secondaryGwAddr := k8utils.GetGatewayEndpoint(t, s.Client, s.TimeoutConfig, secondaryGatewayNN)

			t.Logf("Testing traffic to Secondary Gateway: %s, expecting pod to start with: %s",
				secondaryGwAddr, secondaryBackendDeploymentName)

			secondaryExpectedResponse := trafficutils.BuildExpectedHTTPResponse(
				secondaryRouteHostname,
				secondaryRoutePath,
				http.StatusOK,
				secondaryBackendDeploymentName,
				appBackendNamespace,
			)

			gwhttp.MakeRequestAndExpectEventuallyConsistentResponse(
				t,
				s.RoundTripper,
				s.TimeoutConfig,
				secondaryGwAddr,
				secondaryExpectedResponse,
			)
			t.Logf("Successfully routed traffic via Secondary Gateway, response from pod starting with %s", secondaryBackendDeploymentName)
		})
	},
}
