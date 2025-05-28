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
	"testing"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"

	// Import the tests package to append to ConformanceTests
	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	k8utils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, HTTPRouteMultipleGatewaysDifferentPools)
}

var HTTPRouteMultipleGatewaysDifferentPools = suite.ConformanceTest{
	ShortName:   "HTTPRouteMultipleGatewaysDifferentPools",
	Description: "Validates two HTTPRoutes on different Gateways successfully referencing different InferencePools.",
	Manifests:   []string{"tests/basic/httproute_multiple_gateways_different_pools.yaml"},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			appBackendNamespace     = "gateway-conformance-app-backend"
			infraNamespace          = "gateway-conformance-infra"
			primaryGatewayName      = "conformance-gateway"
			secondaryGatewayName    = "conformance-secondary-gateway"
			routeForPrimaryGWName   = "route-for-primary-gateway"
			routeForSecondaryGWName = "route-for-secondary-gateway"
			primaryPoolName         = "primary-pool"
			secondaryPoolName       = "secondary-pool"
		)

		routeForPrimaryGWNN := types.NamespacedName{Name: routeForPrimaryGWName, Namespace: appBackendNamespace}
		routeForSecondaryGWNN := types.NamespacedName{Name: routeForSecondaryGWName, Namespace: appBackendNamespace}
		primaryPoolNN := types.NamespacedName{Name: primaryPoolName, Namespace: appBackendNamespace}
		secondaryPoolNN := types.NamespacedName{Name: secondaryPoolName, Namespace: appBackendNamespace}
		primaryGatewayNN := types.NamespacedName{Name: primaryGatewayName, Namespace: infraNamespace}
		secondaryGatewayNN := types.NamespacedName{Name: secondaryGatewayName, Namespace: infraNamespace}

		t.Run("Primary HTTPRoute and associated InferencePool should be Accepted and RouteAccepted", func(t *testing.T) {
			k8utils.HTTPRouteAndInferencePoolMustBeAcceptedAndRouteAccepted(
				t,
				s.Client,
				routeForPrimaryGWNN,
				primaryGatewayNN,
				primaryPoolNN,
			)
			t.Logf("Primary path: HTTPRoute %s -> Gateway %s -> InferencePool %s verified.",
				routeForPrimaryGWNN.String(), primaryGatewayNN.String(), primaryPoolNN.String())
		})

		t.Run("Secondary HTTPRoute and associated InferencePool should be Accepted and RouteAccepted", func(t *testing.T) {
			k8utils.HTTPRouteAndInferencePoolMustBeAcceptedAndRouteAccepted(
				t,
				s.Client,
				routeForSecondaryGWNN,
				secondaryGatewayNN,
				secondaryPoolNN,
			)
			t.Logf("Secondary path: HTTPRoute %s -> Gateway %s -> InferencePool %s verified.",
				routeForSecondaryGWNN.String(), secondaryGatewayNN.String(), secondaryPoolNN.String())
		})

		// TODO(#879): Add test to send a request and verify routing to the correct InferencePool.
	},
}
