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

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	k8sutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
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
			appBackendNamespace = "gateway-conformance-app-backend"
			infraNamespace      = "gateway-conformance-infra"
			backendAppLabelKey  = "app"

			primaryGatewayName    = "conformance-gateway"
			routeForPrimaryGWName = "route-for-primary-gateway"
			primaryPoolName       = "primary-pool"
			primaryBackendLabel   = "inference-model-1"
			primaryRoutePath      = "/test-primary-gateway"

			secondaryGatewayName    = "conformance-secondary-gateway"
			routeForSecondaryGWName = "route-for-secondary-gateway"
			secondaryPoolName       = "secondary-pool"
			secondaryBackendLabel   = "inference-model-2"
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
			primarySelector := labels.SelectorFromSet(labels.Set{backendAppLabelKey: primaryBackendLabel})
			primaryPod := k8sutils.GetPod(t, s.Client, appBackendNamespace, primarySelector, s.TimeoutConfig.RequestTimeout)

			trafficutils.MakeRequestAndExpectResponseFromPod(t, s.RoundTripper, s.TimeoutConfig, primaryGwAddr, primaryRoutePath, primaryPod)
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
			secondarySelector := labels.SelectorFromSet(labels.Set{backendAppLabelKey: secondaryBackendLabel})
			secondaryPod := k8sutils.GetPod(t, s.Client, appBackendNamespace, secondarySelector, s.TimeoutConfig.RequestTimeout)

			trafficutils.MakeRequestAndExpectResponseFromPodWithHostname(t, s.RoundTripper, s.TimeoutConfig, secondaryGwAddr, secondaryRoutePath, secondaryRouteHostname, secondaryPod)
		})
	},
}
