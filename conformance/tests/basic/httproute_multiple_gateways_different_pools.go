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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
	gatewayk8utils "sigs.k8s.io/gateway-api/conformance/utils/kubernetes"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"

	// Import the tests package to append to ConformanceTests
	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	"sigs.k8s.io/gateway-api-inference-extension/conformance/utils/config"
	infrakubernetes "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, HTTPRouteMultipleGatewaysDifferentPools)
}

var HTTPRouteMultipleGatewaysDifferentPools = suite.ConformanceTest{
	ShortName:   "HTTPRouteMultipleGatewaysDifferentPools",
	Description: "Validates two HTTPRoutes on different Gateways successfully referencing different InferencePools.",
	Manifests:   []string{"tests/basic/httproute_multiple_gateways_different_pools.yaml"},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		//TODO(877): changing the resoruce names to use primary secondary
		const (
			appBackendNamespace     = "gateway-conformance-app-backend"
			infraNamespace          = "gateway-conformance-infra"
			primaryGatewayName      = "conformance-gateway"
			secondaryGatewayName    = "conformance-secondary-gateway"
			routeForPrimaryGWName   = "route-for-gw1"
			routeForSecondaryGWName = "route-for-gw2"
			primaryPoolName         = "pool-a"
			secondaryPoolName       = "pool-b"
		)

		routeForPrimaryGWNN := types.NamespacedName{Name: routeForPrimaryGWName, Namespace: appBackendNamespace}
		routeForSecondaryGWNN := types.NamespacedName{Name: routeForSecondaryGWName, Namespace: appBackendNamespace}
		primaryPoolNN := types.NamespacedName{Name: primaryPoolName, Namespace: appBackendNamespace}
		secondaryPoolNN := types.NamespacedName{Name: secondaryPoolName, Namespace: appBackendNamespace}
		primaryGatewayNN := types.NamespacedName{Name: primaryGatewayName, Namespace: infraNamespace}
		secondaryGatewayNN := types.NamespacedName{Name: secondaryGatewayName, Namespace: infraNamespace}

		var timeoutConfig config.InferenceExtensionTimeoutConfig = config.DefaultInferenceExtensionTimeoutConfig()

		t.Run("HTTPRoute for Primary Gateway should be Accepted and have ResolvedRefs", func(t *testing.T) {
			acceptedCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionAccepted),
				Status: metav1.ConditionTrue,
				Reason: string(gatewayv1.RouteReasonAccepted),
			}
			gatewayk8utils.HTTPRouteMustHaveCondition(t, s.Client, timeoutConfig.TimeoutConfig, routeForPrimaryGWNN, primaryGatewayNN, acceptedCondition)
			t.Logf("HTTPRoute %s is Accepted by Primary Gateway %s", routeForPrimaryGWNN.String(), primaryGatewayNN.String())

			resolvedRefsCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionResolvedRefs),
				Status: metav1.ConditionTrue,
				Reason: string(gatewayv1.RouteReasonResolvedRefs),
			}
			gatewayk8utils.HTTPRouteMustHaveCondition(t, s.Client, timeoutConfig.TimeoutConfig, routeForPrimaryGWNN, primaryGatewayNN, resolvedRefsCondition)
			t.Logf("HTTPRoute %s has all references resolved by Primary Gateway %s", routeForPrimaryGWNN.String(), primaryGatewayNN.String())
		})

		t.Run("Primary InferencePool (pool-a) should be Accepted", func(t *testing.T) {
			acceptedCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionAccepted),
				Status: metav1.ConditionTrue,
				Reason: string(gatewayv1.RouteReasonAccepted),
			}
			infrakubernetes.InferencePoolMustHaveCondition(t, s.Client, primaryPoolNN, acceptedCondition)
			t.Logf("Primary InferencePool %s parent status shows Accepted by Primary Gateway %s (via HTTPRoute %s)", primaryPoolNN.String(), primaryGatewayNN.String(), routeForPrimaryGWNN.String())
		})

		t.Run("HTTPRoute for Secondary Gateway should be Accepted and have ResolvedRefs", func(t *testing.T) {
			acceptedCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionAccepted),
				Status: metav1.ConditionTrue,
				Reason: string(gatewayv1.RouteReasonAccepted),
			}
			gatewayk8utils.HTTPRouteMustHaveCondition(t, s.Client, timeoutConfig.TimeoutConfig, routeForSecondaryGWNN, secondaryGatewayNN, acceptedCondition)
			t.Logf("HTTPRoute %s is Accepted by Secondary Gateway %s", routeForSecondaryGWNN.String(), secondaryGatewayNN.String())

			resolvedRefsCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionResolvedRefs),
				Status: metav1.ConditionTrue,
				Reason: string(gatewayv1.RouteReasonResolvedRefs),
			}
			gatewayk8utils.HTTPRouteMustHaveCondition(t, s.Client, timeoutConfig.TimeoutConfig, routeForSecondaryGWNN, secondaryGatewayNN, resolvedRefsCondition)
			t.Logf("HTTPRoute %s has all references resolved by Secondary Gateway %s", routeForSecondaryGWNN.String(), secondaryGatewayNN.String())
		})

		t.Run("Secondary InferencePool (pool-b) should be Accepted", func(t *testing.T) {
			acceptedCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionAccepted),
				Status: metav1.ConditionTrue,
				Reason: string(gatewayv1.RouteReasonAccepted),
			}
			infrakubernetes.InferencePoolMustHaveCondition(t, s.Client, secondaryPoolNN, acceptedCondition)
			t.Logf("Secondary InferencePool %s parent status shows Accepted by Secondary Gateway %s (via HTTPRoute %s)", secondaryPoolNN.String(), secondaryGatewayNN.String(), routeForSecondaryGWNN.String())
		})
	},
}
