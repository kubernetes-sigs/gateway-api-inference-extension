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
	k8utils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, HTTPRouteMultipleRulesDifferentPools)
}

// HTTPRouteMultipleRulesDifferentPools defines the test case for validating
// that an HTTPRoute can successfully route to multiple distinct InferencePools
// based on different rules.
var HTTPRouteMultipleRulesDifferentPools = suite.ConformanceTest{
	ShortName:   "HTTPRouteMultipleRulesDifferentPools",
	Description: "Validates that a single HTTPRoute can route to multiple different InferencePools based on distinct rules.",
	Manifests:   []string{"tests/basic/inferencepool_multiple_rules_different_pools.yaml"},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			appBackendNamespace = "gateway-conformance-app-backend"
			infraNamespace      = "gateway-conformance-infra"
			httpRouteName       = "httproute-multi-pool-rules"
			poolPrimaryName     = "pool-primary"
			poolSecondaryName   = "pool-secondary"
			gatewayName         = "conformance-gateway"
		)

		routeNN := types.NamespacedName{Name: httpRouteName, Namespace: appBackendNamespace}
		poolPrimaryNN := types.NamespacedName{Name: poolPrimaryName, Namespace: appBackendNamespace}
		poolSecondaryNN := types.NamespacedName{Name: poolSecondaryName, Namespace: appBackendNamespace}
		gatewayNN := types.NamespacedName{Name: gatewayName, Namespace: infraNamespace}

		var timeoutConfig config.InferenceExtensionTimeoutConfig = config.DefaultInferenceExtensionTimeoutConfig()

		t.Run("HTTPRoute should be Accepted and have ResolvedRefs", func(t *testing.T) {
			acceptedCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionAccepted),
				Status: metav1.ConditionTrue,
				Reason: string(gatewayv1.RouteReasonAccepted),
			}
			gatewayk8utils.HTTPRouteMustHaveCondition(t, s.Client, timeoutConfig.TimeoutConfig, routeNN, gatewayNN, acceptedCondition)
			t.Logf("HTTPRoute %s is Accepted by Gateway %s", routeNN.String(), gatewayNN.String())

			resolvedRefsCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionResolvedRefs),
				Status: metav1.ConditionTrue,
				Reason: string(gatewayv1.RouteReasonResolvedRefs),
			}
			gatewayk8utils.HTTPRouteMustHaveCondition(t, s.Client, timeoutConfig.TimeoutConfig, routeNN, gatewayNN, resolvedRefsCondition)
			t.Logf("HTTPRoute %s has all references resolved by Gateway %s", routeNN.String(), gatewayNN.String())
		})

		t.Run("InferencePool Primary should be Accepted", func(t *testing.T) {
			acceptedCondition := metav1.Condition{
				Type:   string(gatewayv1.GatewayConditionAccepted),
				Status: metav1.ConditionTrue,
				Reason: string(gatewayv1.GatewayReasonAccepted),
			}
			k8utils.InferencePoolMustHaveCondition(t, s.Client, poolPrimaryNN, acceptedCondition)
			t.Logf("InferencePool %s parent status shows Accepted by Gateway %s (via HTTPRoute %s)", poolPrimaryNN.String(), gatewayNN.String(), routeNN.String())
		})

		t.Run("InferencePool Secondary should be Accepted", func(t *testing.T) {
			acceptedCondition := metav1.Condition{
				Type:   string(gatewayv1.GatewayConditionAccepted),
				Status: metav1.ConditionTrue,
				Reason: string(gatewayv1.GatewayReasonAccepted),
			}
			k8utils.InferencePoolMustHaveCondition(t, s.Client, poolSecondaryNN, acceptedCondition)
			t.Logf("InferencePool %s parent status shows Accepted by Gateway %s (via HTTPRoute %s)", poolSecondaryNN.String(), gatewayNN.String(), routeNN.String())
		})
	},
}
