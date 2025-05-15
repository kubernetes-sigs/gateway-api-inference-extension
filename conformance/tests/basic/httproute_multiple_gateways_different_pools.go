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
	"sigs.k8s.io/gateway-api/conformance/utils/suite"

	// Import the tests package to append to ConformanceTests
	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	"sigs.k8s.io/gateway-api-inference-extension/conformance/utils/config"
	infrakubernetes "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
	gatewaykubernetes "sigs.k8s.io/gateway-api/conformance/utils/kubernetes"
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
			appBackendNamespace = "gateway-conformance-app-backend"
			infraNamespace      = "gateway-conformance-infra"
			gateway1Name        = "conformance-gateway"
			gateway2Name        = "conformance-secondary-gateway"
			routeForGW1Name     = "route-for-gw1"
			routeForGW2Name     = "route-for-gw2"
			poolAName           = "pool-a"
			poolBName           = "pool-b"
		)

		routeForGW1NN := types.NamespacedName{Name: routeForGW1Name, Namespace: appBackendNamespace}
		routeForGW2NN := types.NamespacedName{Name: routeForGW2Name, Namespace: appBackendNamespace}
		poolANN := types.NamespacedName{Name: poolAName, Namespace: appBackendNamespace}
		poolBNN := types.NamespacedName{Name: poolBName, Namespace: appBackendNamespace}
		gateway1NN := types.NamespacedName{Name: gateway1Name, Namespace: infraNamespace}
		gateway2NN := types.NamespacedName{Name: gateway2Name, Namespace: infraNamespace}

		var timeoutConfig config.InferenceExtensionTimeoutConfig = config.DefaultInferenceExtensionTimeoutConfig()

		t.Run("HTTPRoute for Gateway 1 should be Accepted and Reconciled", func(t *testing.T) {
			acceptedCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionAccepted),
				Status: metav1.ConditionTrue,
				Reason: string(gatewayv1.RouteReasonAccepted),
			}
			gatewaykubernetes.HTTPRouteMustHaveCondition(t, s.Client, timeoutConfig.TimeoutConfig, routeForGW1NN, gateway1NN, acceptedCondition)
			t.Logf("HTTPRoute %s is Accepted by Gateway %s", routeForGW1NN.String(), gateway1NN.String())

			reconciledCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionType("Reconciled")),
				Status: metav1.ConditionTrue,
				Reason: "ReconciliationSucceeded",
			}
			gatewaykubernetes.HTTPRouteMustHaveCondition(t, s.Client, timeoutConfig.TimeoutConfig, routeForGW1NN, gateway1NN, reconciledCondition)
			t.Logf("HTTPRoute %s is Reconciled by Gateway %s", routeForGW1NN.String(), gateway1NN.String())
		})

		t.Run("InferencePool A (pool-a) should be Accepted", func(t *testing.T) {
			acceptedCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionAccepted),
				Status: metav1.ConditionTrue,
				Reason: string(gatewayv1.RouteReasonAccepted),
			}
			infrakubernetes.InferencePoolMustHaveCondition(t, s.Client, poolANN, acceptedCondition)
			t.Logf("InferencePool %s parent status shows Accepted by Gateway %s (via HTTPRoute %s)", poolANN.String(), gateway1NN.String(), routeForGW1NN.String())
		})

		t.Run("HTTPRoute for Gateway 2 should be Accepted and Reconciled", func(t *testing.T) {
			acceptedCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionAccepted),
				Status: metav1.ConditionTrue,
				Reason: string(gatewayv1.RouteReasonAccepted),
			}
			gatewaykubernetes.HTTPRouteMustHaveCondition(t, s.Client, timeoutConfig.TimeoutConfig, routeForGW2NN, gateway2NN, acceptedCondition)
			t.Logf("HTTPRoute %s is Accepted by Gateway %s", routeForGW2NN.String(), gateway2NN.String())

			reconciledCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionType("Reconciled")),
				Status: metav1.ConditionTrue,
				Reason: "ReconciliationSucceeded",
			}
			gatewaykubernetes.HTTPRouteMustHaveCondition(t, s.Client, timeoutConfig.TimeoutConfig, routeForGW2NN, gateway2NN, reconciledCondition)
			t.Logf("HTTPRoute %s is Reconciled by Gateway %s", routeForGW2NN.String(), gateway2NN.String())
		})

		t.Run("InferencePool B (pool-b) should be Accepted", func(t *testing.T) {
			acceptedCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionAccepted),
				Status: metav1.ConditionTrue,
				Reason: string(gatewayv1.RouteReasonAccepted),
			}
			infrakubernetes.InferencePoolMustHaveCondition(t, s.Client, poolBNN, acceptedCondition)
			t.Logf("InferencePool %s parent status shows Accepted by Gateway %s (via HTTPRoute %s)", poolBNN.String(), gateway2NN.String(), routeForGW2NN.String())
		})
	},
}
