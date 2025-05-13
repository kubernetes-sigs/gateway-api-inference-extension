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
	"sigs.k8s.io/gateway-api/pkg/features"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	infrakubernetes "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, InferencePoolBackendServiceNoEndpoints)
}

var InferencePoolBackendServiceNoEndpoints = suite.ConformanceTest{
	ShortName:   "InferencePoolBackendServiceNoEndpoints",
	Description: "Validates InferencePool and HTTPRoute status when an InferencePool references a Service with no endpoints.",
	Manifests:   []string{"tests/basic/inferencepool_backend_service_no_endpoints.yaml"},
	Features:    []features.FeatureName{},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			appBackendNamespace           = "gateway-conformance-app-backend"
			infraNamespace                = "gateway-conformance-infra"
			poolName                      = "pool-no-endpoints"
			httpRouteName                 = "httproute-for-pool-no-endpoints"
			gatewayName                   = "conformance-gateway"
			expectedPoolReason            = "NoEndpointsAvailable"
			expectedRouteReason           = "BackendNotReady"
			expectedRouteReconciledReason = "ReconciliationFailed"
		)

		poolNN := types.NamespacedName{Name: poolName, Namespace: appBackendNamespace}
		routeNN := types.NamespacedName{Name: httpRouteName, Namespace: appBackendNamespace}
		gatewayNN := types.NamespacedName{Name: gatewayName, Namespace: infraNamespace}

		t.Run("InferencePool should have Accepted condition set to True", func(t *testing.T) {
			acceptedCondition := metav1.Condition{
				Type:   string(gatewayv1.GatewayConditionAccepted),
				Status: metav1.ConditionTrue,
			}
			infrakubernetes.InferencePoolMustHaveCondition(t, s.Client, poolNN, acceptedCondition)
		})

		t.Run("InferencePool should have ResolvedRefs condition set to True", func(t *testing.T) {
			resolvedRefsCondition := metav1.Condition{
				Type:   string(gatewayv1.GatewayConditionType("ResolvedRefs")),
				Status: metav1.ConditionTrue,
			}
			infrakubernetes.InferencePoolMustHaveCondition(t, s.Client, poolNN, resolvedRefsCondition)
		})

		t.Logf("Polling for HTTPRoute %s status to reflect backend issues...", routeNN.String())
		expectedAcceptedRouteCond := metav1.Condition{
			Type:   string(gatewayv1.RouteConditionAccepted),
			Status: metav1.ConditionTrue,
			Reason: string(gatewayv1.RouteReasonAccepted),
		}

		expectedFailureRouteCond := metav1.Condition{
			Type:   string(gatewayv1.RouteConditionType("Reconciled")),
			Status: metav1.ConditionFalse,
			Reason: expectedRouteReconciledReason,
		}

		expectedFailureMessageSubstrings := []string{
			"missing neg status in annotation",
			"gateway-conformance-app-backend/backend-svc-no-endpoints",
		}

		infrakubernetes.HTTPRouteMustHaveParentStatusConditions(t, s.Client, routeNN, gatewayNN, s.ControllerName,
			expectedAcceptedRouteCond, expectedFailureRouteCond, expectedFailureMessageSubstrings)

		t.Logf("TestInferencePoolBackendServiceNoEndpoints completed successfully.")
	},
}
