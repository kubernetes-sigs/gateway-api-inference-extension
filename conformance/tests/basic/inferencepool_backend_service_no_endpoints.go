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

	// Import time package for timeouts if needed
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1" // For standard condition types
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features" // For standard feature names

	// Import the tests package to append to ConformanceTests
	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	infrakubernetes "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
)

func init() {
	// Register the InferencePoolBackendServiceNoEndpoints test case with the conformance suite.
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

		routeNN := types.NamespacedName{Name: httpRouteName, Namespace: appBackendNamespace}
		gatewayNN := types.NamespacedName{Name: gatewayName, Namespace: infraNamespace}
		expectedAcceptedRouteCond := metav1.Condition{
			Type:   string(gatewayv1.RouteConditionAccepted),
			Status: metav1.ConditionTrue,
			Reason: string(gatewayv1.RouteReasonAccepted),
		}

		// The HTTPRoute should eventually be reconciled to False due to the backend issue.
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
