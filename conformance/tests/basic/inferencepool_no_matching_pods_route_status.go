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
	"fmt"
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
	tests.ConformanceTests = append(tests.ConformanceTests, InferencePoolNoMatchingPodsRouteStatus)
}

var InferencePoolNoMatchingPodsRouteStatus = suite.ConformanceTest{
	// TODO: Update based on the outcome of https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/806

	ShortName:   "InferencePoolNoMatchingPodsRouteStatus",
	Description: "Tests HTTPRoute and Gateway status when an HTTPRoute references an InferencePool whose modelServerSelector does not match any running pods.",
	Manifests:   []string{"tests/basic/inferencepool_no_matching_pods_route_status.yaml"},
	Features:    []features.FeatureName{},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			appBackendNamespace   = "gateway-conformance-app-backend"
			infraNamespace        = "gateway-conformance-infra"
			poolName              = "pool-no-pods"
			httpRouteName         = "httproute-for-pool-no-pods"
			gatewayName           = "conformance-gateway"
			expectedRouteReason   = "ReconciliationFailed"
			expectedGatewayReason = "Invalid"
		)

		poolNN := types.NamespacedName{Name: poolName, Namespace: appBackendNamespace}
		routeNN := types.NamespacedName{Name: httpRouteName, Namespace: appBackendNamespace}
		gatewayNN := types.NamespacedName{Name: gatewayName, Namespace: infraNamespace}

		t.Logf("Manifests applied. Waiting for controllers to process InferencePool %s and HTTPRoute %s", poolNN.String(), routeNN.String())

		// Step 1: Verify initial acceptance of the InferencePool by the Gateway (via the HTTPRoute)
		// The InferencePool's .status.parent field should show it's Accepted by the Gateway.
		t.Logf("Verifying InferencePool %s is initially accepted by Gateway %s (via HTTPRoute %s)", poolNN.String(), gatewayNN.String(), routeNN.String())
		acceptedCondition := metav1.Condition{
			Type:   string(gatewayv1.RouteConditionAccepted), // For the route parent status on InferencePool
			Status: metav1.ConditionTrue,
			Reason: string(gatewayv1.RouteReasonAccepted),
		}
		infrakubernetes.InferencePoolMustHaveCondition(t, s.Client, poolNN, acceptedCondition)
		t.Logf("InferencePool %s parent status shows Accepted by Gateway %s", poolNN.String(), gatewayNN.String())

		// Step 2: Observe the status of the HTTPRoute
		// Expect the HTTPRoute to be Accepted but fail Reconciliation due to backend issues.
		t.Logf("Polling for HTTPRoute %s status to reflect backend issues...", routeNN.String())

		expectedAcceptedRouteCond := metav1.Condition{
			Type:   string(gatewayv1.RouteConditionAccepted),
			Status: metav1.ConditionTrue,
		}

		expectedFailureRouteCond := metav1.Condition{
			Type:   string(gatewayv1.RouteConditionType("Reconciled")), // As observed in POC logs
			Status: metav1.ConditionFalse,
			Reason: expectedRouteReason,
		}

		expectedFailureMessageSubstrings := []string{
			"missing Service", // As observed in POC logs
			fmt.Sprintf("%s/%s-epp", appBackendNamespace, poolName),
		}

		infrakubernetes.HTTPRouteMustHaveParentStatusConditions(t, s.Client, routeNN, gatewayNN, s.ControllerName,
			expectedAcceptedRouteCond, expectedFailureRouteCond, expectedFailureMessageSubstrings)

		t.Logf("TestInferencePoolNoMatchingPodsRouteStatus completed.")
	},
}
