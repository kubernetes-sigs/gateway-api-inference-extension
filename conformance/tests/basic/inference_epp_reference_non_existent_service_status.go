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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	inferenceapi "sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	infrakubernetes "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
	gatewaykubernetes "sigs.k8s.io/gateway-api/conformance/utils/kubernetes"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, InferencePoolEPPReferenceNonExistentServiceStatus)
}

// InferencePoolEPPReferenceNonExistentServiceStatus defines the test case for verifying
// HTTPRoute status when it references an InferencePool whose extensionRef points to a non-existent EPP service.
var InferencePoolEPPReferenceNonExistentServiceStatus = suite.ConformanceTest{
	ShortName:   "InferencePoolEPPReferenceNonExistentServiceStatus",
	Description: "Validate HTTPRoute status reports an error when referencing an InferencePool with a non-existent EPP service extensionRef.",
	Manifests:   []string{"tests/basic/inference_epp_reference_non_existent_service_status.yaml"},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		poolNN := types.NamespacedName{
			Name:      "pool-non-existent-epp",
			Namespace: "gateway-conformance-app-backend",
		}
		routeNN := types.NamespacedName{
			Name:      "httproute-for-pool-non-existent-epp",
			Namespace: "gateway-conformance-app-backend",
		}
		gatewayNN := types.NamespacedName{
			Name:      "conformance-gateway",       // As defined in shared manifests
			Namespace: "gateway-conformance-infra", // As defined in shared manifests
		}

		// Step 1: Ensure the HTTPRoute is accepted by the Gateway. This is a prerequisite
		// for the controller to process the backendRefs and potentially report errors.
		httpRouteAcceptedCondition := metav1.Condition{
			Type:   string(gatewayv1.RouteConditionAccepted),
			Status: metav1.ConditionTrue,
			Reason: string(gatewayv1.RouteReasonAccepted),
		}

		// Increase the timeout for HTTPRoute conditions
		testTimeoutConfig := s.TimeoutConfig
		testTimeoutConfig.HTTPRouteMustHaveCondition = 5 * time.Minute

		t.Logf("Waiting for HTTPRoute %s to be accepted by Gateway %s", routeNN.String(), gatewayNN.String())
		gatewaykubernetes.HTTPRouteMustHaveCondition(t, s.Client, testTimeoutConfig, routeNN, gatewayNN, httpRouteAcceptedCondition)
		t.Logf("HTTPRoute %s is Accepted by Gateway %s", routeNN.String(), gatewayNN.String())

		// Step 2: Observe the status of the HTTPRoute.
		// Expected: The HTTPRoute's 'Reconciled' condition should be False
		// because the backend (InferencePool with non-existent EPP service) cannot be reconciled.
		httpRouteReconciledCondition := metav1.Condition{
			Type:   "Reconciled",
			Status: metav1.ConditionFalse,
			Reason: "ReconciliationFailed",
		}

		t.Logf("Waiting for HTTPRoute %s to have condition: Type=%s, Status=%s, Reason=%s",
			routeNN.String(), httpRouteReconciledCondition.Type, httpRouteReconciledCondition.Status, httpRouteReconciledCondition.Reason)

		// The reason and status should indicate the failure.
		gatewaykubernetes.HTTPRouteMustHaveCondition(t, s.Client, s.TimeoutConfig, routeNN, gatewayNN, httpRouteReconciledCondition)

		t.Logf("Successfully verified HTTPRoute %s has Type:%s Status:%s with Reason:%s due to non-existent EPP service",
			routeNN.String(), httpRouteReconciledCondition.Type, httpRouteReconciledCondition.Status, httpRouteReconciledCondition.Reason)

		// step3: Verify InferencePool status remains 'Accepted:True' and doesn't show specific EPP error,
		// as per the observed behavior and the linked issue #806.
		inferencePoolAcceptedCondition := metav1.Condition{
			Type:   string(inferenceapi.InferencePoolConditionAccepted),
			Status: metav1.ConditionTrue,
			Reason: string(inferenceapi.InferencePoolReasonAccepted),
		}
		t.Logf("Verifying InferencePool %s remains Accepted: True", poolNN.String())
		infrakubernetes.InferencePoolMustHaveCondition(t, s.Client, poolNN, inferencePoolAcceptedCondition)
		t.Logf("InferencePool %s remains Accepted: True, confirming error propagates to HTTPRoute", poolNN.String())
	},
}
