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

// TODO(#864) refactor the structure to put all tests directly under tests instead of creating subfolders.
package basic

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
	"sigs.k8s.io/gateway-api/conformance/utils/kubernetes"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, HTTPRouteInvalidInferencePoolRef)
}

var HTTPRouteInvalidInferencePoolRef = suite.ConformanceTest{
	ShortName:   "HTTPRouteInvalidInferencePoolRef",
	Description: "Tests HTTPRoute status when it references an InferencePool that does not exist.",
	Manifests:   []string{"tests/basic/httproute_invalid_inferencepool_ref.yaml"},
	Features: []features.FeatureName{
		features.FeatureName("SupportInferencePool"),
		features.SupportGateway,
	},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			appBackendNamespace = "gateway-conformance-app-backend"
			infraNamespace      = "gateway-conformance-infra"
			routeName           = "httproute-to-non-existent-pool"
			gatewayName         = "conformance-gateway"
		)
		routeNN := types.NamespacedName{Name: routeName, Namespace: appBackendNamespace}
		gatewayNN := types.NamespacedName{Name: gatewayName, Namespace: infraNamespace}

		t.Run("HTTPRoute should have Accepted=True and ResolvedRefs=False for non-existent InferencePool", func(t *testing.T) {
			acceptedCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionAccepted),
				Status: metav1.ConditionTrue,
				Reason: string(gatewayv1.RouteReasonAccepted),
			}
			kubernetes.HTTPRouteMustHaveCondition(t, s.Client, s.TimeoutConfig, routeNN, gatewayNN, acceptedCondition)

			resolvedRefsCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionResolvedRefs),
				Status: metav1.ConditionFalse,
				Reason: string(gatewayv1.RouteReasonBackendNotFound),
			}
			kubernetes.HTTPRouteMustHaveCondition(t, s.Client, s.TimeoutConfig, routeNN, gatewayNN, resolvedRefsCondition)

			t.Logf("Successfully verified HTTPRoute %s has conditions: Accepted=True and ResolvedRefs=False (Reason: BackendNotFound) for Gateway %s",
				routeNN.String(), gatewayNN.String())
		})
	},
}
