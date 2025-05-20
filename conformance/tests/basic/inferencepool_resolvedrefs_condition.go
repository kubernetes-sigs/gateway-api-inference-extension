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
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features"

	// Import the tests package to append to ConformanceTests
	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	k8sutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
	gatewayk8sutils "sigs.k8s.io/gateway-api/conformance/utils/kubernetes"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, InferencePoolResolvedRefsCondition)
}

// InferencePoolResolvedRefsCondition defines the test case for verifying
// that an InferencePool correctly surfaces the "ResolvedRefs" condition type
// as it is referenced by other Gateway API resources.
var InferencePoolResolvedRefsCondition = suite.ConformanceTest{
	ShortName:   "InferencePoolResolvedRefsCondition",
	Description: "Verify that an InferencePool correctly surfaces the 'ResolvedRefs' condition type, indicating whether it is successfully referenced by other Gateway API resources.",
	Manifests:   []string{"tests/basic/inferencepool_resolvedrefs_condition.yaml"},
	Features:    []features.FeatureName{},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			appBackendNamespace = "gateway-conformance-app-backend"
			poolName            = "multi-gateway-pool"
			gateway1Name        = "gateway-1"
			gateway2Name        = "gateway-2"
			httpRoute1Name      = "httproute-for-gw1"
			httpRoute2Name      = "httproute-for-gw2"
			reasonRefsResolved  = string(gatewayv1.RouteReasonResolvedRefs)
			reasonNoRefsFound   = "NoMatchingParent"
		)

		poolNN := types.NamespacedName{Name: poolName, Namespace: appBackendNamespace}
		httpRoute1NN := types.NamespacedName{Name: httpRoute1Name, Namespace: appBackendNamespace}
		httpRoute2NN := types.NamespacedName{Name: httpRoute2Name, Namespace: appBackendNamespace}
		gateway1NN := types.NamespacedName{Name: gateway1Name, Namespace: appBackendNamespace}
		gateway2NN := types.NamespacedName{Name: gateway2Name, Namespace: appBackendNamespace}

		routeAcceptedCondition := metav1.Condition{
			Type:   string(gatewayv1.RouteConditionAccepted),
			Status: metav1.ConditionTrue,
			Reason: string(gatewayv1.RouteReasonAccepted),
		}
		routeResolvedRefsCondition := metav1.Condition{
			Type:   string(gatewayv1.RouteConditionResolvedRefs),
			Status: metav1.ConditionTrue,
			Reason: string(gatewayv1.RouteReasonResolvedRefs),
		}

		t.Logf("Waiting for HTTPRoute %s to be Accepted by Gateway %s", httpRoute1NN.String(), gateway1Name)
		gatewayk8sutils.HTTPRouteMustHaveCondition(t, s.Client, s.TimeoutConfig, httpRoute1NN, gateway1NN, routeAcceptedCondition)
		t.Logf("Waiting for HTTPRoute %s to have ResolvedRefs: True by Gateway %s", httpRoute1NN.String(), gateway1Name)
		gatewayk8sutils.HTTPRouteMustHaveCondition(t, s.Client, s.TimeoutConfig, httpRoute1NN, gateway1NN, routeResolvedRefsCondition)

		t.Logf("Waiting for HTTPRoute %s to be Accepted by Gateway %s", httpRoute2NN.String(), gateway2Name)
		gatewayk8sutils.HTTPRouteMustHaveCondition(t, s.Client, s.TimeoutConfig, httpRoute2NN, gateway2NN, routeAcceptedCondition)
		t.Logf("Waiting for HTTPRoute %s to have ResolvedRefs: True by Gateway %s", httpRoute2NN.String(), gateway2Name)
		gatewayk8sutils.HTTPRouteMustHaveCondition(t, s.Client, s.TimeoutConfig, httpRoute2NN, gateway2NN, routeResolvedRefsCondition)

		t.Run("InferencePool should show ResolvedRefs: True when referenced by multiple HTTPRoutes", func(t *testing.T) {
			expectedCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionResolvedRefs),
				Status: metav1.ConditionTrue,
				Reason: reasonRefsResolved,
			}
			k8sutils.InferencePoolMustHaveCondition(t, s.Client, poolNN, expectedCondition)
			t.Logf("InferencePool %s has ResolvedRefs: True as expected with two references.", poolNN.String())
		})

		t.Run("Delete httproute-for-gw1", func(t *testing.T) {
			httproute1 := &gatewayv1.HTTPRoute{
				ObjectMeta: metav1.ObjectMeta{
					Name:      httpRoute1NN.Name,
					Namespace: httpRoute1NN.Namespace,
				},
			}
			t.Logf("Deleting HTTPRoute %s", httpRoute1NN.String())
			require.NoError(t, s.Client.Delete(context.TODO(), httproute1), "failed to delete httproute-for-gw1")
			time.Sleep(s.TimeoutConfig.GatewayMustHaveCondition)
		})

		t.Run("InferencePool should still show ResolvedRefs: True after one HTTPRoute is deleted", func(t *testing.T) {
			expectedCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionResolvedRefs),
				Status: metav1.ConditionTrue,
				Reason: reasonRefsResolved,
			}
			k8sutils.InferencePoolMustHaveCondition(t, s.Client, poolNN, expectedCondition)
			t.Logf("InferencePool %s still has ResolvedRefs: True as expected with one reference remaining.", poolNN.String())
		})

		t.Run("Delete httproute-for-gw2", func(t *testing.T) {
			httproute2 := &gatewayv1.HTTPRoute{
				ObjectMeta: metav1.ObjectMeta{
					Name:      httpRoute2NN.Name,
					Namespace: httpRoute2NN.Namespace,
				},
			}
			t.Logf("Deleting HTTPRoute %s", httpRoute2NN.String())
			require.NoError(t, s.Client.Delete(context.TODO(), httproute2), "failed to delete httproute-for-gw2")
			time.Sleep(s.TimeoutConfig.GatewayMustHaveCondition)
		})

		t.Run("InferencePool should show ResolvedRefs: False after all HTTPRoutes are deleted", func(t *testing.T) {
			expectedCondition := metav1.Condition{
				Type:   string(gatewayv1.RouteConditionResolvedRefs),
				Status: metav1.ConditionFalse,
				Reason: reasonNoRefsFound,
			}
			k8sutils.InferencePoolMustHaveCondition(t, s.Client, poolNN, expectedCondition)
			t.Logf("InferencePool %s has ResolvedRefs: False as expected with no references.", poolNN.String())
		})

		t.Logf("TestInferencePoolResolvedRefsCondition completed.")
	},
}
