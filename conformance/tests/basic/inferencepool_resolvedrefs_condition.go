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

	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	k8sutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
	gatewayk8sutils "sigs.k8s.io/gateway-api/conformance/utils/kubernetes"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, InferencePoolParentStatus)
}

var InferencePoolParentStatus = suite.ConformanceTest{
	ShortName:   "InferencePoolResolvedRefsCondition",
	Description: "Verify that an InferencePool correctly updates its parent-specific status (e.g., Accepted condition) when referenced by HTTPRoutes attached to shared Gateways, and clears parent statuses when no longer referenced.",
	Manifests:   []string{"tests/basic/inferencepool_resolvedrefs_condition.yaml"},
	Features:    []features.FeatureName{},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			appBackendNamespace        = "gateway-conformance-app-backend"
			infraNamespace             = "gateway-conformance-infra"
			poolName                   = "multi-gateway-pool"
			sharedGateway1Name         = "conformance-gateway"
			sharedGateway2Name         = "conformance-secondary-gateway"
			httpRoute1Name             = "httproute-for-gw1"
			httpRoute2Name             = "httproute-for-gw2"
			reasonPoolAcceptedByParent = "Accepted"
		)

		poolNN := types.NamespacedName{Name: poolName, Namespace: appBackendNamespace}
		httpRoute1NN := types.NamespacedName{Name: httpRoute1Name, Namespace: appBackendNamespace}
		httpRoute2NN := types.NamespacedName{Name: httpRoute2Name, Namespace: appBackendNamespace}
		gateway1NN := types.NamespacedName{Name: sharedGateway1Name, Namespace: infraNamespace}
		gateway2NN := types.NamespacedName{Name: sharedGateway2Name, Namespace: infraNamespace}

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

		t.Logf("Waiting for HTTPRoute %s to be Accepted and have ResolvedRefs by Gateway %s", httpRoute1NN.String(), gateway1NN.String())
		gatewayk8sutils.HTTPRouteMustHaveCondition(t, s.Client, s.TimeoutConfig, httpRoute1NN, gateway1NN, routeAcceptedCondition)
		gatewayk8sutils.HTTPRouteMustHaveCondition(t, s.Client, s.TimeoutConfig, httpRoute1NN, gateway1NN, routeResolvedRefsCondition)

		t.Logf("Waiting for HTTPRoute %s to be Accepted and have ResolvedRefs by Gateway %s", httpRoute2NN.String(), gateway2NN.String())
		gatewayk8sutils.HTTPRouteMustHaveCondition(t, s.Client, s.TimeoutConfig, httpRoute2NN, gateway2NN, routeAcceptedCondition)
		gatewayk8sutils.HTTPRouteMustHaveCondition(t, s.Client, s.TimeoutConfig, httpRoute2NN, gateway2NN, routeResolvedRefsCondition)

		t.Run("InferencePool should show Accepted:True by parents when referenced by multiple HTTPRoutes", func(t *testing.T) {
			expectedCondition := metav1.Condition{
				Type:   string(gatewayv1.GatewayConditionAccepted),
				Status: metav1.ConditionTrue,
				Reason: reasonPoolAcceptedByParent,
			}
			k8sutils.InferencePoolMustHaveCondition(t, s.Client, poolNN, expectedCondition)
			t.Logf("InferencePool %s has parent status Accepted:True as expected with two references.", poolNN.String())
		})

		t.Run("Delete httproute-for-gw1", func(t *testing.T) {
			httproute1 := &gatewayv1.HTTPRoute{
				ObjectMeta: metav1.ObjectMeta{Name: httpRoute1NN.Name, Namespace: httpRoute1NN.Namespace},
			}
			t.Logf("Deleting HTTPRoute %s", httpRoute1NN.String())
			require.NoError(t, s.Client.Delete(context.TODO(), httproute1), "failed to delete httproute-for-gw1")
			time.Sleep(s.TimeoutConfig.GatewayMustHaveCondition)
		})

		t.Run("InferencePool should still show Accepted:True by parent after one HTTPRoute is deleted", func(t *testing.T) {
			expectedCondition := metav1.Condition{
				Type:   string(gatewayv1.GatewayConditionAccepted),
				Status: metav1.ConditionTrue,
				Reason: reasonPoolAcceptedByParent,
			}
			k8sutils.InferencePoolMustHaveCondition(t, s.Client, poolNN, expectedCondition)
			t.Logf("InferencePool %s still has parent status Accepted:True as expected with one reference remaining.", poolNN.String())
		})

		t.Run("Delete httproute-for-gw2", func(t *testing.T) {
			httproute2 := &gatewayv1.HTTPRoute{
				ObjectMeta: metav1.ObjectMeta{Name: httpRoute2NN.Name, Namespace: httpRoute2NN.Namespace},
			}
			t.Logf("Deleting HTTPRoute %s", httpRoute2NN.String())
			require.NoError(t, s.Client.Delete(context.TODO(), httproute2), "failed to delete httproute-for-gw2")
		})

		t.Run("InferencePool should have no parent statuses after all HTTPRoutes are deleted", func(t *testing.T) {
			t.Logf("Waiting for InferencePool %s to have no parent statuses.", poolNN.String())
			k8sutils.InferencePoolMustHaveNoParents(t, s.Client, poolNN)
			t.Logf("InferencePool %s correctly shows no parent statuses, indicating it's no longer referenced.", poolNN.String())
		})

		t.Logf("InferencePoolResolvedRefsCondition completed.")
	},
}
