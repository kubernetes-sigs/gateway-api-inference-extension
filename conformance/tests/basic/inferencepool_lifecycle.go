/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUTHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package basic

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features"

	inferenceapi "sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	infrakubernetes "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, InferencePoolLifecycle)
}

var InferencePoolLifecycle = suite.ConformanceTest{
	ShortName:   "InferencePoolLifecycle",
	Description: "Tests basic CRUD (Create, Read, Update, Delete) operations for an InferencePool resource, assuming current API definition.",
	Manifests:   []string{"tests/basic/inferencepool_lifecycle.yaml"},
	Features:    []features.FeatureName{},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		poolName := "inferencepool-lifecycle"

		const hardcodedNamespace = "gateway-conformance-app-backend"
		poolNN := types.NamespacedName{
			Name:      poolName,
			Namespace: hardcodedNamespace,
		}

		acceptedCondition := metav1.Condition{
			Type:   string(gatewayv1.GatewayConditionAccepted),
			Status: metav1.ConditionTrue,
			Reason: "",
		}

		t.Run("Step 1 & 2: Create and Read InferencePool", func(t *testing.T) {
			infrakubernetes.InferencePoolMustHaveCondition(t, s.Client, poolNN, acceptedCondition)

			createdPool := &inferenceapi.InferencePool{}
			err := s.Client.Get(context.Background(), poolNN, createdPool)
			require.NoErrorf(t, err, "Failed to get InferencePool %s/%s", poolNN.Namespace, poolNN.Name)

			// Verify fields based on the current API (using .Spec.Selector)
			// TODO: https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/766
			// API Discrepancy: InferencePoolSpec pod selector field name mismatch between API
			// Proposal 002 and current Go type definition
			require.NotNil(t, createdPool.Spec.Selector, "Selector should not be nil")
			require.Equal(t, inferenceapi.LabelValue("infra-backend-v1"), createdPool.Spec.Selector[inferenceapi.LabelKey("app")],
				"Selector 'app' label does not match manifest")

			t.Logf("Successfully created and read InferencePool %s/%s", poolNN.Namespace, poolNN.Name)
		})

		t.Run("Step 3: Update InferencePool", func(t *testing.T) {
			originalPool := &inferenceapi.InferencePool{}
			err := s.Client.Get(context.Background(), poolNN, originalPool)
			require.NoErrorf(t, err, "Failed to get InferencePool %s for update", poolNN.String())

			updatedPool := originalPool.DeepCopy()
			newSelector := map[inferenceapi.LabelKey]inferenceapi.LabelValue{
				inferenceapi.LabelKey("app"): inferenceapi.LabelValue("lifecycle-test-app-updated"),
			}
			updatedPool.Spec.Selector = newSelector

			err = s.Client.Update(context.Background(), updatedPool)
			require.NoErrorf(t, err, "Failed to update InferencePool %s", poolNN.String())
			infrakubernetes.InferencePoolMustHaveSelector(t, s.Client, poolNN, newSelector)

			fetchedAfterUpdate := &inferenceapi.InferencePool{}
			err = s.Client.Get(context.Background(), poolNN, fetchedAfterUpdate)
			require.NoErrorf(t, err, "Failed to get InferencePool %s after update verification", poolNN.String())
			require.NotNil(t, fetchedAfterUpdate.Spec.Selector, "Updated Selector should not be nil")
			require.Equal(t, inferenceapi.LabelValue("lifecycle-test-app-updated"), fetchedAfterUpdate.Spec.Selector[inferenceapi.LabelKey("app")],
				"Selector 'app' label was not updated as expected")
			t.Logf("Successfully updated InferencePool %s/%s", poolNN.Namespace, poolNN.Name)
		})

		t.Run("Step 4: Delete InferencePool", func(t *testing.T) {
			t.Logf("Attempting to explicitly delete InferencePool %s/%s for verification", poolNN.Namespace, poolNN.Name)
			poolToDelete := &inferenceapi.InferencePool{
				ObjectMeta: metav1.ObjectMeta{
					Name:      poolName,
					Namespace: hardcodedNamespace,
				},
			}
			err := s.Client.Delete(context.Background(), poolToDelete)
			if err != nil && !apierrors.IsNotFound(err) {
				require.NoErrorf(t, err, "Failed to delete InferencePool %s", poolNN.String())
			}

			infrakubernetes.InferencePoolMustBeDeleted(t, s.Client, poolNN)
			t.Logf("Successfully verified deletion of InferencePool %s/%s", poolNN.Namespace, poolNN.Name)
		})
	},
}
