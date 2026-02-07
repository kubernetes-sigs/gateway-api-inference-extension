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

package controller

import (
	"context"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	poolutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/pool"
	utiltest "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

var (
	inferencePool = utiltest.MakeInferencePool("test-pool1").Namespace("ns1").ObjRef()
	infObjective1 = utiltest.MakeInferenceObjective("model1").
			Namespace(inferencePool.Namespace).
			Priority(1).
			CreationTimestamp(metav1.Unix(1000, 0)).
			PoolName(inferencePool.Name).
			PoolGroup("inference.networking.k8s.io").ObjRef()
	infObjective1Pool2 = utiltest.MakeInferenceObjective(infObjective1.Name).
				Namespace(infObjective1.Namespace).
				Priority(*infObjective1.Spec.Priority).
				CreationTimestamp(metav1.Unix(1001, 0)).
				PoolName("test-pool2").
				PoolGroup("inference.networking.k8s.io").ObjRef()
	infObjective1Critical = utiltest.MakeInferenceObjective(infObjective1.Name).
				Namespace(infObjective1.Namespace).
				Priority(2).
				CreationTimestamp(metav1.Unix(1003, 0)).
				PoolName(inferencePool.Name).
				PoolGroup("inference.networking.k8s.io").ObjRef()
	infObjective1Deleted = utiltest.MakeInferenceObjective(infObjective1.Name).
				Namespace(infObjective1.Namespace).
				CreationTimestamp(metav1.Unix(1004, 0)).
				DeletionTimestamp().
				PoolName(inferencePool.Name).
				PoolGroup("inference.networking.k8s.io").ObjRef()
	infObjective1DiffGroup = utiltest.MakeInferenceObjective(infObjective1.Name).
				Namespace(inferencePool.Namespace).
				Priority(1).
				CreationTimestamp(metav1.Unix(1005, 0)).
				PoolName(inferencePool.Name).
				PoolGroup("inference.networking.x-k8s.io").ObjRef()
	infObjective2 = utiltest.MakeInferenceObjective("model2").
			Namespace(inferencePool.Namespace).
			CreationTimestamp(metav1.Unix(1000, 0)).
			PoolName(inferencePool.Name).
			PoolGroup("inference.networking.k8s.io").ObjRef()

	infPool = utiltest.MakeInferencePool("test-alpha-pool").
		Namespace("ns1").
		Labels(map[string]string{
			"graduation": "beta",
			"plan":       "plus",
		}).
		ObjRef()
)

func TestInferenceObjectiveReconciler(t *testing.T) {
	tests := []struct {
		name                  string
		objectivessInStore    []*v1alpha2.InferenceObjective
		objectivesInAPIServer []*v1alpha2.InferenceObjective
		objective             *v1alpha2.InferenceObjective
		incomingReq           *types.NamespacedName
		wantObjectives        []*v1alpha2.InferenceObjective
		wantResult            ctrl.Result
	}{
		{
			name:           "Empty store, add new objective",
			objective:      infObjective1,
			wantObjectives: []*v1alpha2.InferenceObjective{infObjective1},
		},
		{
			name:               "Existing objective changed pools",
			objectivessInStore: []*v1alpha2.InferenceObjective{infObjective1},
			objective:          infObjective1Pool2,
			wantObjectives:     []*v1alpha2.InferenceObjective{},
		},
		{
			name:               "Not found, delete existing objective",
			objectivessInStore: []*v1alpha2.InferenceObjective{infObjective1},
			incomingReq:        &types.NamespacedName{Name: infObjective1.Name, Namespace: infObjective1.Namespace},
			wantObjectives:     []*v1alpha2.InferenceObjective{},
		},
		{
			name:               "Deletion timestamp set, delete existing objective",
			objectivessInStore: []*v1alpha2.InferenceObjective{infObjective1},
			objective:          infObjective1Deleted,
			wantObjectives:     []*v1alpha2.InferenceObjective{},
		},
		{
			name:               "Objective changed priority",
			objectivessInStore: []*v1alpha2.InferenceObjective{infObjective1},
			objective:          infObjective1Critical,
			wantObjectives:     []*v1alpha2.InferenceObjective{infObjective1Critical},
		},
		{
			name:               "Objective not found, no matching existing objective to delete",
			objectivessInStore: []*v1alpha2.InferenceObjective{infObjective1},
			incomingReq:        &types.NamespacedName{Name: "non-existent-objective", Namespace: inferencePool.Namespace},
			wantObjectives:     []*v1alpha2.InferenceObjective{infObjective1},
		},
		{
			name:               "Add to existing",
			objectivessInStore: []*v1alpha2.InferenceObjective{infObjective1},
			objective:          infObjective2,
			wantObjectives:     []*v1alpha2.InferenceObjective{infObjective1, infObjective2},
		},
		{
			name:               "Objective deleted due to group mismatch for the inference inferencePool",
			objectivessInStore: []*v1alpha2.InferenceObjective{infObjective1},
			objective:          infObjective1DiffGroup,
			wantObjectives:     []*v1alpha2.InferenceObjective{},
		},
		{
			name:           "Objective ignored due to group mismatch for the inference inferencePool",
			objective:      infObjective1DiffGroup,
			wantObjectives: []*v1alpha2.InferenceObjective{},
		},
	}
	for _, test := range tests {
		period := time.Second
		factories := []datalayer.EndpointFactory{
			backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, period),
			datalayer.NewEndpointFactory([]fwkdl.DataSource{&datalayer.FakeDataSource{}}, period),
		}
		for _, epf := range factories {
			t.Run(test.name, func(t *testing.T) {
				// Create a fake client with no InferenceObjective objects.
				scheme := runtime.NewScheme()
				_ = clientgoscheme.AddToScheme(scheme)
				_ = v1alpha2.Install(scheme)
				_ = v1.Install(scheme)
				initObjs := []client.Object{}
				if test.objective != nil {
					initObjs = append(initObjs, test.objective)
				}
				for _, m := range test.objectivesInAPIServer {
					initObjs = append(initObjs, m)
				}
				fakeClient := fake.NewClientBuilder().
					WithScheme(scheme).
					WithObjects(initObjs...).
					Build()
				ds := datastore.NewDatastore(t.Context(), epf, 0)
				for _, m := range test.objectivessInStore {
					ds.ObjectiveSet(m)
				}
				endpointPool := poolutil.InferencePoolToEndpointPool(inferencePool)
				_ = ds.PoolSet(context.Background(), fakeClient, endpointPool)
				reconciler := &InferenceObjectiveReconciler{
					Reader:    fakeClient,
					Datastore: ds,
					PoolGKNN: common.GKNN{
						NamespacedName: types.NamespacedName{Name: inferencePool.Name, Namespace: inferencePool.Namespace},
						GroupKind:      schema.GroupKind{Group: inferencePool.GroupVersionKind().Group, Kind: inferencePool.GroupVersionKind().Kind},
					},
				}
				if test.incomingReq == nil {
					test.incomingReq = &types.NamespacedName{Name: test.objective.Name, Namespace: test.objective.Namespace}
				}

				// Call Reconcile.
				result, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: *test.incomingReq})
				if err != nil {
					t.Fatalf("expected no error when resource is not found, got %v", err)
				}

				if diff := cmp.Diff(result, test.wantResult); diff != "" {
					t.Errorf("Unexpected result diff (+got/-want): %s", diff)
				}

				if len(test.wantObjectives) != len(ds.ObjectiveGetAll()) {
					t.Errorf("Unexpected; want: %d, got:%d", len(test.wantObjectives), len(ds.ObjectiveGetAll()))
				}
				if diff := diffStore(ds, diffStoreParams{wantPool: endpointPool, wantObjectives: test.wantObjectives}); diff != "" {
					t.Errorf("Unexpected diff (+got/-want): %s", diff)
				}

			})
		}
	}
}

func TestInferenceObjectiveReconcilerWithPoolSelector(t *testing.T) {
	// Create an objective with poolSelector matching the pool's labels
	infObjectiveWithSelector := &v1alpha2.InferenceObjective{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "obj-with-selector",
			Namespace:         infPool.Namespace,
			CreationTimestamp: metav1.Unix(1000, 0),
		},
		Spec: v1alpha2.InferenceObjectiveSpec{
			Priority: ptr.To(5),
			PoolSelector: &v1alpha2.PoolSelector{
				Group: v1.GroupName,
				Kind:  "InferencePool",
				MatchLabels: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
					"graduation": "beta",
				},
			},
		},
	}

	// Create an objective with poolSelector that doesn't match
	infObjectiveNonMatchingSelector := &v1alpha2.InferenceObjective{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "obj-non-matching",
			Namespace:         infPool.Namespace,
			CreationTimestamp: metav1.Unix(1000, 0),
		},
		Spec: v1alpha2.InferenceObjectiveSpec{
			Priority: ptr.To(3),
			PoolSelector: &v1alpha2.PoolSelector{
				Group: v1.GroupName,
				Kind:  "InferencePool",
				MatchLabels: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
					"graduation": "ga", // Does not match pool's "beta" label
				},
			},
		},
	}

	// Create an objective with matchExpressions
	infObjectiveWithExpressions := &v1alpha2.InferenceObjective{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "obj-with-expressions",
			Namespace:         infPool.Namespace,
			CreationTimestamp: metav1.Unix(1000, 0),
		},
		Spec: v1alpha2.InferenceObjectiveSpec{
			Priority: ptr.To(7),
			PoolSelector: &v1alpha2.PoolSelector{
				Group: v1.GroupName,
				Kind:  "InferencePool",
				MatchExpressions: []v1alpha2.LabelSelectorRequirement{
					{
						Key:      "plan",
						Operator: v1alpha2.LabelSelectorOpIn,
						Values:   []v1alpha2.LabelValue{"plus", "pro"},
					},
				},
			},
		},
	}

	tests := []struct {
		name           string
		objective      *v1alpha2.InferenceObjective
		wantObjectives []*v1alpha2.InferenceObjective
	}{
		{
			name:           "Objective with matching poolSelector is added",
			objective:      infObjectiveWithSelector,
			wantObjectives: []*v1alpha2.InferenceObjective{infObjectiveWithSelector},
		},
		{
			name:           "Objective with non-matching poolSelector is not added",
			objective:      infObjectiveNonMatchingSelector,
			wantObjectives: []*v1alpha2.InferenceObjective{},
		},
		{
			name:           "Objective with matching matchExpressions is added",
			objective:      infObjectiveWithExpressions,
			wantObjectives: []*v1alpha2.InferenceObjective{infObjectiveWithExpressions},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			period := time.Second
			epf := backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, period)

			// Create a fake client with the pool and objective
			scheme := runtime.NewScheme()
			_ = clientgoscheme.AddToScheme(scheme)
			_ = v1alpha2.Install(scheme)
			_ = v1.Install(scheme)

			initObjs := []client.Object{infPool, test.objective}
			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(initObjs...).
				Build()

			ds := datastore.NewDatastore(t.Context(), epf, 0)

			reconciler := &InferenceObjectiveReconciler{
				Reader:    fakeClient,
				Datastore: ds,
				PoolGKNN: common.GKNN{
					NamespacedName: types.NamespacedName{
						Name:      infPool.Name,
						Namespace: infPool.Namespace,
					},
					GroupKind: schema.GroupKind{
						Group: v1.GroupName,
						Kind:  "InferencePool",
					},
				},
			}

			// Call Reconcile
			req := ctrl.Request{
				NamespacedName: types.NamespacedName{
					Name:      test.objective.Name,
					Namespace: test.objective.Namespace,
				},
			}
			_, err := reconciler.Reconcile(context.Background(), req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Check the datastore
			gotObjectives := ds.ObjectiveGetAll()
			if len(gotObjectives) != len(test.wantObjectives) {
				t.Errorf("unexpected number of objectives; want: %d, got: %d", len(test.wantObjectives), len(gotObjectives))
			}

			if len(test.wantObjectives) > 0 {
				got := ds.ObjectiveGet(test.objective.Name)
				if got == nil {
					t.Errorf("expected objective %s to be in datastore", test.objective.Name)
				}
			}
		})
	}
}

func TestPoolSelectorToLabelSelector(t *testing.T) {
	tests := []struct {
		name          string
		selector      *v1alpha2.PoolSelector
		poolLabels    map[string]string
		expectedMatch bool
		expectError   bool
	}{
		{
			name: "matchLabels matches",
			selector: &v1alpha2.PoolSelector{
				MatchLabels: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
					"env": "prod",
				},
			},
			poolLabels:    map[string]string{"env": "prod", "tier": "backend"},
			expectedMatch: true,
		},
		{
			name: "matchLabels does not match",
			selector: &v1alpha2.PoolSelector{
				MatchLabels: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
					"env": "staging",
				},
			},
			poolLabels:    map[string]string{"env": "prod"},
			expectedMatch: false,
		},
		{
			name: "matchExpressions In operator matches",
			selector: &v1alpha2.PoolSelector{
				MatchExpressions: []v1alpha2.LabelSelectorRequirement{
					{
						Key:      "tier",
						Operator: v1alpha2.LabelSelectorOpIn,
						Values:   []v1alpha2.LabelValue{"frontend", "backend"},
					},
				},
			},
			poolLabels:    map[string]string{"tier": "backend"},
			expectedMatch: true,
		},
		{
			name: "matchExpressions In operator does not match",
			selector: &v1alpha2.PoolSelector{
				MatchExpressions: []v1alpha2.LabelSelectorRequirement{
					{
						Key:      "tier",
						Operator: v1alpha2.LabelSelectorOpIn,
						Values:   []v1alpha2.LabelValue{"frontend", "backend"},
					},
				},
			},
			poolLabels:    map[string]string{"tier": "database"},
			expectedMatch: false,
		},
		{
			name: "matchExpressions NotIn operator matches",
			selector: &v1alpha2.PoolSelector{
				MatchExpressions: []v1alpha2.LabelSelectorRequirement{
					{
						Key:      "env",
						Operator: v1alpha2.LabelSelectorOpNotIn,
						Values:   []v1alpha2.LabelValue{"dev", "staging"},
					},
				},
			},
			poolLabels:    map[string]string{"env": "prod"},
			expectedMatch: true,
		},
		{
			name: "matchExpressions Exists operator matches",
			selector: &v1alpha2.PoolSelector{
				MatchExpressions: []v1alpha2.LabelSelectorRequirement{
					{
						Key:      "feature-flag",
						Operator: v1alpha2.LabelSelectorOpExists,
					},
				},
			},
			poolLabels:    map[string]string{"feature-flag": "enabled"},
			expectedMatch: true,
		},
		{
			name: "matchExpressions Exists operator does not match",
			selector: &v1alpha2.PoolSelector{
				MatchExpressions: []v1alpha2.LabelSelectorRequirement{
					{
						Key:      "feature-flag",
						Operator: v1alpha2.LabelSelectorOpExists,
					},
				},
			},
			poolLabels:    map[string]string{"other-label": "value"},
			expectedMatch: false,
		},
		{
			name: "matchExpressions DoesNotExist operator matches",
			selector: &v1alpha2.PoolSelector{
				MatchExpressions: []v1alpha2.LabelSelectorRequirement{
					{
						Key:      "deprecated",
						Operator: v1alpha2.LabelSelectorOpDoesNotExist,
					},
				},
			},
			poolLabels:    map[string]string{"env": "prod"},
			expectedMatch: true,
		},
		{
			name: "combined matchLabels and matchExpressions",
			selector: &v1alpha2.PoolSelector{
				MatchLabels: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
					"env": "prod",
				},
				MatchExpressions: []v1alpha2.LabelSelectorRequirement{
					{
						Key:      "tier",
						Operator: v1alpha2.LabelSelectorOpIn,
						Values:   []v1alpha2.LabelValue{"frontend", "backend"},
					},
				},
			},
			poolLabels:    map[string]string{"env": "prod", "tier": "backend"},
			expectedMatch: true,
		},
		{
			name:          "empty selector matches everything",
			selector:      &v1alpha2.PoolSelector{},
			poolLabels:    map[string]string{"any": "label"},
			expectedMatch: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			labelSelector, err := poolSelectorToLabelSelector(test.selector)
			if test.expectError {
				if err == nil {
					t.Fatalf("expected error but got none")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			matches := labelSelector.Matches(labels.Set(test.poolLabels))
			if matches != test.expectedMatch {
				t.Errorf("expected match=%v, got match=%v", test.expectedMatch, matches)
			}
		})
	}
}
