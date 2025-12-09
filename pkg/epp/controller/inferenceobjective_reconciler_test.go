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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
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
			datalayer.NewEndpointFactory([]datalayer.DataSource{&datalayer.FakeDataSource{}}, period),
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
