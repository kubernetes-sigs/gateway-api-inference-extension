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
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	utiltest "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

var (
	pool      = utiltest.MakeInferencePool("test-pool1").Namespace("ns1").ObjRef()
	infModel1 = utiltest.MakeInferenceModel("model1").
			Namespace(pool.Namespace).
			ModelName("fake model1").
			Criticality(v1alpha2.Standard).
			CreationTimestamp(metav1.Unix(1000, 0)).
			PoolName(pool.Name).ObjRef()
	infModel1Pool2 = utiltest.MakeInferenceModel(infModel1.Name).
			Namespace(infModel1.Namespace).
			ModelName(infModel1.Spec.ModelName).
			Criticality(*infModel1.Spec.Criticality).
			CreationTimestamp(metav1.Unix(1001, 0)).
			PoolName("test-pool2").ObjRef()
	infModel1NS2 = utiltest.MakeInferenceModel(infModel1.Name).
			Namespace("ns2").
			ModelName(infModel1.Spec.ModelName).
			Criticality(*infModel1.Spec.Criticality).
			CreationTimestamp(metav1.Unix(1002, 0)).
			PoolName(pool.Name).ObjRef()
	infModel1Critical = utiltest.MakeInferenceModel(infModel1.Name).
				Namespace(infModel1.Namespace).
				ModelName(infModel1.Spec.ModelName).
				Criticality(v1alpha2.Critical).
				CreationTimestamp(metav1.Unix(1003, 0)).
				PoolName(pool.Name).ObjRef()
	infModel1Deleted = utiltest.MakeInferenceModel(infModel1.Name).
				Namespace(infModel1.Namespace).
				ModelName(infModel1.Spec.ModelName).
				CreationTimestamp(metav1.Unix(1004, 0)).
				DeletionTimestamp().
				PoolName(pool.Name).ObjRef()
	// Same ModelName, different object with newer creation timestamp
	infModel1Newer = utiltest.MakeInferenceModel("model1-newer").
			Namespace(pool.Namespace).
			ModelName("fake model1").
			Criticality(v1alpha2.Standard).
			CreationTimestamp(metav1.Unix(1005, 0)).
			PoolName(pool.Name).ObjRef()
	// Same ModelName, different object with older creation timestamp
	infModel1Older = utiltest.MakeInferenceModel("model1-older").
			Namespace(pool.Namespace).
			ModelName("fake model1").
			Criticality(v1alpha2.Standard).
			CreationTimestamp(metav1.Unix(999, 0)).
			PoolName(pool.Name).ObjRef()

	infModel2 = utiltest.MakeInferenceModel("model2").
			Namespace(pool.Namespace).
			ModelName("fake model2").
			CreationTimestamp(metav1.Unix(1000, 0)).
			PoolName(pool.Name).ObjRef()
)

func TestInferenceModelReconciler(t *testing.T) {
	tests := []struct {
		name              string
		modelsInStore     []*v1alpha2.InferenceModel
		modelsInAPIServer []*v1alpha2.InferenceModel
		model             *v1alpha2.InferenceModel
		incomingReq       *types.NamespacedName
		wantModels        []*v1alpha2.InferenceModel
		wantResult        ctrl.Result
	}{
		{
			name:       "Empty store, add new model",
			model:      infModel1,
			wantModels: []*v1alpha2.InferenceModel{infModel1},
		},
		{
			name:          "Existing model changed pools",
			modelsInStore: []*v1alpha2.InferenceModel{infModel1},
			model:         infModel1Pool2,
			wantModels:    []*v1alpha2.InferenceModel{},
		},
		{
			name:          "Not found, delete existing model",
			modelsInStore: []*v1alpha2.InferenceModel{infModel1},
			incomingReq:   &types.NamespacedName{Name: infModel1.Name, Namespace: infModel1.Namespace},
			wantModels:    []*v1alpha2.InferenceModel{},
		},
		{
			name:          "Deletion timestamp set, delete existing model",
			modelsInStore: []*v1alpha2.InferenceModel{infModel1},
			model:         infModel1Deleted,
			wantModels:    []*v1alpha2.InferenceModel{},
		},
		{
			name:          "Model referencing a different pool, different pool name but same namespace",
			modelsInStore: []*v1alpha2.InferenceModel{infModel1},
			model:         infModel1NS2,
			wantModels:    []*v1alpha2.InferenceModel{infModel1},
		},
		{
			name:              "Existing model changed pools, replaced with another",
			modelsInStore:     []*v1alpha2.InferenceModel{infModel1},
			model:             infModel1Pool2,
			modelsInAPIServer: []*v1alpha2.InferenceModel{infModel1Newer},
			wantModels:        []*v1alpha2.InferenceModel{infModel1Newer},
		},
		{
			name:              "Not found, delete existing model, replaced with another",
			modelsInStore:     []*v1alpha2.InferenceModel{infModel1},
			incomingReq:       &types.NamespacedName{Name: infModel1.Name, Namespace: infModel1.Namespace},
			modelsInAPIServer: []*v1alpha2.InferenceModel{infModel1Newer},
			wantModels:        []*v1alpha2.InferenceModel{infModel1Newer},
		},
		{
			name:              "Deletion timestamp set, delete existing model, replaced with another",
			modelsInStore:     []*v1alpha2.InferenceModel{infModel1},
			model:             infModel1Deleted,
			modelsInAPIServer: []*v1alpha2.InferenceModel{infModel1Newer},
			wantModels:        []*v1alpha2.InferenceModel{infModel1Newer},
		},
		{
			name:          "Older instance of the model observed",
			modelsInStore: []*v1alpha2.InferenceModel{infModel1},
			model:         infModel1Older,
			wantModels:    []*v1alpha2.InferenceModel{infModel1Older},
		},
		{
			name:          "Model changed criticality",
			modelsInStore: []*v1alpha2.InferenceModel{infModel1},
			model:         infModel1Critical,
			wantModels:    []*v1alpha2.InferenceModel{infModel1Critical},
		},
		{
			name:          "Model not found, no matching existing model to delete",
			modelsInStore: []*v1alpha2.InferenceModel{infModel1},
			incomingReq:   &types.NamespacedName{Name: "non-existent-model", Namespace: pool.Namespace},
			wantModels:    []*v1alpha2.InferenceModel{infModel1},
		},
		{
			name:          "Add to existing",
			modelsInStore: []*v1alpha2.InferenceModel{infModel1},
			model:         infModel2,
			wantModels:    []*v1alpha2.InferenceModel{infModel1, infModel2},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Create a fake client with no InferenceModel objects.
			scheme := runtime.NewScheme()
			_ = clientgoscheme.AddToScheme(scheme)
			_ = v1alpha2.Install(scheme)
			initObjs := []client.Object{}
			if test.model != nil {
				initObjs = append(initObjs, test.model)
			}
			for _, m := range test.modelsInAPIServer {
				initObjs = append(initObjs, m)
			}

			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(initObjs...).
				WithIndex(&v1alpha2.InferenceModel{}, datastore.ModelNameIndexKey, indexInferenceModelsByModelName).
				Build()
			pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, time.Second)
			ds := datastore.NewDatastore(t.Context(), pmf)
			for _, m := range test.modelsInStore {
				ds.ModelSetIfOlder(m)
			}
			_ = ds.PoolSet(context.Background(), fakeClient, pool)
			reconciler := &InferenceModelReconciler{
				Client:             fakeClient,
				Record:             record.NewFakeRecorder(10),
				Datastore:          ds,
				PoolNamespacedName: types.NamespacedName{Name: pool.Name, Namespace: pool.Namespace},
			}
			if test.incomingReq == nil {
				test.incomingReq = &types.NamespacedName{Name: test.model.Name, Namespace: test.model.Namespace}
			}

			// Call Reconcile.
			result, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: *test.incomingReq})
			if err != nil {
				t.Fatalf("expected no error when resource is not found, got %v", err)
			}

			if diff := cmp.Diff(result, test.wantResult); diff != "" {
				t.Errorf("Unexpected result diff (+got/-want): %s", diff)
			}

			if len(test.wantModels) != len(ds.ModelGetAll()) {
				t.Errorf("Unexpected; want: %d, got:%d", len(test.wantModels), len(ds.ModelGetAll()))
			}

			if diff := diffStore(ds, diffStoreParams{wantPool: pool, wantModels: test.wantModels}); diff != "" {
				t.Errorf("Unexpected diff (+got/-want): %s", diff)
			}

		})
	}
}
