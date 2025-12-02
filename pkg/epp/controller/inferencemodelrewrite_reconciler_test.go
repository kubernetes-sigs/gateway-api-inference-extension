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
	"github.com/google/go-cmp/cmp/cmpopts"
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
	poolForRewrite = utiltest.MakeInferencePool("test-pool1").Namespace("ns1").ObjRef()
	rewrite1       = &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "rewrite1",
			Namespace:         poolForRewrite.Namespace,
			CreationTimestamp: metav1.Unix(1000, 0),
		},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			PoolRef: &v1alpha2.PoolObjectReference{
				Name:  v1alpha2.ObjectName(poolForRewrite.Name),
				Group: v1alpha2.Group(poolForRewrite.GroupVersionKind().Group),
			},
		},
	}
	rewrite1Pool2 = &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{
			Name:              rewrite1.Name,
			Namespace:         rewrite1.Namespace,
			CreationTimestamp: metav1.Unix(1001, 0),
		},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			PoolRef: &v1alpha2.PoolObjectReference{
				Name:  "test-pool2",
				Group: v1alpha2.Group(poolForRewrite.GroupVersionKind().Group),
			},
		},
	}
	rewrite1Updated = &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{
			Name:              rewrite1.Name,
			Namespace:         rewrite1.Namespace,
			CreationTimestamp: metav1.Unix(1003, 0),
		},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			PoolRef: &v1alpha2.PoolObjectReference{
				Name:  v1alpha2.ObjectName(poolForRewrite.Name),
				Group: v1alpha2.Group(poolForRewrite.GroupVersionKind().Group),
			},
			Rules: []v1alpha2.InferenceModelRewriteRule{{}},
		},
	}
	rewrite1Deleted = &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{
			Name:              rewrite1.Name,
			Namespace:         rewrite1.Namespace,
			CreationTimestamp: metav1.Unix(1004, 0),
			DeletionTimestamp: &metav1.Time{Time: time.Now()},
			Finalizers:        []string{"deleted"},
		},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			PoolRef: &v1alpha2.PoolObjectReference{
				Name:  v1alpha2.ObjectName(poolForRewrite.Name),
				Group: v1alpha2.Group(poolForRewrite.GroupVersionKind().Group),
			},
		},
	}
	rewrite2 = &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "rewrite2",
			Namespace:         poolForRewrite.Namespace,
			CreationTimestamp: metav1.Unix(1001, 0),
		},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			PoolRef: &v1alpha2.PoolObjectReference{
				Name:  v1alpha2.ObjectName(poolForRewrite.Name),
				Group: v1alpha2.Group(poolForRewrite.GroupVersionKind().Group),
			},
		},
	}
)

func TestInferenceModelRewriteReconciler(t *testing.T) {
	tests := []struct {
		name                string
		rewritesInStore     []*v1alpha2.InferenceModelRewrite
		rewritesInAPIServer []*v1alpha2.InferenceModelRewrite
		rewrite             *v1alpha2.InferenceModelRewrite
		incomingReq         *types.NamespacedName
		wantRewrites        []*v1alpha2.InferenceModelRewrite
		wantResult          ctrl.Result
	}{
		{
			name:         "Empty store, add new rewrite",
			rewrite:      rewrite1,
			wantRewrites: []*v1alpha2.InferenceModelRewrite{rewrite1},
		},
		{
			name:            "Existing rewrite changed pools",
			rewritesInStore: []*v1alpha2.InferenceModelRewrite{rewrite1},
			rewrite:         rewrite1Pool2,
			wantRewrites:    []*v1alpha2.InferenceModelRewrite{},
		},
		{
			name:            "Not found, delete existing rewrite",
			rewritesInStore: []*v1alpha2.InferenceModelRewrite{rewrite1},
			incomingReq:     &types.NamespacedName{Name: rewrite1.Name, Namespace: rewrite1.Namespace},
			wantRewrites:    []*v1alpha2.InferenceModelRewrite{},
		},
		{
			name:            "Deletion timestamp set, delete existing rewrite",
			rewritesInStore: []*v1alpha2.InferenceModelRewrite{rewrite1},
			rewrite:         rewrite1Deleted,
			incomingReq:     &types.NamespacedName{Name: rewrite1Deleted.Name, Namespace: rewrite1Deleted.Namespace},
			wantRewrites:    []*v1alpha2.InferenceModelRewrite{},
		},
		{
			name:            "Rewrite updated",
			rewritesInStore: []*v1alpha2.InferenceModelRewrite{rewrite1},
			rewrite:         rewrite1Updated,
			wantRewrites:    []*v1alpha2.InferenceModelRewrite{rewrite1Updated},
		},
		{
			name:            "Rewrite not found, no matching existing rewrite to delete",
			rewritesInStore: []*v1alpha2.InferenceModelRewrite{rewrite1},
			incomingReq:     &types.NamespacedName{Name: "non-existent-rewrite", Namespace: poolForRewrite.Namespace},
			wantRewrites:    []*v1alpha2.InferenceModelRewrite{rewrite1},
		},
		{
			name:            "Add to existing",
			rewritesInStore: []*v1alpha2.InferenceModelRewrite{rewrite1},
			rewrite:         rewrite2,
			wantRewrites:    []*v1alpha2.InferenceModelRewrite{rewrite1, rewrite2},
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
				scheme := runtime.NewScheme()
				_ = clientgoscheme.AddToScheme(scheme)
				_ = v1alpha2.Install(scheme)
				_ = v1.Install(scheme)
				initObjs := []client.Object{}
				if test.rewrite != nil {
					initObjs = append(initObjs, test.rewrite)
				}
				for _, r := range test.rewritesInAPIServer {
					initObjs = append(initObjs, r)
				}
				fakeClient := fake.NewClientBuilder().
					WithScheme(scheme).
					WithObjects(initObjs...).
					Build()
				ds := datastore.NewDatastore(t.Context(), epf, 0)
				for _, r := range test.rewritesInStore {
					ds.ModelRewriteSet(r)
				}
				endpointPool := poolutil.InferencePoolToEndpointPool(poolForRewrite)
				_ = ds.PoolSet(context.Background(), fakeClient, endpointPool)
				reconciler := &InferenceModelRewriteReconciler{
					Reader:    fakeClient,
					Datastore: ds,
					PoolGKNN: common.GKNN{
						NamespacedName: types.NamespacedName{Name: poolForRewrite.Name, Namespace: poolForRewrite.Namespace},
						GroupKind:      schema.GroupKind{Group: poolForRewrite.GroupVersionKind().Group, Kind: poolForRewrite.GroupVersionKind().Kind},
					},
				}
				if test.incomingReq == nil {
					test.incomingReq = &types.NamespacedName{Name: test.rewrite.Name, Namespace: test.rewrite.Namespace}
				}

				result, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: *test.incomingReq})
				if err != nil {
					t.Fatalf("expected no error, got %v", err)
				}

				if diff := cmp.Diff(result, test.wantResult); diff != "" {
					t.Errorf("Unexpected result diff (+got/-want): %s", diff)
				}

				if len(test.wantRewrites) != len(ds.ModelRewriteGetAll()) {
					t.Errorf("Unexpected number of rewrites; want: %d, got:%d", len(test.wantRewrites), len(ds.ModelRewriteGetAll()))
				}

				if diff := diffStoreRewrites(ds, test.wantRewrites); diff != "" {
					t.Errorf("Unexpected diff (+got/-want): %s", diff)
				}
			})
		}
	}
}

func diffStoreRewrites(ds datastore.Datastore, wantRewrites []*v1alpha2.InferenceModelRewrite) string {
	if wantRewrites == nil {
		wantRewrites = []*v1alpha2.InferenceModelRewrite{}
	}
	gotRewrites := ds.ModelRewriteGetAll()

	less := func(a, b *v1alpha2.InferenceModelRewrite) bool {
		if a.Namespace != b.Namespace {
			return a.Namespace < b.Namespace
		}
		return a.Name < b.Name
	}

	if diff := cmp.Diff(wantRewrites, gotRewrites, cmpopts.SortSlices(less)); diff != "" {
		return "rewrites:" + diff
	}
	return ""
}
