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
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	utiltest "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

var (
	selector_v1 = map[string]string{"app": "vllm_v1"}
	selector_v2 = map[string]string{"app": "vllm_v2"}
	pool1       = utiltest.MakeInferencePool("pool1").
			Namespace("pool1-ns").
			Selector(selector_v1).
			TargetPortNumber(8080).ObjRef()
	pool2 = utiltest.MakeInferencePool("pool2").Namespace("pool2-ns").ObjRef()
	pods  = []*corev1.Pod{
		// Two ready pods matching pool1
		utiltest.MakePod("pod1").
			Namespace("pool1-ns").
			Labels(selector_v1).ReadyCondition().ObjRef(),
		utiltest.MakePod("pod2").
			Namespace("pool1-ns").
			Labels(selector_v1).
			ReadyCondition().ObjRef(),
		// A not ready pod matching pool1
		utiltest.MakePod("pod3").
			Namespace("pool1-ns").
			Labels(selector_v1).ObjRef(),
		// A pod not matching pool1 namespace
		utiltest.MakePod("pod4").
			Namespace("pool2-ns").
			Labels(selector_v1).
			ReadyCondition().ObjRef(),
		// A ready pod matching pool1 with a new selector
		utiltest.MakePod("pod5").
			Namespace("pool1-ns").
			Labels(selector_v2).
			ReadyCondition().ObjRef(),
	}
)

func TestInferencePoolReconciler(t *testing.T) {
	// The best practice is to use table-driven tests, however in this scaenario it seems
	// more logical to do a single test with steps that depend on each other.

	// Set up the scheme.
	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	_ = v1alpha2.Install(scheme)

	// Create a fake client with the pool and the pods.
	initialObjects := []client.Object{pool1, pool2}
	for i := range pods {
		initialObjects = append(initialObjects, pods[i])
	}
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(initialObjects...).
		Build()

	// Create a request for the existing resource.
	namespacedName := types.NamespacedName{Name: pool1.Name, Namespace: pool1.Namespace}
	req := ctrl.Request{NamespacedName: namespacedName}
	ctx := context.Background()

	pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, time.Second)
	datastore := datastore.NewDatastore(ctx, pmf)
	inferencePoolReconciler := &InferencePoolReconciler{Client: fakeClient, Datastore: datastore}

	// Step 1: Inception, only ready pods matching pool1 are added to the store.
	if _, err := inferencePoolReconciler.Reconcile(ctx, req); err != nil {
		t.Errorf("Unexpected InferencePool reconcile error: %v", err)
	}
	if diff := diffStore(datastore, diffStoreParams{wantPool: pool1, wantPods: []string{"pod1", "pod2"}}); diff != "" {
		t.Errorf("Unexpected diff (+got/-want): %s", diff)
	}

	newPool1 := &v1alpha2.InferencePool{}
	if err := fakeClient.Get(ctx, req.NamespacedName, newPool1); err != nil {
		t.Errorf("Unexpected pool get error: %v", err)
	}
	newPool1.Spec.Selector = map[v1alpha2.LabelKey]v1alpha2.LabelValue{"app": "vllm_v2"}
	if err := fakeClient.Update(ctx, newPool1, &client.UpdateOptions{}); err != nil {
		t.Errorf("Unexpected pool update error: %v", err)
	}

	if _, err := inferencePoolReconciler.Reconcile(ctx, req); err != nil {
		t.Errorf("Unexpected InferencePool reconcile error: %v", err)
	}
	if diff := diffStore(datastore, diffStoreParams{wantPool: newPool1, wantPods: []string{"pod5"}}); diff != "" {
		t.Errorf("Unexpected diff (+got/-want): %s", diff)
	}

	// Step 3: update the pool port
	if err := fakeClient.Get(ctx, req.NamespacedName, newPool1); err != nil {
		t.Errorf("Unexpected pool get error: %v", err)
	}
	newPool1.Spec.TargetPortNumber = 9090
	if err := fakeClient.Update(ctx, newPool1, &client.UpdateOptions{}); err != nil {
		t.Errorf("Unexpected pool update error: %v", err)
	}
	if _, err := inferencePoolReconciler.Reconcile(ctx, req); err != nil {
		t.Errorf("Unexpected InferencePool reconcile error: %v", err)
	}
	if diff := diffStore(datastore, diffStoreParams{wantPool: newPool1, wantPods: []string{"pod5"}}); diff != "" {
		t.Errorf("Unexpected diff (+got/-want): %s", diff)
	}

	// Step 4: delete the pool to trigger a datastore clear
	if err := fakeClient.Get(ctx, req.NamespacedName, newPool1); err != nil {
		t.Errorf("Unexpected pool get error: %v", err)
	}
	if err := fakeClient.Delete(ctx, newPool1, &client.DeleteOptions{}); err != nil {
		t.Errorf("Unexpected pool delete error: %v", err)
	}
	if _, err := inferencePoolReconciler.Reconcile(ctx, req); err != nil {
		t.Errorf("Unexpected InferencePool reconcile error: %v", err)
	}
	if diff := diffStore(datastore, diffStoreParams{wantPods: []string{}}); diff != "" {
		t.Errorf("Unexpected diff (+got/-want): %s", diff)
	}
}

type diffStoreParams struct {
	wantPool   *v1alpha2.InferencePool
	wantPods   []string
	wantModels []*v1alpha2.InferenceModel
}

func diffStore(datastore datastore.Datastore, params diffStoreParams) string {
	gotPool, _ := datastore.PoolGet()
	if diff := cmp.Diff(params.wantPool, gotPool); diff != "" {
		return "pool:" + diff
	}

	// Default wantPods if not set because PodGetAll returns an empty slice when empty.
	if params.wantPods == nil {
		params.wantPods = []string{}
	}
	gotPods := []string{}
	for _, pm := range datastore.PodGetAll() {
		gotPods = append(gotPods, pm.GetPod().NamespacedName.Name)
	}
	if diff := cmp.Diff(params.wantPods, gotPods, cmpopts.SortSlices(func(a, b string) bool { return a < b })); diff != "" {
		return "pods:" + diff
	}

	// Default wantModels if not set because ModelGetAll returns an empty slice when empty.
	if params.wantModels == nil {
		params.wantModels = []*v1alpha2.InferenceModel{}
	}
	gotModels := datastore.ModelGetAll()
	if diff := utiltest.DiffModelLists(params.wantModels, gotModels); diff != "" {
		return "models:" + diff
	}
	return ""
}
