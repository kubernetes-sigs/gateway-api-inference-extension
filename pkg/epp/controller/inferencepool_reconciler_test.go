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

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	utiltesting "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

var (
	selector_v1 = map[string]string{"app": "vllm_v1"}
	selector_v2 = map[string]string{"app": "vllm_v2"}
	pool1       = &v1alpha2.InferencePool{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pool1",
			Namespace: "pool1-ns",
		},
		Spec: v1alpha2.InferencePoolSpec{
			Selector:         map[v1alpha2.LabelKey]v1alpha2.LabelValue{"app": "vllm_v1"},
			TargetPortNumber: 8080,
		},
	}
	pool2 = &v1alpha2.InferencePool{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pool2",
			Namespace: "pool2-ns",
		},
	}
	pods = []corev1.Pod{
		// Two ready pods matching pool1
		utiltesting.MakePod("pod1", "pool1-ns").Labels(selector_v1).ReadyCondition().Obj(),
		utiltesting.MakePod("pod2", "pool1-ns").Labels(selector_v1).ReadyCondition().Obj(),
		// A not ready pod matching pool1
		utiltesting.MakePod("pod3", "pool1-ns").Labels(selector_v1).Obj(),
		// A pod not matching pool1 namespace
		utiltesting.MakePod("pod4", "pool2-ns").Labels(selector_v1).ReadyCondition().Obj(),
		// A ready pod matching pool1 with a new selector
		utiltesting.MakePod("pod5", "pool1-ns").Labels(selector_v2).ReadyCondition().Obj(),
	}
)

func TestReconcile_InferencePoolReconciler(t *testing.T) {
	// The best practice is to use table-driven tests, however in this scaenario it seems
	// more logical to do a single test with steps that depend on each other.

	// Set up the scheme.
	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	_ = v1alpha2.AddToScheme(scheme)

	// Create a fake client with the pool and the pods.
	initialObjects := []client.Object{pool1, pool2}
	for i := range pods {
		initialObjects = append(initialObjects, &pods[i])
	}
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(initialObjects...).
		Build()

	// Create a request for the existing resource.
	namespacedName := types.NamespacedName{Name: pool1.Name, Namespace: pool1.Namespace}
	req := ctrl.Request{NamespacedName: namespacedName}
	ctx := context.Background()

	datastore := datastore.NewDatastore()
	inferencePoolReconciler := &InferencePoolReconciler{PoolNamespacedName: namespacedName, Client: fakeClient, Datastore: datastore}

	// Step 1: Inception, only ready pods matching pool1 are added to the store.
	if _, err := inferencePoolReconciler.Reconcile(ctx, req); err != nil {
		t.Errorf("Unexpected InferencePool reconcile error: %v", err)
	}
	if diff := diffPool(datastore, pool1, []string{"pod1", "pod2"}); diff != "" {
		t.Errorf("Unexpected diff (+got/-want): %s", diff)
	}

	// Step 2: A reconcile on pool2 should not change anything.
	if _, err := inferencePoolReconciler.Reconcile(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Name: pool2.Name, Namespace: pool2.Namespace}}); err != nil {
		t.Errorf("Unexpected InferencePool reconcile error: %v", err)
	}
	if diff := diffPool(datastore, pool1, []string{"pod1", "pod2"}); diff != "" {
		t.Errorf("Unexpected diff (+got/-want): %s", diff)
	}

	// Step 3: update the pool selector to include more pods
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
	if diff := diffPool(datastore, newPool1, []string{"pod5"}); diff != "" {
		t.Errorf("Unexpected diff (+got/-want): %s", diff)
	}

	// Step 4: update the pool port
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
	if diff := diffPool(datastore, newPool1, []string{"pod5"}); diff != "" {
		t.Errorf("Unexpected diff (+got/-want): %s", diff)
	}

	// Step 5: delete the pool to trigger a datastore clear
	if err := fakeClient.Get(ctx, req.NamespacedName, newPool1); err != nil {
		t.Errorf("Unexpected pool get error: %v", err)
	}
	if err := fakeClient.Delete(ctx, newPool1, &client.DeleteOptions{}); err != nil {
		t.Errorf("Unexpected pool delete error: %v", err)
	}
	if _, err := inferencePoolReconciler.Reconcile(ctx, req); err != nil {
		t.Errorf("Unexpected InferencePool reconcile error: %v", err)
	}
	if diff := diffPool(datastore, nil, []string{}); diff != "" {
		t.Errorf("Unexpected diff (+got/-want): %s", diff)
	}
}

func diffPool(datastore datastore.Datastore, wantPool *v1alpha2.InferencePool, wantPods []string) string {
	gotPool, _ := datastore.PoolGet()
	if diff := cmp.Diff(wantPool, gotPool); diff != "" {
		return diff
	}
	gotPods := []string{}
	for _, pm := range datastore.PodGetAll() {
		gotPods = append(gotPods, pm.NamespacedName.Name)
	}
	return cmp.Diff(wantPods, gotPods, cmpopts.SortSlices(func(a, b string) bool { return a < b }))
}
