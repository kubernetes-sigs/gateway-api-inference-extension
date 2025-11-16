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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/pool"
	utiltest "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

var (
	selector_v1 = map[string]string{"app": "vllm_v1"}
	selector_v2 = map[string]string{"app": "vllm_v2"}
	pods        = []*corev1.Pod{
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
	gvk := schema.GroupVersionKind{
		Group:   v1.GroupVersion.Group,
		Version: v1.GroupVersion.Version,
		Kind:    "InferencePool",
	}
	pool1 := utiltest.MakeInferencePool("pool1").
		Namespace("pool1-ns").
		Selector(selector_v1).
		TargetPorts(8080).
		EndpointPickerRef("epp-service").ObjRef()
	pool1.SetGroupVersionKind(gvk)
	pool2 := utiltest.MakeInferencePool("pool2").Namespace("pool2-ns").EndpointPickerRef("epp-service").ObjRef()
	pool2.SetGroupVersionKind(gvk)

	period := time.Second
	factories := []datalayer.EndpointFactory{
		backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, period),
		datalayer.NewEndpointFactory([]datalayer.DataSource{&datalayer.FakeDataSource{}}, period),
	}
	for _, epf := range factories {
		// Set up the scheme.
		scheme := runtime.NewScheme()
		_ = clientgoscheme.AddToScheme(scheme)
		_ = v1alpha2.Install(scheme)
		_ = v1.Install(scheme)
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
		gknn := common.GKNN{
			NamespacedName: namespacedName,
			GroupKind: schema.GroupKind{
				Group: pool1.GroupVersionKind().Group,
				Kind:  pool1.GroupVersionKind().Kind,
			},
		}
		req := ctrl.Request{NamespacedName: namespacedName}
		ctx := context.Background()

		ds := datastore.NewDatastore(ctx, epf, 0)
		inferencePoolReconciler := &InferencePoolReconciler{Reader: fakeClient, Datastore: ds, PoolGKNN: gknn}

		// Step 1: Inception, only ready pods matching pool1 are added to the store.
		if _, err := inferencePoolReconciler.Reconcile(ctx, req); err != nil {
			t.Errorf("Unexpected InferencePool reconcile error: %v", err)
		}
		endpointPool1 := pool.InferencePoolToEndpointPool(pool1)
		if diff := diffStore(ds, diffStoreParams{wantPool: endpointPool1, wantEndpoints: []string{"pod1-rank-0", "pod2-rank-0"}}); diff != "" {
			t.Errorf("Unexpected diff (+got/-want): %s", diff)
		}

		newPool1 := &v1.InferencePool{}
		if err := fakeClient.Get(ctx, req.NamespacedName, newPool1); err != nil {
			t.Errorf("Unexpected inferencePool get error: %v", err)
		}
		newPool1.Spec.Selector = v1.LabelSelector{
			MatchLabels: map[v1.LabelKey]v1.LabelValue{"app": "vllm_v2"},
		}
		if err := fakeClient.Update(ctx, newPool1, &client.UpdateOptions{}); err != nil {
			t.Errorf("Unexpected inferencePool update error: %v", err)
		}
		if _, err := inferencePoolReconciler.Reconcile(ctx, req); err != nil {
			t.Errorf("Unexpected InferencePool reconcile error: %v", err)
		}
		newEndpointPool1 := pool.InferencePoolToEndpointPool(newPool1)
		if diff := diffStore(ds, diffStoreParams{wantPool: newEndpointPool1, wantEndpoints: []string{"pod5-rank-0"}}); diff != "" {
			t.Errorf("Unexpected diff (+got/-want): %s", diff)
		}

		// Step 3: update the inferencePool port
		if err := fakeClient.Get(ctx, req.NamespacedName, newPool1); err != nil {
			t.Errorf("Unexpected inferencePool get error: %v", err)
		}
		newPool1.Spec.TargetPorts = []v1.Port{{Number: 9090}}
		if err := fakeClient.Update(ctx, newPool1, &client.UpdateOptions{}); err != nil {
			t.Errorf("Unexpected inferencePool update error: %v", err)
		}
		if _, err := inferencePoolReconciler.Reconcile(ctx, req); err != nil {
			t.Errorf("Unexpected InferencePool reconcile error: %v", err)
		}
		newEndpointPool1 = pool.InferencePoolToEndpointPool(newPool1)
		if diff := diffStore(ds, diffStoreParams{wantPool: newEndpointPool1, wantEndpoints: []string{"pod5-rank-0"}}); diff != "" {
			t.Errorf("Unexpected diff (+got/-want): %s", diff)
		}

		// Step 4: delete the inferencePool to trigger a datastore clear
		if err := fakeClient.Get(ctx, req.NamespacedName, newPool1); err != nil {
			t.Errorf("Unexpected inferencePool get error: %v", err)
		}
		if err := fakeClient.Delete(ctx, newPool1, &client.DeleteOptions{}); err != nil {
			t.Errorf("Unexpected inferencePool delete error: %v", err)
		}
		if _, err := inferencePoolReconciler.Reconcile(ctx, req); err != nil {
			t.Errorf("Unexpected InferencePool reconcile error: %v", err)
		}
		if diff := diffStore(ds, diffStoreParams{wantEndpoints: []string{}}); diff != "" {
			t.Errorf("Unexpected diff (+got/-want): %s", diff)
		}
	}
}

type diffStoreParams struct {
	wantPool       *datalayer.EndpointPool
	wantEndpoints  []string
	wantObjectives []*v1alpha2.InferenceObjective
}

func diffStore(store datastore.Datastore, params diffStoreParams) string {
	gotPool, _ := store.PoolGet()
	if diff := cmp.Diff(params.wantPool, gotPool); diff != "" {
		return "inferencePool:" + diff
	}

	// Default wantPods if not set because PodGetAll returns an empty slice when empty.
	if params.wantEndpoints == nil {
		params.wantEndpoints = []string{}
	}
	gotEndpoints := []string{}
	for _, em := range store.PodList(datastore.AllPodsPredicate) {
		gotEndpoints = append(gotEndpoints, em.GetMetadata().NamespacedName.Name)
	}
	if diff := cmp.Diff(params.wantEndpoints, gotEndpoints, cmpopts.SortSlices(func(a, b string) bool { return a < b })); diff != "" {
		return "endpoints:" + diff
	}

	// Default wantModels if not set because ModelGetAll returns an empty slice when empty.
	if params.wantObjectives == nil {
		params.wantObjectives = []*v1alpha2.InferenceObjective{}
	}

	if diff := cmp.Diff(params.wantObjectives, store.ObjectiveGetAll(), cmpopts.SortSlices(func(a, b *v1alpha2.InferenceObjective) bool {
		return a.Name < b.Name
	})); diff != "" {
		return "models:" + diff
	}
	return ""
}

// Duplicate it as it is just a temporary code
// "inference.networking.x-k8s.io" InferencePool will get removed in the near future.
func TestXInferencePoolReconciler(t *testing.T) {
	// The best practice is to use table-driven tests, however in this scaenario it seems
	// more logical to do a single test with steps that depend on each other.
	gvk := schema.GroupVersionKind{
		Group:   v1alpha2.GroupVersion.Group,
		Version: v1alpha2.GroupVersion.Version,
		Kind:    "InferencePool",
	}
	pool1 := utiltest.MakeAlphaInferencePool("pool1").
		Namespace("pool1-ns").
		Selector(selector_v1).
		ExtensionRef("epp-service").
		TargetPortNumber(8080).ObjRef()
	pool2 := utiltest.MakeAlphaInferencePool("pool2").
		Namespace("pool2-ns").
		ExtensionRef("epp-service").
		TargetPortNumber(8080).ObjRef()
	pool1.SetGroupVersionKind(gvk)
	pool2.SetGroupVersionKind(gvk)

	period := time.Second
	factories := []datalayer.EndpointFactory{
		backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, period),
		datalayer.NewEndpointFactory([]datalayer.DataSource{&datalayer.FakeDataSource{}}, period),
	}
	for _, epf := range factories {
		// Set up the scheme.
		scheme := runtime.NewScheme()
		_ = clientgoscheme.AddToScheme(scheme)
		_ = v1alpha2.Install(scheme)
		_ = v1.Install(scheme)
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
		gknn := common.GKNN{
			NamespacedName: namespacedName,
			GroupKind: schema.GroupKind{
				Group: pool1.GroupVersionKind().Group,
				Kind:  pool1.GroupVersionKind().Kind,
			},
		}
		req := ctrl.Request{NamespacedName: namespacedName}
		ctx := context.Background()

		ds := datastore.NewDatastore(ctx, epf, 0)
		inferencePoolReconciler := &InferencePoolReconciler{Reader: fakeClient, Datastore: ds, PoolGKNN: gknn}

		// Step 1: Inception, only ready pods matching pool1 are added to the store.
		if _, err := inferencePoolReconciler.Reconcile(ctx, req); err != nil {
			t.Errorf("Unexpected InferencePool reconcile error: %v", err)
		}
		endpointPool1 := pool.AlphaInferencePoolToEndpointPool(pool1)
		if diff := xDiffStore(ds, xDiffStoreParams{wantPool: endpointPool1, wantPods: []string{"pod1-rank-0", "pod2-rank-0"}}); diff != "" {
			t.Errorf("Unexpected diff (+got/-want): %s", diff)
		}

		newPool1 := &v1alpha2.InferencePool{}
		if err := fakeClient.Get(ctx, req.NamespacedName, newPool1); err != nil {
			t.Errorf("Unexpected inferencePool get error: %v", err)
		}
		newPool1.Spec.Selector = map[v1alpha2.LabelKey]v1alpha2.LabelValue{"app": "vllm_v2"}
		if err := fakeClient.Update(ctx, newPool1, &client.UpdateOptions{}); err != nil {
			t.Errorf("Unexpected inferencePool update error: %v", err)
		}

		if _, err := inferencePoolReconciler.Reconcile(ctx, req); err != nil {
			t.Errorf("Unexpected InferencePool reconcile error: %v", err)
		}
		newEndpointPool1 := pool.AlphaInferencePoolToEndpointPool(newPool1)
		if diff := xDiffStore(ds, xDiffStoreParams{wantPool: newEndpointPool1, wantPods: []string{"pod5-rank-0"}}); diff != "" {
			t.Errorf("Unexpected diff (+got/-want): %s", diff)
		}

		// Step 3: update the inferencePool port
		if err := fakeClient.Get(ctx, req.NamespacedName, newPool1); err != nil {
			t.Errorf("Unexpected inferencePool get error: %v", err)
		}
		newPool1.Spec.TargetPortNumber = 9090
		if err := fakeClient.Update(ctx, newPool1, &client.UpdateOptions{}); err != nil {
			t.Errorf("Unexpected inferencePool update error: %v", err)
		}
		if _, err := inferencePoolReconciler.Reconcile(ctx, req); err != nil {
			t.Errorf("Unexpected InferencePool reconcile error: %v", err)
		}
		newEndpointPool1 = pool.AlphaInferencePoolToEndpointPool(newPool1)
		if diff := xDiffStore(ds, xDiffStoreParams{wantPool: newEndpointPool1, wantPods: []string{"pod5-rank-0"}}); diff != "" {
			t.Errorf("Unexpected diff (+got/-want): %s", diff)
		}

		// Step 4: delete the inferencePool to trigger a datastore clear
		if err := fakeClient.Get(ctx, req.NamespacedName, newPool1); err != nil {
			t.Errorf("Unexpected inferencePool get error: %v", err)
		}
		if err := fakeClient.Delete(ctx, newPool1, &client.DeleteOptions{}); err != nil {
			t.Errorf("Unexpected inferencePool delete error: %v", err)
		}
		if _, err := inferencePoolReconciler.Reconcile(ctx, req); err != nil {
			t.Errorf("Unexpected InferencePool reconcile error: %v", err)
		}
		if diff := xDiffStore(ds, xDiffStoreParams{wantPods: []string{}}); diff != "" {
			t.Errorf("Unexpected diff (+got/-want): %s", diff)
		}
	}
}

type xDiffStoreParams struct {
	wantPool       *datalayer.EndpointPool
	wantPods       []string
	wantObjectives []*v1alpha2.InferenceObjective
}

func xDiffStore(store datastore.Datastore, params xDiffStoreParams) string {
	gotPool, _ := store.PoolGet()
	if gotPool == nil && params.wantPool == nil {
		return ""
	}

	if diff := cmp.Diff(params.wantPool, gotPool); diff != "" {
		return "inferencePool:" + diff
	}

	// Default wantPods if not set because PodGetAll returns an empty slice when empty.
	if params.wantPods == nil {
		params.wantPods = []string{}
	}
	gotPods := []string{}
	for _, em := range store.PodList(datastore.AllPodsPredicate) {
		gotPods = append(gotPods, em.GetMetadata().NamespacedName.Name)
	}
	if diff := cmp.Diff(params.wantPods, gotPods, cmpopts.SortSlices(func(a, b string) bool { return a < b })); diff != "" {
		return "pods:" + diff
	}

	// Default wantModels if not set because ModelGetAll returns an empty slice when empty.
	if params.wantObjectives == nil {
		params.wantObjectives = []*v1alpha2.InferenceObjective{}
	}

	if diff := cmp.Diff(params.wantObjectives, store.ObjectiveGetAll(), cmpopts.SortSlices(func(a, b *v1alpha2.InferenceObjective) bool {
		return a.Name < b.Name
	})); diff != "" {
		return "models:" + diff
	}
	return ""
}
