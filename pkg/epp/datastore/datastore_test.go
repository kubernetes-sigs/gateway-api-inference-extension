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

package datastore

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	testutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

func TestPool(t *testing.T) {
	pool1Selector := map[string]string{"app": "vllm_v1"}
	pool1 := testutil.MakeInferencePool("pool1").
		Namespace("default").
		Selector(pool1Selector).ObjRef()
	tests := []struct {
		name            string
		inferencePool   *v1alpha2.InferencePool
		labels          map[string]string
		wantSynced      bool
		wantPool        *v1alpha2.InferencePool
		wantErr         error
		wantLabelsMatch bool
	}{
		{
			name:            "Ready when InferencePool exists in data store",
			inferencePool:   pool1,
			labels:          pool1Selector,
			wantSynced:      true,
			wantPool:        pool1,
			wantLabelsMatch: true,
		},
		{
			name:            "Labels not matched",
			inferencePool:   pool1,
			labels:          map[string]string{"app": "vllm_v2"},
			wantSynced:      true,
			wantPool:        pool1,
			wantLabelsMatch: false,
		},
		{
			name:       "Not ready when InferencePool is nil in data store",
			wantErr:    errPoolNotSynced,
			wantSynced: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set up the scheme.
			scheme := runtime.NewScheme()
			_ = clientgoscheme.AddToScheme(scheme)
			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				Build()
			pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, time.Second)
			datastore := NewDatastore(context.Background(), pmf)
			_ = datastore.PoolSet(context.Background(), fakeClient, tt.inferencePool)
			gotPool, gotErr := datastore.PoolGet()
			if diff := cmp.Diff(tt.wantErr, gotErr, cmpopts.EquateErrors()); diff != "" {
				t.Errorf("Unexpected error diff (+got/-want): %s", diff)
			}
			if diff := cmp.Diff(tt.wantPool, gotPool); diff != "" {
				t.Errorf("Unexpected pool diff (+got/-want): %s", diff)
			}
			gotSynced := datastore.PoolHasSynced()
			if diff := cmp.Diff(tt.wantSynced, gotSynced); diff != "" {
				t.Errorf("Unexpected synced diff (+got/-want): %s", diff)
			}
			if tt.labels != nil {
				gotLabelsMatch := datastore.PoolLabelsMatch(tt.labels)
				if diff := cmp.Diff(tt.wantLabelsMatch, gotLabelsMatch); diff != "" {
					t.Errorf("Unexpected labels match diff (+got/-want): %s", diff)
				}
			}
		})
	}
}

func TestModel(t *testing.T) {
	chatModel := "chat"
	tsModel := "food-review"
	model1ts := testutil.MakeInferenceModel("model1").
		CreationTimestamp(metav1.Unix(1000, 0)).
		ModelName(tsModel).ObjRef()
	// Same model name as model1ts, different object name.
	model2ts := testutil.MakeInferenceModel("model2").
		CreationTimestamp(metav1.Unix(1001, 0)).
		ModelName(tsModel).ObjRef()
	// Same model name as model1ts, newer timestamp
	model1tsNewer := testutil.MakeInferenceModel("model1").
		CreationTimestamp(metav1.Unix(1002, 0)).
		Criticality(v1alpha2.Critical).
		ModelName(tsModel).ObjRef()
	model2tsNewer := testutil.MakeInferenceModel("model2").
		CreationTimestamp(metav1.Unix(1003, 0)).
		ModelName(tsModel).ObjRef()
	// Same object name as model2ts, different model name.
	model2chat := testutil.MakeInferenceModel(model2ts.Name).
		CreationTimestamp(metav1.Unix(1005, 0)).
		ModelName(chatModel).ObjRef()

	tests := []struct {
		name           string
		existingModels []*v1alpha2.InferenceModel
		op             func(ds Datastore) bool
		wantOpResult   bool
		wantModels     []*v1alpha2.InferenceModel
	}{
		{
			name: "Add model1 with food-review as modelName",
			op: func(ds Datastore) bool {
				return ds.ModelSetIfOlder(model1ts)
			},
			wantModels:   []*v1alpha2.InferenceModel{model1ts},
			wantOpResult: true,
		},
		{
			name:           "Set model1 with the same modelName, but with diff criticality and newer creation timestamp, should update.",
			existingModels: []*v1alpha2.InferenceModel{model1ts},
			op: func(ds Datastore) bool {
				return ds.ModelSetIfOlder(model1tsNewer)
			},
			wantOpResult: true,
			wantModels:   []*v1alpha2.InferenceModel{model1tsNewer},
		},
		{
			name:           "set model2 with the same modelName, but newer creation timestamp, should not update.",
			existingModels: []*v1alpha2.InferenceModel{model1tsNewer},
			op: func(ds Datastore) bool {
				return ds.ModelSetIfOlder(model2tsNewer)
			},
			wantOpResult: false,
			wantModels:   []*v1alpha2.InferenceModel{model1tsNewer},
		},
		{
			name:           "Set model2 with the same modelName, but older creation timestamp, should update",
			existingModels: []*v1alpha2.InferenceModel{model1tsNewer},
			op: func(ds Datastore) bool {
				return ds.ModelSetIfOlder(model2ts)
			},
			wantOpResult: true,
			wantModels:   []*v1alpha2.InferenceModel{model2ts},
		},
		{
			name:           "Set model1 with the food-review modelName, both models should exist",
			existingModels: []*v1alpha2.InferenceModel{model2chat},
			op: func(ds Datastore) bool {
				return ds.ModelSetIfOlder(model1ts)
			},
			wantOpResult: true,
			wantModels:   []*v1alpha2.InferenceModel{model2chat, model1ts},
		},
		{
			name:           "Set model1 with the food-review modelName, both models should exist",
			existingModels: []*v1alpha2.InferenceModel{model2chat, model1ts},
			op: func(ds Datastore) bool {
				return ds.ModelSetIfOlder(model1ts)
			},
			wantOpResult: true,
			wantModels:   []*v1alpha2.InferenceModel{model2chat, model1ts},
		},
		{
			name:           "Getting by model name, chat -> model2",
			existingModels: []*v1alpha2.InferenceModel{model2chat, model1ts},
			op: func(ds Datastore) bool {
				gotChat := ds.ModelGet(chatModel)
				return gotChat != nil && cmp.Diff(model2chat, gotChat) == ""
			},
			wantOpResult: true,
			wantModels:   []*v1alpha2.InferenceModel{model2chat, model1ts},
		},
		{
			name:           "Delete the model",
			existingModels: []*v1alpha2.InferenceModel{model2chat, model1ts},
			op: func(ds Datastore) bool {
				existing := ds.ModelDelete(types.NamespacedName{Name: model1ts.Name, Namespace: model1ts.Namespace})
				got := ds.ModelGet(tsModel)
				return existing != nil && got == nil

			},
			wantOpResult: true,
			wantModels:   []*v1alpha2.InferenceModel{model2chat},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, time.Second)
			ds := NewDatastore(t.Context(), pmf)
			for _, m := range test.existingModels {
				ds.ModelSetIfOlder(m)
			}

			gotOpResult := test.op(ds)
			if gotOpResult != test.wantOpResult {
				t.Errorf("Unexpected operation result, want: %v, got: %v", test.wantOpResult, gotOpResult)
			}

			if diff := testutil.DiffModelLists(test.wantModels, ds.ModelGetAll()); diff != "" {
				t.Errorf("Unexpected models diff: %s", diff)
			}

		})
	}
}

var (
	pod1 = &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
		},
	}
	pod1Metrics = &backendmetrics.Metrics{
		WaitingQueueSize:    0,
		KVCacheUsagePercent: 0.2,
		MaxActiveModels:     2,
		ActiveModels: map[string]int{
			"foo": 1,
			"bar": 1,
		},
		WaitingModels: map[string]int{},
	}
	pod2 = &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod2",
		},
	}
	pod2Metrics = &backendmetrics.Metrics{
		WaitingQueueSize:    1,
		KVCacheUsagePercent: 0.2,
		MaxActiveModels:     2,
		ActiveModels: map[string]int{
			"foo1": 1,
			"bar1": 1,
		},
		WaitingModels: map[string]int{},
	}
	pod1NamespacedName = types.NamespacedName{Name: pod1.Name, Namespace: pod1.Namespace}
	pod2NamespacedName = types.NamespacedName{Name: pod2.Name, Namespace: pod2.Namespace}
	inferencePool      = &v1alpha2.InferencePool{
		Spec: v1alpha2.InferencePoolSpec{
			TargetPortNumber: 8000,
		},
	}
)

func TestMetrics(t *testing.T) {
	tests := []struct {
		name      string
		pmc       backendmetrics.PodMetricsClient
		storePods []*corev1.Pod
		want      []*backendmetrics.Metrics
	}{
		{
			name: "Probing metrics success",
			pmc: &backendmetrics.FakePodMetricsClient{
				Res: map[types.NamespacedName]*backendmetrics.Metrics{
					pod1NamespacedName: pod1Metrics,
					pod2NamespacedName: pod2Metrics,
				},
			},
			storePods: []*corev1.Pod{pod1, pod2},
			want:      []*backendmetrics.Metrics{pod1Metrics, pod2Metrics},
		},
		{
			name: "Only pods in are probed",
			pmc: &backendmetrics.FakePodMetricsClient{
				Res: map[types.NamespacedName]*backendmetrics.Metrics{
					pod1NamespacedName: pod1Metrics,
					pod2NamespacedName: pod2Metrics,
				},
			},
			storePods: []*corev1.Pod{pod1},
			want:      []*backendmetrics.Metrics{pod1Metrics},
		},
		{
			name: "Probing metrics error",
			pmc: &backendmetrics.FakePodMetricsClient{
				Err: map[types.NamespacedName]error{
					pod2NamespacedName: errors.New("injected error"),
				},
				Res: map[types.NamespacedName]*backendmetrics.Metrics{
					pod1NamespacedName: pod1Metrics,
				},
			},
			storePods: []*corev1.Pod{pod1, pod2},
			want: []*backendmetrics.Metrics{
				pod1Metrics,
				// Failed to fetch pod2 metrics so it remains the default values.
				{
					ActiveModels:        map[string]int{},
					WaitingModels:       map[string]int{},
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0,
					MaxActiveModels:     0,
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			// Set up the scheme.
			scheme := runtime.NewScheme()
			_ = clientgoscheme.AddToScheme(scheme)
			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				Build()
			pmf := backendmetrics.NewPodMetricsFactory(test.pmc, time.Millisecond)
			ds := NewDatastore(ctx, pmf)
			_ = ds.PoolSet(ctx, fakeClient, inferencePool)
			for _, pod := range test.storePods {
				ds.PodUpdateOrAddIfNotExist(pod)
			}
			assert.EventuallyWithT(t, func(t *assert.CollectT) {
				got := ds.PodGetAll()
				metrics := []*backendmetrics.Metrics{}
				for _, one := range got {
					metrics = append(metrics, one.GetMetrics())
				}
				diff := cmp.Diff(test.want, metrics, cmpopts.IgnoreFields(backendmetrics.Metrics{}, "UpdateTime"), cmpopts.SortSlices(func(a, b *backendmetrics.Metrics) bool {
					return a.String() < b.String()
				}))
				assert.Equal(t, "", diff, "Unexpected diff (+got/-want)")
			}, 5*time.Second, time.Millisecond)
		})
	}
}
