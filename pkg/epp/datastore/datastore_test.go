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
	"net"
	"strconv"
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
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	pooltuil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/pool"
	testutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

func TestPool(t *testing.T) {
	pool1Selector := map[string]string{"app": "vllm_v1"}
	pool1 := testutil.MakeInferencePool("pool1").
		Namespace("default").
		Selector(pool1Selector).ObjRef()
	tests := []struct {
		name            string
		inferencePool   *v1.InferencePool
		labels          map[string]string
		wantSynced      bool
		wantPool        *v1.InferencePool
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
		period := time.Second
		factories := []datalayer.EndpointFactory{
			backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, period),
			datalayer.NewEndpointFactory([]datalayer.DataSource{&datalayer.FakeDataSource{}}, period),
		}
		for _, epf := range factories {
			t.Run(tt.name, func(t *testing.T) {
				// Set up the scheme.
				scheme := runtime.NewScheme()
				_ = clientgoscheme.AddToScheme(scheme)
				fakeClient := fake.NewClientBuilder().
					WithScheme(scheme).
					Build()

				ds := NewDatastore(context.Background(), epf, 0)
				_ = ds.PoolSet(context.Background(), fakeClient, pooltuil.InferencePoolToEndpointPool(tt.inferencePool))
				gotPool, gotErr := ds.PoolGet()
				if diff := cmp.Diff(tt.wantErr, gotErr, cmpopts.EquateErrors()); diff != "" {
					t.Errorf("Unexpected error diff (+got/-want): %s", diff)
				}
				if diff := cmp.Diff(pooltuil.InferencePoolToEndpointPool(tt.wantPool), gotPool); diff != "" {
					t.Errorf("Unexpected pool diff (+got/-want): %s", diff)
				}
				gotSynced := ds.PoolHasSynced()
				if diff := cmp.Diff(tt.wantSynced, gotSynced); diff != "" {
					t.Errorf("Unexpected synced diff (+got/-want): %s", diff)
				}
				if tt.labels != nil {
					gotLabelsMatch := ds.PoolLabelsMatch(tt.labels)
					if diff := cmp.Diff(tt.wantLabelsMatch, gotLabelsMatch); diff != "" {
						t.Errorf("Unexpected labels match diff (+got/-want): %s", diff)
					}
				}
			})
		}
	}
}

func TestObjective(t *testing.T) {
	chatModel := "chat"
	tsModel := "food-review"
	model1ts := testutil.MakeInferenceObjective("model1").ObjRef()
	// Same model name as model1ts, different object name.
	model2ts := testutil.MakeInferenceObjective("model2").ObjRef()
	// Same model name as model1ts, newer timestamp
	model1tsCritical := testutil.MakeInferenceObjective("model1").
		Priority(2).ObjRef()
	// Same object name as model2ts, different model name.
	model2chat := testutil.MakeInferenceObjective(model2ts.Name).ObjRef()

	tests := []struct {
		name           string
		existingModels []*v1alpha2.InferenceObjective
		op             func(ds Datastore) bool
		wantOpResult   bool
		wantModels     []*v1alpha2.InferenceObjective
	}{
		{
			name: "Add model1 with food-review as modelName",
			op: func(ds Datastore) bool {
				ds.ObjectiveSet(model1ts)
				return cmp.Diff(ds.ObjectiveGet(model1ts.Name), model1ts) == ""
			},
			wantModels:   []*v1alpha2.InferenceObjective{model1ts},
			wantOpResult: true,
		},
		{
			name:           "Set model1 with the same modelName, but with diff priority, should update.",
			existingModels: []*v1alpha2.InferenceObjective{model1ts},
			op: func(ds Datastore) bool {
				ds.ObjectiveSet(model1tsCritical)
				return cmp.Diff(ds.ObjectiveGet(model1tsCritical.Name), model1tsCritical) == ""
			},
			wantOpResult: true,
			wantModels:   []*v1alpha2.InferenceObjective{model1tsCritical},
		},
		{
			name:           "Set model1 with the food-review modelName, both models should exist",
			existingModels: []*v1alpha2.InferenceObjective{model2chat},
			op: func(ds Datastore) bool {
				ds.ObjectiveSet(model1ts)
				return cmp.Diff(ds.ObjectiveGet(model1ts.Name), model1ts) == ""
			},
			wantOpResult: true,
			wantModels:   []*v1alpha2.InferenceObjective{model2chat, model1ts},
		},
		{
			name:           "Set model1 with the food-review modelName, both models should exist",
			existingModels: []*v1alpha2.InferenceObjective{model2chat, model1ts},
			op: func(ds Datastore) bool {
				ds.ObjectiveSet(model1ts)
				return cmp.Diff(ds.ObjectiveGet(model1ts.Name), model1ts) == ""
			},
			wantOpResult: true,
			wantModels:   []*v1alpha2.InferenceObjective{model2chat, model1ts},
		},
		{
			name:           "Getting by model name, chat -> model2",
			existingModels: []*v1alpha2.InferenceObjective{model2chat, model1ts},
			op: func(ds Datastore) bool {
				gotChat := ds.ObjectiveGet(chatModel)
				return gotChat != nil && cmp.Diff(model2chat, gotChat) == ""
			},
			wantOpResult: false,
			wantModels:   []*v1alpha2.InferenceObjective{model2chat, model1ts},
		},
		{
			name:           "Delete the model",
			existingModels: []*v1alpha2.InferenceObjective{model2chat, model1ts},
			op: func(ds Datastore) bool {
				ds.ObjectiveDelete(types.NamespacedName{Name: model1ts.Name, Namespace: model1ts.Namespace})
				got := ds.ObjectiveGet(tsModel)
				return got == nil

			},
			wantOpResult: true,
			wantModels:   []*v1alpha2.InferenceObjective{model2chat},
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
				ds := NewDatastore(t.Context(), epf, 0)
				for _, m := range test.existingModels {
					ds.ObjectiveSet(m)
				}

				gotOpResult := test.op(ds)
				if gotOpResult != test.wantOpResult {
					t.Errorf("Unexpected operation result, want: %v, got: %v", test.wantOpResult, gotOpResult)
				}

				if diff := cmp.Diff(test.wantModels, ds.ObjectiveGetAll(), cmpopts.SortSlices(func(a, b *v1alpha2.InferenceObjective) bool {
					return a.Name < b.Name
				})); diff != "" {
					t.Errorf("Unexpected models diff: %s", diff)
				}
			})
		}
	}
}

var (
	pod1 = &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
		},
	}
	pod1Metrics = &backendmetrics.MetricsState{
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
	pod2Metrics = &backendmetrics.MetricsState{
		WaitingQueueSize:    1,
		KVCacheUsagePercent: 0.2,
		MaxActiveModels:     2,
		ActiveModels: map[string]int{
			"foo1": 1,
			"bar1": 1,
		},
		WaitingModels: map[string]int{},
	}

	pod1NamespacedName = types.NamespacedName{Name: pod1.Name + "-rank-0", Namespace: pod1.Namespace}
	pod2NamespacedName = types.NamespacedName{Name: pod2.Name + "-rank-0", Namespace: pod2.Namespace}
	inferencePool      = &v1.InferencePool{
		Spec: v1.InferencePoolSpec{
			TargetPorts: []v1.Port{{Number: v1.PortNumber(int32(8000))}},
		},
	}
	inferencePoolMultiTarget = &v1.InferencePool{
		Spec: v1.InferencePoolSpec{
			TargetPorts: []v1.Port{{Number: v1.PortNumber(int32(8000))}, {Number: v1.PortNumber(int32(8001))}},
		},
	}

	inferencePoolTargetPort       = strconv.Itoa(int(inferencePool.Spec.TargetPorts[0].Number))
	inferencePoolMultiTargetPort0 = strconv.Itoa(int(inferencePoolMultiTarget.Spec.TargetPorts[0].Number))
	inferencePoolMultiTargetPort1 = strconv.Itoa(int(inferencePoolMultiTarget.Spec.TargetPorts[1].Number))
)

func TestMetrics(t *testing.T) {
	tests := []struct {
		name      string
		metrics   map[types.NamespacedName]*backendmetrics.MetricsState
		err       map[types.NamespacedName]error
		storePods []*corev1.Pod
		want      []*backendmetrics.MetricsState
		predict   func(backendmetrics.PodMetrics) bool
	}{
		{
			name: "Probing metrics success",
			metrics: map[types.NamespacedName]*backendmetrics.MetricsState{
				pod1NamespacedName: pod1Metrics,
				pod2NamespacedName: pod2Metrics,
			},
			storePods: []*corev1.Pod{pod1, pod2},
			want:      []*backendmetrics.MetricsState{pod1Metrics, pod2Metrics},
		},
		{
			name: "Only pods in are probed",
			metrics: map[types.NamespacedName]*backendmetrics.MetricsState{
				pod1NamespacedName: pod1Metrics,
				pod2NamespacedName: pod2Metrics,
			},
			storePods: []*corev1.Pod{pod1},
			want:      []*backendmetrics.MetricsState{pod1Metrics},
		},
		{
			name: "Probing metrics error",
			err: map[types.NamespacedName]error{
				pod2NamespacedName: errors.New("injected error"),
			},
			metrics: map[types.NamespacedName]*backendmetrics.MetricsState{
				pod1NamespacedName: pod1Metrics,
				pod2NamespacedName: pod2Metrics,
			},
			storePods: []*corev1.Pod{pod1, pod2},
			want: []*backendmetrics.MetricsState{pod1Metrics,
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
		period := time.Millisecond
		factories := []datalayer.EndpointFactory{
			backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{Res: test.metrics, Err: test.err}, period),
			datalayer.NewEndpointFactory([]datalayer.DataSource{&datalayer.FakeDataSource{Metrics: test.metrics, Errors: test.err}}, period),
		}
		for _, epf := range factories {
			t.Run(test.name, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				// Set up the scheme.
				scheme := runtime.NewScheme()
				_ = clientgoscheme.AddToScheme(scheme)
				fakeClient := fake.NewClientBuilder().
					WithScheme(scheme).
					Build()
				ds := NewDatastore(ctx, epf, 0)
				_ = ds.PoolSet(ctx, fakeClient, pooltuil.InferencePoolToEndpointPool(inferencePool))
				for _, pod := range test.storePods {
					ds.PodUpdateOrAddIfNotExist(pod)
				}
				time.Sleep(1 * time.Second) // Give some time for the metrics to be fetched.
				if test.predict == nil {
					test.predict = AllPodsPredicate
				}
				assert.EventuallyWithT(t, func(t *assert.CollectT) {
					got := ds.PodList(test.predict)
					metrics := []*backendmetrics.MetricsState{}
					for _, one := range got {
						metrics = append(metrics, one.GetMetrics())
					}
					diff := cmp.Diff(test.want, metrics, cmpopts.IgnoreFields(backendmetrics.MetricsState{}, "UpdateTime"), cmpopts.SortSlices(func(a, b *backendmetrics.MetricsState) bool {
						return a.String() < b.String()
					}))
					assert.Equal(t, "", diff, "Unexpected diff (+got/-want)")
				}, 5*time.Second, time.Millisecond)
			})
		}
	}
}

func TestPods(t *testing.T) {
	tests := []struct {
		name         string
		op           func(ctx context.Context, ds Datastore)
		existingPods []*corev1.Pod
		wantPods     []*corev1.Pod
	}{
		{
			name:         "Add new pod, no existing pods, should add",
			existingPods: []*corev1.Pod{},
			wantPods:     []*corev1.Pod{pod1},
			op: func(ctx context.Context, ds Datastore) {
				ds.PodUpdateOrAddIfNotExist(pod1)
			},
		},
		{
			name:         "Add new pod, with existing pods, should add",
			existingPods: []*corev1.Pod{pod1},
			wantPods:     []*corev1.Pod{pod1, pod2},
			op: func(ctx context.Context, ds Datastore) {
				ds.PodUpdateOrAddIfNotExist(pod2)
			},
		},
		{
			name:         "Delete the pod",
			existingPods: []*corev1.Pod{pod1, pod2},
			wantPods:     []*corev1.Pod{pod1},
			op: func(ctx context.Context, ds Datastore) {
				ds.PodDelete(pod2.Name)
			},
		},
		{
			name:         "Delete the pod that doesn't exist",
			existingPods: []*corev1.Pod{pod1},
			wantPods:     []*corev1.Pod{pod1},
			op: func(ctx context.Context, ds Datastore) {
				ds.PodDelete(pod2.Name)
			},
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
				ctx := context.Background()
				ds := NewDatastore(t.Context(), epf, 0)
				fakeClient := fake.NewFakeClient()
				if err := ds.PoolSet(ctx, fakeClient, pooltuil.InferencePoolToEndpointPool(inferencePool)); err != nil {
					t.Error(err)
				}
				for _, pod := range test.existingPods {
					ds.PodUpdateOrAddIfNotExist(pod)
				}

				test.op(ctx, ds)
				var gotPods []*corev1.Pod
				for _, pm := range ds.PodList(AllPodsPredicate) {
					pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: pm.GetMetadata().PodName, Namespace: pm.GetMetadata().NamespacedName.Namespace}, Status: corev1.PodStatus{PodIP: pm.GetMetadata().GetIPAddress()}}
					gotPods = append(gotPods, pod)
				}
				if !cmp.Equal(gotPods, test.wantPods, cmpopts.SortSlices(func(a, b *corev1.Pod) bool { return a.Name < b.Name })) {
					t.Errorf("got (%v) != want (%v);", gotPods, test.wantPods)
				}
			})
		}
	}
}

func TestEndpointMetadata(t *testing.T) {
	tests := []struct {
		name              string
		op                func(ctx context.Context, ds Datastore)
		pool              *v1.InferencePool
		existingPods      []*corev1.Pod
		wantEndpointMetas []*datalayer.EndpointMetadata
	}{
		{
			name:         "Add new pod, no existing pods, should add",
			existingPods: []*corev1.Pod{},
			wantEndpointMetas: []*datalayer.EndpointMetadata{
				{
					NamespacedName: types.NamespacedName{
						Name:      pod1.Name + "-rank-0",
						Namespace: pod1.Namespace,
					},

					PodName:     pod1.Name,
					Address:     pod1.Status.PodIP,
					Port:        inferencePoolTargetPort,
					MetricsHost: net.JoinHostPort(pod1.Status.PodIP, inferencePoolTargetPort),
					Labels:      map[string]string{},
				},
			},
			op: func(ctx context.Context, ds Datastore) {
				ds.PodUpdateOrAddIfNotExist(pod1)
			},
			pool: inferencePool,
		},
		{
			name:         "Add new pod, no existing pods, should add, multiple target ports",
			existingPods: []*corev1.Pod{},
			wantEndpointMetas: []*datalayer.EndpointMetadata{
				{
					NamespacedName: types.NamespacedName{
						Name:      pod1.Name + "-rank-0",
						Namespace: pod1.Namespace,
					},

					PodName:     pod1.Name,
					Address:     pod1.Status.PodIP,
					Port:        inferencePoolMultiTargetPort0,
					MetricsHost: net.JoinHostPort(pod1.Status.PodIP, inferencePoolMultiTargetPort0),
					Labels:      map[string]string{},
				},
				{
					NamespacedName: types.NamespacedName{
						Name:      pod1.Name + "-rank-1",
						Namespace: pod1.Namespace,
					},

					PodName:     pod1.Name,
					Address:     pod1.Status.PodIP,
					Port:        inferencePoolMultiTargetPort1,
					MetricsHost: net.JoinHostPort(pod1.Status.PodIP, inferencePoolMultiTargetPort1),
					Labels:      map[string]string{},
				},
			},
			op: func(ctx context.Context, ds Datastore) {
				ds.PodUpdateOrAddIfNotExist(pod1)
			},
			pool: inferencePoolMultiTarget,
		},
		{
			name:         "Add new pod, with existing pods, should add, multiple target ports",
			existingPods: []*corev1.Pod{pod1},
			wantEndpointMetas: []*datalayer.EndpointMetadata{
				{
					NamespacedName: types.NamespacedName{
						Name:      pod1.Name + "-rank-0",
						Namespace: pod1.Namespace,
					},

					PodName:     pod1.Name,
					Address:     pod1.Status.PodIP,
					Port:        inferencePoolMultiTargetPort0,
					MetricsHost: net.JoinHostPort(pod1.Status.PodIP, inferencePoolMultiTargetPort0),
					Labels:      map[string]string{},
				},
				{
					NamespacedName: types.NamespacedName{
						Name:      pod1.Name + "-rank-1",
						Namespace: pod1.Namespace,
					},

					PodName:     pod1.Name,
					Address:     pod1.Status.PodIP,
					Port:        inferencePoolMultiTargetPort1,
					MetricsHost: net.JoinHostPort(pod1.Status.PodIP, inferencePoolMultiTargetPort1),
					Labels:      map[string]string{},
				},
				{
					NamespacedName: types.NamespacedName{
						Name:      pod2.Name + "-rank-0",
						Namespace: pod2.Namespace,
					},

					PodName:     pod2.Name,
					Address:     pod2.Status.PodIP,
					Port:        inferencePoolMultiTargetPort0,
					MetricsHost: net.JoinHostPort(pod1.Status.PodIP, inferencePoolMultiTargetPort0),
					Labels:      map[string]string{},
				},
				{
					NamespacedName: types.NamespacedName{
						Name:      pod2.Name + "-rank-1",
						Namespace: pod2.Namespace,
					},

					PodName:     pod2.Name,
					Address:     pod2.Status.PodIP,
					Port:        inferencePoolMultiTargetPort1,
					MetricsHost: net.JoinHostPort(pod1.Status.PodIP, inferencePoolMultiTargetPort1),
					Labels:      map[string]string{},
				},
			},
			op: func(ctx context.Context, ds Datastore) {
				ds.PodUpdateOrAddIfNotExist(pod2)
			},
			pool: inferencePoolMultiTarget,
		},
		{
			name:         "Delete the pod, multiple target ports",
			existingPods: []*corev1.Pod{pod1, pod2},
			wantEndpointMetas: []*datalayer.EndpointMetadata{
				{
					NamespacedName: types.NamespacedName{
						Name:      pod1.Name + "-rank-0",
						Namespace: pod1.Namespace,
					},

					PodName:     pod1.Name,
					Address:     pod1.Status.PodIP,
					Port:        inferencePoolMultiTargetPort0,
					MetricsHost: net.JoinHostPort(pod1.Status.PodIP, inferencePoolMultiTargetPort0),
					Labels:      map[string]string{},
				},
				{
					NamespacedName: types.NamespacedName{
						Name:      pod1.Name + "-rank-1",
						Namespace: pod1.Namespace,
					},

					PodName:     pod1.Name,
					Address:     pod1.Status.PodIP,
					Port:        inferencePoolMultiTargetPort1,
					MetricsHost: net.JoinHostPort(pod1.Status.PodIP, inferencePoolMultiTargetPort1),
					Labels:      map[string]string{},
				},
			},
			op: func(ctx context.Context, ds Datastore) {
				ds.PodDelete(pod2.Name)
			},
			pool: inferencePoolMultiTarget,
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
				ctx := context.Background()
				ds := NewDatastore(t.Context(), epf, 0)
				fakeClient := fake.NewFakeClient()
				if err := ds.PoolSet(ctx, fakeClient, pooltuil.InferencePoolToEndpointPool(test.pool)); err != nil {
					t.Error(err)
				}
				for _, pod := range test.existingPods {
					ds.PodUpdateOrAddIfNotExist(pod)
				}

				test.op(ctx, ds)
				var gotMetadata []*datalayer.EndpointMetadata
				for _, pm := range ds.PodList(AllPodsPredicate) {
					gotMetadata = append(gotMetadata, pm.GetMetadata())
				}
				if diff := cmp.Diff(test.wantEndpointMetas, gotMetadata, cmpopts.SortSlices(func(a, b *datalayer.EndpointMetadata) bool { return a.NamespacedName.Name < b.NamespacedName.Name })); diff != "" {
					t.Errorf("ConvertTo() mismatch (-want +got):\n%s", diff)
				}
			})
		}
	}
}
