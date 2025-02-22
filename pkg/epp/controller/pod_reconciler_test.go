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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	utiltest "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

var (
	basePod1  = &datastore.PodMetrics{Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod1"}, Address: "address-1", ScrapePath: "/metrics", ScrapePort: 8000}}
	basePod2  = &datastore.PodMetrics{Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod2"}, Address: "address-2", ScrapePath: "/metrics", ScrapePort: 8000}}
	basePod3  = &datastore.PodMetrics{Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod3"}, Address: "address-3", ScrapePath: "/metrics", ScrapePort: 8000}}
	basePod11 = &datastore.PodMetrics{Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod1"}, Address: "address-11", ScrapePath: "/metrics", ScrapePort: 8000}}
)

func TestPodReconciler(t *testing.T) {
	tests := []struct {
		name        string
		datastore   datastore.Datastore
		incomingPod *corev1.Pod
		wantPods    []datastore.Pod
		req         *ctrl.Request
	}{
		{
			name: "Add new pod",
			datastore: datastore.NewFakeDatastore([]*datastore.PodMetrics{basePod1, basePod2}, nil, &v1alpha2.InferencePool{
				Spec: v1alpha2.InferencePoolSpec{
					TargetPortNumber: int32(8000),
					Selector: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
						"some-key": "some-val",
					},
				},
			}),
			incomingPod: utiltest.MakePod(basePod3.NamespacedName.Name).
				Labels(map[string]string{"some-key": "some-val"}).
				IP(basePod3.Address).
				ReadyCondition().ObjRef(),
			wantPods: []datastore.Pod{basePod1.Pod, basePod2.Pod, basePod3.Pod},
		},
		{
			name: "Update pod1 address",
			datastore: datastore.NewFakeDatastore([]*datastore.PodMetrics{basePod1, basePod2}, nil, &v1alpha2.InferencePool{
				Spec: v1alpha2.InferencePoolSpec{
					TargetPortNumber: int32(8000),
					Selector: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
						"some-key": "some-val",
					},
				},
			}),
			incomingPod: utiltest.MakePod(basePod11.NamespacedName.Name).
				Labels(map[string]string{"some-key": "some-val"}).
				IP(basePod11.Address).
				ReadyCondition().ObjRef(),
			wantPods: []datastore.Pod{basePod11.Pod, basePod2.Pod},
		},
		{
			name: "Delete pod with DeletionTimestamp",
			datastore: datastore.NewFakeDatastore([]*datastore.PodMetrics{basePod1, basePod2}, nil, &v1alpha2.InferencePool{
				Spec: v1alpha2.InferencePoolSpec{
					TargetPortNumber: int32(8000),
					Selector: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
						"some-key": "some-val",
					},
				},
			}),
			incomingPod: utiltest.MakePod("pod1").
				Labels(map[string]string{"some-key": "some-val"}).
				DeletionTimestamp().
				ReadyCondition().ObjRef(),
			wantPods: []datastore.Pod{basePod2.Pod},
		},
		{
			name: "Delete notfound pod",
			datastore: datastore.NewFakeDatastore([]*datastore.PodMetrics{basePod1, basePod2}, nil, &v1alpha2.InferencePool{
				Spec: v1alpha2.InferencePoolSpec{
					TargetPortNumber: int32(8000),
					Selector: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
						"some-key": "some-val",
					},
				},
			}),
			req:      &ctrl.Request{NamespacedName: types.NamespacedName{Name: "pod1"}},
			wantPods: []datastore.Pod{basePod2.Pod},
		},
		{
			name: "New pod, not ready, valid selector",
			datastore: datastore.NewFakeDatastore([]*datastore.PodMetrics{basePod1, basePod2}, nil, &v1alpha2.InferencePool{
				Spec: v1alpha2.InferencePoolSpec{
					TargetPortNumber: int32(8000),
					Selector: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
						"some-key": "some-val",
					},
				},
			}),
			incomingPod: utiltest.MakePod("pod3").
				Labels(map[string]string{"some-key": "some-val"}).ObjRef(),
			wantPods: []datastore.Pod{basePod1.Pod, basePod2.Pod},
		},
		{
			name: "Remove pod that does not match selector",
			datastore: datastore.NewFakeDatastore([]*datastore.PodMetrics{basePod1, basePod2}, nil, &v1alpha2.InferencePool{
				Spec: v1alpha2.InferencePoolSpec{
					TargetPortNumber: int32(8000),
					Selector: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
						"some-key": "some-val",
					},
				},
			}),
			incomingPod: utiltest.MakePod("pod1").
				Labels(map[string]string{"some-wrong-key": "some-val"}).
				ReadyCondition().ObjRef(),
			wantPods: []datastore.Pod{basePod2.Pod},
		},
		{
			name: "Remove pod that is not ready",
			datastore: datastore.NewFakeDatastore([]*datastore.PodMetrics{basePod1, basePod2}, nil, &v1alpha2.InferencePool{
				Spec: v1alpha2.InferencePoolSpec{
					TargetPortNumber: int32(8000),
					Selector: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
						"some-key": "some-val",
					},
				},
			}),
			incomingPod: utiltest.MakePod("pod1").
				Labels(map[string]string{"some-wrong-key": "some-val"}).
				ReadyCondition().ObjRef(),
			wantPods: []datastore.Pod{basePod2.Pod},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Set up the scheme.
			scheme := runtime.NewScheme()
			_ = clientgoscheme.AddToScheme(scheme)
			initialObjects := []client.Object{}
			if test.incomingPod != nil {
				initialObjects = append(initialObjects, test.incomingPod)
			}
			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(initialObjects...).
				Build()

			podReconciler := &PodReconciler{Client: fakeClient, Datastore: test.datastore}
			namespacedName := types.NamespacedName{Name: test.incomingPod.Name, Namespace: test.incomingPod.Namespace}
			if test.req == nil {
				test.req = &ctrl.Request{NamespacedName: namespacedName}
			}
			if _, err := podReconciler.Reconcile(context.Background(), *test.req); err != nil {
				t.Errorf("Unexpected InferencePool reconcile error: %v", err)
			}

			var gotPods []datastore.Pod
			test.datastore.PodRange(func(k, v any) bool {
				pod := v.(*datastore.PodMetrics)
				if v != nil {
					gotPods = append(gotPods, pod.Pod)
				}
				return true
			})
			if !cmp.Equal(gotPods, test.wantPods, cmpopts.SortSlices(func(a, b datastore.Pod) bool { return a.NamespacedName.String() < b.NamespacedName.String() })) {
				t.Errorf("got (%v) != want (%v);", gotPods, test.wantPods)
			}
		})
	}
}
