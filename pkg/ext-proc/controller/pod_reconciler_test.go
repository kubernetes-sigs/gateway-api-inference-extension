package controller

import (
	"context"
	"sync"
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
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/datastore"
)

var (
	basePod1  = &datastore.PodMetrics{Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod1"}, Address: "address-1"}}
	basePod2  = &datastore.PodMetrics{Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod2"}, Address: "address-2"}}
	basePod3  = &datastore.PodMetrics{Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod3"}, Address: "address-3"}}
	basePod11 = &datastore.PodMetrics{Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod1"}, Address: "address-11"}}
)

func TestUpdateDatastore_PodReconciler(t *testing.T) {
	now := metav1.Now()
	tests := []struct {
		name        string
		datastore   datastore.Datastore
		incomingPod *corev1.Pod
		wantPods    []datastore.Pod
		req         *ctrl.Request
	}{
		{
			name: "Add new pod",
			datastore: datastore.NewFakeDatastore(populateMap(basePod1, basePod2), nil, &v1alpha1.InferencePool{
				Spec: v1alpha1.InferencePoolSpec{
					TargetPortNumber: int32(8000),
					Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{
						"some-key": "some-val",
					},
				},
			}),
			incomingPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: basePod3.NamespacedName.Name,
					Labels: map[string]string{
						"some-key": "some-val",
					},
				},
				Status: corev1.PodStatus{
					PodIP: basePod3.Address,
					Conditions: []corev1.PodCondition{
						{
							Type:   corev1.PodReady,
							Status: corev1.ConditionTrue,
						},
					},
				},
			},
			wantPods: []datastore.Pod{basePod1.Pod, basePod2.Pod, basePod3.Pod},
		},
		{
			name: "Update pod1 address",
			datastore: datastore.NewFakeDatastore(populateMap(basePod1, basePod2), nil, &v1alpha1.InferencePool{
				Spec: v1alpha1.InferencePoolSpec{
					TargetPortNumber: int32(8000),
					Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{
						"some-key": "some-val",
					},
				},
			}),
			incomingPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: basePod11.NamespacedName.Name,
					Labels: map[string]string{
						"some-key": "some-val",
					},
				},
				Status: corev1.PodStatus{
					PodIP: basePod11.Address,
					Conditions: []corev1.PodCondition{
						{
							Type:   corev1.PodReady,
							Status: corev1.ConditionTrue,
						},
					},
				},
			},
			wantPods: []datastore.Pod{basePod11.Pod, basePod2.Pod},
		},
		{
			name: "Delete pod with DeletionTimestamp",
			datastore: datastore.NewFakeDatastore(populateMap(basePod1, basePod2), nil, &v1alpha1.InferencePool{
				Spec: v1alpha1.InferencePoolSpec{
					TargetPortNumber: int32(8000),
					Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{
						"some-key": "some-val",
					},
				},
			}),
			incomingPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
					Labels: map[string]string{
						"some-key": "some-val",
					},
					DeletionTimestamp: &now,
					Finalizers:        []string{"finalizer"},
				},
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{
						{
							Type:   corev1.PodReady,
							Status: corev1.ConditionTrue,
						},
					},
				},
			},
			wantPods: []datastore.Pod{basePod2.Pod},
		},
		{
			name: "Delete notfound pod",
			datastore: datastore.NewFakeDatastore(populateMap(basePod1, basePod2), nil, &v1alpha1.InferencePool{
				Spec: v1alpha1.InferencePoolSpec{
					TargetPortNumber: int32(8000),
					Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{
						"some-key": "some-val",
					},
				},
			}),
			req:      &ctrl.Request{NamespacedName: types.NamespacedName{Name: "pod1"}},
			wantPods: []datastore.Pod{basePod2.Pod},
		},
		{
			name: "New pod, not ready, valid selector",
			datastore: datastore.NewFakeDatastore(populateMap(basePod1, basePod2), nil, &v1alpha1.InferencePool{
				Spec: v1alpha1.InferencePoolSpec{
					TargetPortNumber: int32(8000),
					Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{
						"some-key": "some-val",
					},
				},
			}),
			incomingPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod3",
					Labels: map[string]string{
						"some-key": "some-val",
					},
				},
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{
						{
							Type:   corev1.PodReady,
							Status: corev1.ConditionFalse,
						},
					},
				},
			},
			wantPods: []datastore.Pod{basePod1.Pod, basePod2.Pod},
		},
		{
			name: "Remove pod that does not match selector",
			datastore: datastore.NewFakeDatastore(populateMap(basePod1, basePod2), nil, &v1alpha1.InferencePool{
				Spec: v1alpha1.InferencePoolSpec{
					TargetPortNumber: int32(8000),
					Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{
						"some-key": "some-val",
					},
				},
			}),
			incomingPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
					Labels: map[string]string{
						"some-wrong-key": "some-val",
					},
				},
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{
						{
							Type:   corev1.PodReady,
							Status: corev1.ConditionTrue,
						},
					},
				},
			},
			wantPods: []datastore.Pod{basePod2.Pod},
		},
		{
			name: "Remove pod that is not ready",
			datastore: datastore.NewFakeDatastore(populateMap(basePod1, basePod2), nil, &v1alpha1.InferencePool{
				Spec: v1alpha1.InferencePoolSpec{
					TargetPortNumber: int32(8000),
					Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{
						"some-key": "some-val",
					},
				},
			}),
			incomingPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
					Labels: map[string]string{
						"some-wrong-key": "some-val",
					},
				},
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{
						{
							Type:   corev1.PodReady,
							Status: corev1.ConditionFalse,
						},
					},
				},
			},
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

func populateMap(pods ...*datastore.PodMetrics) *sync.Map {
	newMap := &sync.Map{}
	for _, pod := range pods {
		newMap.Store(pod.NamespacedName, &datastore.PodMetrics{Pod: datastore.Pod{NamespacedName: pod.NamespacedName, Address: pod.Address}})
	}
	return newMap
}
