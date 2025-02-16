package backend

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
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
)

var (
	basePod1 = &PodMetrics{NamespacedName: types.NamespacedName{Name: "pod1"}, Address: ":8000"}
	basePod2 = &PodMetrics{NamespacedName: types.NamespacedName{Name: "pod2"}, Address: ":8000"}
	basePod3 = &PodMetrics{NamespacedName: types.NamespacedName{Name: "pod3"}, Address: ":8000"}
)

func TestUpdateDatastore_PodReconciler(t *testing.T) {
	now := metav1.Now()
	tests := []struct {
		name        string
		datastore   Datastore
		incomingPod *corev1.Pod
		wantPods    []types.NamespacedName
		req         *ctrl.Request
	}{
		{
			name: "Add new pod",
			datastore: &datastore{
				pods: populateMap(basePod1, basePod2),
				pool: &v1alpha1.InferencePool{
					Spec: v1alpha1.InferencePoolSpec{
						TargetPortNumber: int32(8000),
						Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{
							"some-key": "some-val",
						},
					},
				},
			},
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
							Status: corev1.ConditionTrue,
						},
					},
				},
			},
			wantPods: []types.NamespacedName{basePod1.NamespacedName, basePod2.NamespacedName, basePod3.NamespacedName},
		},
		{
			name: "Delete pod with DeletionTimestamp",
			datastore: &datastore{
				pods: populateMap(basePod1, basePod2),
				pool: &v1alpha1.InferencePool{
					Spec: v1alpha1.InferencePoolSpec{
						TargetPortNumber: int32(8000),
						Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{
							"some-key": "some-val",
						},
					},
				},
			},
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
			wantPods: []types.NamespacedName{basePod2.NamespacedName},
		},
		{
			name: "Delete notfound pod",
			datastore: &datastore{
				pods: populateMap(basePod1, basePod2),
				pool: &v1alpha1.InferencePool{
					Spec: v1alpha1.InferencePoolSpec{
						TargetPortNumber: int32(8000),
						Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{
							"some-key": "some-val",
						},
					},
				},
			},
			req:      &ctrl.Request{NamespacedName: types.NamespacedName{Name: "pod1"}},
			wantPods: []types.NamespacedName{basePod2.NamespacedName},
		},
		{
			name: "New pod, not ready, valid selector",
			datastore: &datastore{
				pods: populateMap(basePod1, basePod2),
				pool: &v1alpha1.InferencePool{
					Spec: v1alpha1.InferencePoolSpec{
						TargetPortNumber: int32(8000),
						Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{
							"some-key": "some-val",
						},
					},
				},
			},
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
			wantPods: []types.NamespacedName{basePod1.NamespacedName, basePod2.NamespacedName},
		},
		{
			name: "Remove pod that does not match selector",
			datastore: &datastore{
				pods: populateMap(basePod1, basePod2),
				pool: &v1alpha1.InferencePool{
					Spec: v1alpha1.InferencePoolSpec{
						TargetPortNumber: int32(8000),
						Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{
							"some-key": "some-val",
						},
					},
				},
			},
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
			wantPods: []types.NamespacedName{basePod2.NamespacedName},
		},
		{
			name: "Remove pod that is not ready",
			datastore: &datastore{
				pods: populateMap(basePod1, basePod2),
				pool: &v1alpha1.InferencePool{
					Spec: v1alpha1.InferencePoolSpec{
						TargetPortNumber: int32(8000),
						Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{
							"some-key": "some-val",
						},
					},
				},
			},
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
			wantPods: []types.NamespacedName{basePod2.NamespacedName},
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

			var gotPods []types.NamespacedName
			test.datastore.PodRange(func(k, v any) bool {
				pod := v.(*PodMetrics)
				if v != nil {
					gotPods = append(gotPods, pod.NamespacedName)
				}
				return true
			})
			if !cmp.Equal(gotPods, test.wantPods, cmpopts.SortSlices(func(a, b types.NamespacedName) bool { return a.String() < b.String() })) {
				t.Errorf("got (%v) != want (%v);", gotPods, test.wantPods)
			}
		})
	}
}
