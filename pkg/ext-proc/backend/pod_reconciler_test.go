package backend

import (
	"sync"
	"testing"

<<<<<<< HEAD:pkg/ext-proc/backend/endpointslice_reconcilier_test.go
=======
	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
<<<<<<< HEAD
>>>>>>> d5d500b (reversion to pod reconciliation):pkg/ext-proc/backend/pod_reconciler_test.go
	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
=======
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
>>>>>>> 8042d21 (adding ready check and unit tests)
)

var (
	basePod1 = Pod{Name: "pod1"}
	basePod2 = Pod{Name: "pod2"}
	basePod3 = Pod{Name: "pod3"}
)

func TestUpdateDatastore_EndpointSliceReconciler(t *testing.T) {
	tests := []struct {
		name        string
		datastore   *K8sDatastore
		incomingPod *corev1.Pod
		wantPods    *sync.Map
	}{
		{
			name: "Add new pod",
			datastore: &K8sDatastore{
				pods: populateMap(basePod1, basePod2),
				inferencePool: &v1alpha1.InferencePool{
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
			},
			wantPods: populateMap(basePod1, basePod2, basePod3),
		},
		{
			name: "New pod, but its not ready yet. Do not add.",
			datastore: &K8sDatastore{
				pods: populateMap(basePod1, basePod2),
				inferencePool: &v1alpha1.InferencePool{
					Spec: v1alpha1.InferencePoolSpec{
						TargetPortNumber: int32(8000),
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
			wantPods: populateMap(basePod1, basePod2),
		},
		{
			name: "Existing pod not ready, remove",
			datastore: &K8sDatastore{
				pods: populateMap(basePod1, basePod2),
				inferencePool: &v1alpha1.InferencePool{
					Spec: v1alpha1.InferencePoolSpec{
						TargetPortNumber: int32(8000),
					},
				},
			},
			incomingPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
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
			wantPods: populateMap(basePod2),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			endpointSliceReconciler := &PodReconciler{Datastore: test.datastore}
			endpointSliceReconciler.updateDatastore(test.incomingPod, test.datastore.inferencePool)

			if mapsEqual(endpointSliceReconciler.Datastore.pods, test.wantPods) {
				t.Errorf("Unexpected output pod mismatch. \n Got %v \n Want: %v \n",
					endpointSliceReconciler.Datastore.pods,
					test.wantPods)
			}
		})
	}
}

func truePointer() *bool {
	primitivePointersAreSilly := true
	return &primitivePointersAreSilly
}
