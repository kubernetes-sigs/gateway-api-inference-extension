package backend

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var (
	basePod1 = Pod{Name: "pod1", Address: ":8000"}
	basePod2 = Pod{Name: "pod2", Address: ":8000"}
	basePod3 = Pod{Name: "pod3", Address: ":8000"}
)

func TestUpdateDatastore_EndpointSliceReconciler(t *testing.T) {
	tests := []struct {
		name        string
		datastore   *K8sDatastore
		incomingPod *corev1.Pod
		wantPods    []string
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
			wantPods: []string{basePod1.Name, basePod2.Name, basePod3.Name},
		},
		{
			name: "Remove pod that does not match selector",
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
					Name: "pod1",
					Labels: map[string]string{
						"some-wrong-key": "some-val",
					},
				},
			},
			wantPods: []string{basePod2.Name},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			podReconciler := &PodReconciler{Datastore: test.datastore}
			podReconciler.updateDatastore(test.incomingPod, test.datastore.inferencePool)
			var gotPods []string
			test.datastore.pods.Range(func(k, v any) bool {
				pod := k.(Pod)
				if v != nil {
					gotPods = append(gotPods, pod.Name)
				}
				return true
			})
			if !cmp.Equal(gotPods, test.wantPods, cmpopts.SortSlices(func(a, b string) bool { return a < b })) {
				t.Errorf("got (%v) != want (%v);", gotPods, test.wantPods)
			}
		})
	}
}
