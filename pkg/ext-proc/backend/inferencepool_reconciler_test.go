package backend

import (
	"reflect"
	"testing"

	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var (
	pool1 = &v1alpha1.InferencePool{
		Spec: v1alpha1.InferencePoolSpec{
			Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{"app": "vllm"},
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test-pool",
			ResourceVersion: "50",
		},
	}
	// Different name, same RV doesn't really make sense, but helps with testing the
	// updateStore impl which relies on the equality of RVs alone.
	modPool1SameRV = &v1alpha1.InferencePool{
		Spec: v1alpha1.InferencePoolSpec{
			Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{"app": "vllm"},
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test-pool-mod",
			ResourceVersion: "50",
		},
	}
	modPool1DiffRV = &v1alpha1.InferencePool{
		Spec: v1alpha1.InferencePoolSpec{
			Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{"app": "vllm"},
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test-pool-mod",
			ResourceVersion: "51",
		},
	}
)

func TestUpdateDatastore_InferencePoolReconciler(t *testing.T) {
	tests := []struct {
		name         string
		datastore    *K8sDatastore
		incomingPool *v1alpha1.InferencePool
		wantPool     *v1alpha1.InferencePool
	}{
		{
			name:         "InferencePool not set, should set InferencePool",
			datastore:    &K8sDatastore{},
			incomingPool: pool1.DeepCopy(),
			wantPool:     pool1,
		},
		{
			name: "InferencePool set, matching RVs, do nothing",
			datastore: &K8sDatastore{
				inferencePool: pool1.DeepCopy(),
			},
			incomingPool: modPool1SameRV.DeepCopy(),
			wantPool:     pool1,
		},
		{
			name: "InferencePool set, differing RVs, re-set InferencePool",
			datastore: &K8sDatastore{
				inferencePool: pool1.DeepCopy(),
			},
			incomingPool: modPool1DiffRV.DeepCopy(),
			wantPool:     modPool1DiffRV,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			inferencePoolReconciler := &InferencePoolReconciler{Datastore: test.datastore}
			inferencePoolReconciler.updateDatastore(test.incomingPool)

			gotPool := inferencePoolReconciler.Datastore.inferencePool
			if !reflect.DeepEqual(gotPool, test.wantPool) {
				t.Errorf("Unexpected InferencePool: want %#v, got: %#v", test.wantPool, gotPool)
			}
		})
	}
}
