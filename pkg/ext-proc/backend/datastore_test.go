package backend

import (
	"testing"

	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestHasSynced(t *testing.T) {
	tests := []struct {
		name          string
		inferencePool *v1alpha1.InferencePool
		hasSynced     bool
	}{
		{
			name: "Ready when InferencePool exists in data store",
			inferencePool: &v1alpha1.InferencePool{
				ObjectMeta: v1.ObjectMeta{
					Name:      "test-pool",
					Namespace: "default",
				},
			},
			hasSynced: true,
		},
		{
			name:          "Not ready when InferencePool is nil in data store",
			inferencePool: nil,
			hasSynced:     false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			datastore := NewK8sDataStore()
			// Set the inference pool
			if tt.inferencePool != nil {
				datastore.setInferencePool(tt.inferencePool)
			}
			// Check if the data store has been initialized
			hasSynced := datastore.HasSynced()
			if hasSynced != tt.hasSynced {
				t.Errorf("IsInitialized() = %v, want %v", hasSynced, tt.hasSynced)
			}
		})
	}
}

func TestRandomWeightedDraw(t *testing.T) {
	tests := []struct {
		name  string
		model *v1alpha1.InferenceModel
		want  string
	}{
		{
			name: "'random' distribution",
			model: &v1alpha1.InferenceModel{
				Spec: v1alpha1.InferenceModelSpec{
					TargetModels: []v1alpha1.TargetModel{
						{
							Name:   "canary",
							Weight: 50,
						},
						{
							Name:   "v1",
							Weight: 50,
						},
					},
				},
			},
			want: "canary",
		},
		{
			name: "'random' distribution",
			model: &v1alpha1.InferenceModel{
				Spec: v1alpha1.InferenceModelSpec{
					TargetModels: []v1alpha1.TargetModel{
						{
							Name:   "canary",
							Weight: 25,
						},
						{
							Name:   "v1.1",
							Weight: 55,
						},
						{
							Name:   "v1",
							Weight: 50,
						},
					},
				},
			},
			want: "v1",
		},
		{
			name: "'random' distribution",
			model: &v1alpha1.InferenceModel{
				Spec: v1alpha1.InferenceModelSpec{
					TargetModels: []v1alpha1.TargetModel{
						{
							Name:   "canary",
							Weight: 20,
						},
						{
							Name:   "v1.1",
							Weight: 20,
						},
						{
							Name:   "v1",
							Weight: 10,
						},
					},
				},
			},
			want: "v1.1",
		},
	}
	var seedVal int64 = 420
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for range 10000 {
				model := RandomWeightedDraw(test.model, seedVal)
				if model != test.want {
					t.Errorf("Model returned!: %v", model)
					break
				}
			}
		})
	}
}
