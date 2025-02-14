package backend

import (
	"testing"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
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
							Weight: pointer(50),
						},
						{
							Name:   "v1",
							Weight: pointer(50),
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
							Weight: pointer(25),
						},
						{
							Name:   "v1.1",
							Weight: pointer(55),
						},
						{
							Name:   "v1",
							Weight: pointer(50),
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
							Weight: pointer(20),
						},
						{
							Name:   "v1.1",
							Weight: pointer(20),
						},
						{
							Name:   "v1",
							Weight: pointer(10),
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

func pointer(v int32) *int32 {
	return &v
}
