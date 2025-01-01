package backend

import (
	"testing"

	"inference.networking.x-k8s.io/llm-instance-gateway/api/v1alpha1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestIsReady(t *testing.T) {
	tests := []struct {
		name            string
		inferencePool   *v1alpha1.InferencePool
		inferenceModels []*v1alpha1.InferenceModel
		expectedReady   bool
	}{
		{
			name: "Ready when at least one model matches configured pool",
			inferencePool: &v1alpha1.InferencePool{
				ObjectMeta: v1.ObjectMeta{
					Name:      "test-pool",
					Namespace: "default",
				},
			},
			inferenceModels: []*v1alpha1.InferenceModel{
				{
					ObjectMeta: v1.ObjectMeta{
						Name:      "test-model",
						Namespace: "default",
					},
					Spec: v1alpha1.InferenceModelSpec{
						PoolRef: v1alpha1.PoolObjectReference{
							Name: "other-pool",
						},
					},
				},
				{
					ObjectMeta: v1.ObjectMeta{
						Name:      "test-model",
						Namespace: "default",
					},
					Spec: v1alpha1.InferenceModelSpec{
						PoolRef: v1alpha1.PoolObjectReference{
							Name: "test-pool",
						},
					},
				},
			},
			expectedReady: true,
		},
		{
			name: "Not ready when model references non-matching pool",
			inferencePool: &v1alpha1.InferencePool{
				ObjectMeta: v1.ObjectMeta{
					Name:      "test-pool",
					Namespace: "default",
				},
			},
			inferenceModels: []*v1alpha1.InferenceModel{
				{
					ObjectMeta: v1.ObjectMeta{
						Name:      "test-model",
						Namespace: "default",
					},
					Spec: v1alpha1.InferenceModelSpec{
						PoolRef: v1alpha1.PoolObjectReference{
							Name: "other-pool",
						},
					},
				},
			},
			expectedReady: false,
		},
		{
			name:          "Not ready when pool is nil",
			inferencePool: nil,
			inferenceModels: []*v1alpha1.InferenceModel{
				{Spec: v1alpha1.InferenceModelSpec{}},
				{Spec: v1alpha1.InferenceModelSpec{}},
			},
			expectedReady: false,
		},
		{
			name:          "Not ready when models are missing",
			inferencePool: &v1alpha1.InferencePool{},
			inferenceModels: []*v1alpha1.InferenceModel{
				nil,
			},
			expectedReady: false,
		},
		{
			name:            "Not ready when models are empty",
			inferencePool:   &v1alpha1.InferencePool{},
			inferenceModels: []*v1alpha1.InferenceModel{},
			expectedReady:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			datastore := NewK8sDataStore()

			// Set the inference pool
			if tt.inferencePool != nil {
				datastore.setInferencePool(tt.inferencePool)
			}

			// Set the inference models
			for _, model := range tt.inferenceModels {
				if model != nil {
					datastore.InferenceModels.Store(model.Spec.ModelName, model)
				}
			}

			// Check readiness
			isReady := datastore.IsReady()
			if isReady != tt.expectedReady {
				t.Errorf("IsReady() = %v, want %v", isReady, tt.expectedReady)
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
