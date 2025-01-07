package backend

import (
	"testing"

	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
)

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
