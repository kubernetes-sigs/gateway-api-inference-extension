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

package handlers

import (
	"testing"

	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func TestRandomWeightedDraw(t *testing.T) {
	logger := logutil.NewTestLogger()
	tests := []struct {
		name  string
		model *v1alpha2.InferenceModel
		want  string
	}{
		{
			name: "'random' distribution",
			model: &v1alpha2.InferenceModel{
				Spec: v1alpha2.InferenceModelSpec{
					TargetModels: []v1alpha2.TargetModel{
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
			model: &v1alpha2.InferenceModel{
				Spec: v1alpha2.InferenceModelSpec{
					TargetModels: []v1alpha2.TargetModel{
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
			model: &v1alpha2.InferenceModel{
				Spec: v1alpha2.InferenceModelSpec{
					TargetModels: []v1alpha2.TargetModel{
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
		{
			name: "weighted distribution with weight unset",
			model: &v1alpha2.InferenceModel{
				Spec: v1alpha2.InferenceModelSpec{
					TargetModels: []v1alpha2.TargetModel{
						{
							Name: "canary",
						},
						{
							Name: "v1.1",
						},
						{
							Name: "v1",
						},
					},
				},
			},
			want: "canary",
		},
	}
	var seedVal int64 = 420
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for range 10000 {
				model := RandomWeightedDraw(logger, test.model, seedVal)
				if model != test.want {
					t.Errorf("Model returned: %v != %v", model, test.want)
					break
				}
			}
		})
	}
}

func pointer(v int32) *int32 {
	return &v
}
