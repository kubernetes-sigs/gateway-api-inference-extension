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

package datastore

import (
	"testing"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
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
			datastore := NewDatastore()
			// Set the inference pool
			if tt.inferencePool != nil {
				datastore.PoolSet(tt.inferencePool)
			}
			// Check if the data store has been initialized
			hasSynced := datastore.PoolHasSynced()
			if hasSynced != tt.hasSynced {
				t.Errorf("IsInitialized() = %v, want %v", hasSynced, tt.hasSynced)
			}
		})
	}
}

func TestRandomWeightedDraw(t *testing.T) {
	logger := logutil.NewTestLogger()
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
				model := RandomWeightedDraw(logger, test.model, seedVal)
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
