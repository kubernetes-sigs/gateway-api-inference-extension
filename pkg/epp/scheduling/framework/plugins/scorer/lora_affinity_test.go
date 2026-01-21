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

package scorer

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	types "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/scheduling"
)

func TestLoraAffinityScorer(t *testing.T) {
	tests := []struct {
		name                   string
		request                *types.LLMRequest
		endpoints              []types.Endpoint
		expectedScoresEndpoint map[string]float64 // Map of endpoint name to expected score
	}{
		{
			name:    "Target model is active",
			request: &types.LLMRequest{TargetModel: "active-model-1"},
			endpoints: []types.Endpoint{
				&types.PodMetrics{
					EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
					Metrics: &datalayer.Metrics{
						ActiveModels:    map[string]int{"active-model-1": 1},
						WaitingModels:   map[string]int{},
						MaxActiveModels: 5,
					},
				},
			},
			expectedScoresEndpoint: map[string]float64{
				"pod1": 1.0,
			},
		},
		{
			name:    "Target model is waiting",
			request: &types.LLMRequest{TargetModel: "active-model-1"},
			endpoints: []types.Endpoint{
				&types.PodMetrics{
					EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
					Metrics: &datalayer.Metrics{
						ActiveModels:    map[string]int{"active-model-2": 2},
						WaitingModels:   map[string]int{"active-model-1": 1},
						MaxActiveModels: 2,
					},
				},
			},
			expectedScoresEndpoint: map[string]float64{
				"pod1": 0.6,
			},
		},
		{
			name:    "Endpoints have no space for new model",
			request: &types.LLMRequest{TargetModel: "active-model-1"},
			endpoints: []types.Endpoint{
				&types.PodMetrics{
					EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
					Metrics: &datalayer.Metrics{
						ActiveModels:    map[string]int{"active-model-2": 2},
						WaitingModels:   map[string]int{"active-model-3": 1},
						MaxActiveModels: 2,
					},
				},
				&types.PodMetrics{
					EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}},
					Metrics: &datalayer.Metrics{
						ActiveModels:    map[string]int{},
						WaitingModels:   map[string]int{},
						MaxActiveModels: 0,
					},
				},
			},
			expectedScoresEndpoint: map[string]float64{
				"pod1": 0.0,
				"pod2": 0.0,
			},
		},
		{
			name:    "Multipleendpoints with mixed active and waiting models",
			request: &types.LLMRequest{TargetModel: "active-model-1"},
			endpoints: []types.Endpoint{
				&types.PodMetrics{
					EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
					Metrics: &datalayer.Metrics{
						ActiveModels:    map[string]int{"active-model-1": 1},
						WaitingModels:   map[string]int{},
						MaxActiveModels: 5,
					},
				},
				&types.PodMetrics{
					EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}},
					Metrics: &datalayer.Metrics{
						ActiveModels:    map[string]int{"active-model-2": 4},
						WaitingModels:   map[string]int{"active-model-1": 1},
						MaxActiveModels: 5,
					},
				},
				&types.PodMetrics{
					EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}},
					Metrics: &datalayer.Metrics{
						ActiveModels:    map[string]int{"active-model-2": 1},
						WaitingModels:   map[string]int{},
						MaxActiveModels: 2,
					},
				},
				&types.PodMetrics{
					EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod4"}},
					Metrics: &datalayer.Metrics{
						ActiveModels:    map[string]int{"active-model-3": 1},
						WaitingModels:   map[string]int{"active-model-1": 1},
						MaxActiveModels: 2,
					},
				},
				&types.PodMetrics{
					EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod5"}},
					Metrics: &datalayer.Metrics{
						ActiveModels:    map[string]int{"active-model-4": 1, "active-model-5": 1},
						WaitingModels:   map[string]int{},
						MaxActiveModels: 2,
					},
				},
			},
			expectedScoresEndpoint: map[string]float64{
				"pod1": 1.0,
				"pod2": 0.8,
				"pod3": 0.8,
				"pod4": 0.6,
				"pod5": 0.0,
			},
		},
		{
			name:                   "Empty pods slice",
			request:                &types.LLMRequest{TargetModel: "modelA"},
			endpoints:              []types.Endpoint{},
			expectedScoresEndpoint: map[string]float64{}, // No pods, no scores
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scorer := &LoraAffinityScorer{}
			scores := scorer.Score(context.Background(), types.NewCycleState(), test.request, test.endpoints)

			for _, endpoint := range test.endpoints {
				expectedScore, ok := test.expectedScoresEndpoint[endpoint.GetMetadata().NamespacedName.Name]
				if !ok {
					t.Fatalf("Expected score not found for endpoint %s in test %s", endpoint.GetMetadata().NamespacedName, test.name)
				}
				assert.InDelta(t, expectedScore, scores[endpoint], 0.0001, "Endpoint %s should have score %f", endpoint.GetMetadata().NamespacedName.Name, expectedScore)
			}
			assert.Len(t, scores, len(test.expectedScoresEndpoint), "Number of scored endpoints should match expected")
		})
	}
}
