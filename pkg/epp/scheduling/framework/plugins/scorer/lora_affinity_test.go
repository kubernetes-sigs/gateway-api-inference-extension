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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

func TestLoraAffinityScorer(t *testing.T) {
	tests := []struct {
		name              string
		request           *types.LLMRequest
		pods              []types.Pod
		expectedScoresPod map[string]float64 // Map of pod name to expected score
	}{
		{
			name:    "Target model is active",
			request: &types.LLMRequest{TargetModel: "active-model-1"},
			pods: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
					MetricsState: &backendmetrics.MetricsState{
						ActiveModels:    map[string]int{"active-model-1": 1},
						WaitingModels:   map[string]int{},
						MaxActiveModels: 5,
					},
				},
			},
			expectedScoresPod: map[string]float64{
				"pod1": 1.0,
			},
		},
		{
			name:    "Target model is waiting",
			request: &types.LLMRequest{TargetModel: "active-model-1"},
			pods: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
					MetricsState: &backendmetrics.MetricsState{
						ActiveModels:    map[string]int{"active-model-2": 2},
						WaitingModels:   map[string]int{"active-model-1": 1},
						MaxActiveModels: 2,
					},
				},
			},
			expectedScoresPod: map[string]float64{
				"pod1": 0.6,
			},
		},
		{
			name:    "Pods have no space for new model",
			request: &types.LLMRequest{TargetModel: "active-model-1"},
			pods: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
					MetricsState: &backendmetrics.MetricsState{
						ActiveModels:    map[string]int{"active-model-2": 2},
						WaitingModels:   map[string]int{"active-model-3": 1},
						MaxActiveModels: 2,
					},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}},
					MetricsState: &backendmetrics.MetricsState{
						ActiveModels:    map[string]int{},
						WaitingModels:   map[string]int{},
						MaxActiveModels: 0,
					},
				},
			},
			expectedScoresPod: map[string]float64{
				"pod1": 0.0,
				"pod2": 0.0,
			},
		},
		{
			name:    "Multiple pods with mixed active and waiting models",
			request: &types.LLMRequest{TargetModel: "active-model-1"},
			pods: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
					MetricsState: &backendmetrics.MetricsState{
						ActiveModels:    map[string]int{"active-model-1": 1},
						WaitingModels:   map[string]int{},
						MaxActiveModels: 5,
					},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}},
					MetricsState: &backendmetrics.MetricsState{
						ActiveModels:    map[string]int{"active-model-2": 4},
						WaitingModels:   map[string]int{"active-model-1": 1},
						MaxActiveModels: 5,
					},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}},
					MetricsState: &backendmetrics.MetricsState{
						ActiveModels:    map[string]int{"active-model-2": 1},
						WaitingModels:   map[string]int{},
						MaxActiveModels: 2,
					},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod4"}},
					MetricsState: &backendmetrics.MetricsState{
						ActiveModels:    map[string]int{"active-model-3": 1},
						WaitingModels:   map[string]int{"active-model-1": 1},
						MaxActiveModels: 2,
					},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod5"}},
					MetricsState: &backendmetrics.MetricsState{
						ActiveModels:    map[string]int{"active-model-4": 1, "active-model-5": 1},
						WaitingModels:   map[string]int{},
						MaxActiveModels: 2,
					},
				},
			},
			expectedScoresPod: map[string]float64{
				"pod1": 1.0,
				"pod2": 0.8,
				"pod3": 0.8,
				"pod4": 0.6,
				"pod5": 0.0,
			},
		},
		{
			name:              "Empty pods slice",
			request:           &types.LLMRequest{TargetModel: "modelA"},
			pods:              []types.Pod{},
			expectedScoresPod: map[string]float64{}, // No pods, no scores
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scorer := &LoraAffinityScorer{}
			scores := scorer.Score(context.Background(), types.NewCycleState(), test.request, test.pods)

			for _, pod := range test.pods {
				expectedScore, ok := test.expectedScoresPod[pod.GetPod().NamespacedName.Name]
				if !ok {
					t.Fatalf("Expected score not found for pod %s in test %s", pod.GetPod().NamespacedName, test.name)
				}
				assert.InDelta(t, expectedScore, scores[pod], 0.0001, "Pod %s should have score %f", pod.GetPod().NamespacedName.Name, expectedScore)
			}
			assert.Len(t, scores, len(test.expectedScoresPod), "Number of scored pods should match expected")
		})
	}
}
