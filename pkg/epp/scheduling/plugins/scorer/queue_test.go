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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

func TestQueueScorer(t *testing.T) {
	tests := []struct {
		name              string
		pods              []types.Pod
		expectedScoresPod map[int]float64 // Map of pod index to expected score
	}{
		{
			name: "Different queue sizes",
			pods: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{}, Metrics: &backendmetrics.Metrics{WaitingQueueSize: 10}},
				&types.PodMetrics{Pod: &backend.Pod{}, Metrics: &backendmetrics.Metrics{WaitingQueueSize: 5}},
				&types.PodMetrics{Pod: &backend.Pod{}, Metrics: &backendmetrics.Metrics{WaitingQueueSize: 0}},
			},
			expectedScoresPod: map[int]float64{
				0: 0.0, // Longest queue (10) gets lowest score
				1: 0.5, // Medium queue (5) gets medium score
				2: 1.0, // Shortest queue (0) gets highest score
			},
		},
		{
			name: "Same queue sizes",
			pods: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{}, Metrics: &backendmetrics.Metrics{WaitingQueueSize: 5}},
				&types.PodMetrics{Pod: &backend.Pod{}, Metrics: &backendmetrics.Metrics{WaitingQueueSize: 5}},
			},
			expectedScoresPod: map[int]float64{
				0: 1.0, // When all pods have the same queue size, they get the same neutral score
				1: 1.0,
			},
		},
		{
			name: "Zero queue sizes",
			pods: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{}, Metrics: &backendmetrics.Metrics{WaitingQueueSize: 0}},
				&types.PodMetrics{Pod: &backend.Pod{}, Metrics: &backendmetrics.Metrics{WaitingQueueSize: 0}},
			},
			expectedScoresPod: map[int]float64{
				0: 1.0,
				1: 1.0,
			},
		},
	}

	scorer := &QueueScorer{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := types.NewSchedulingContext(context.Background(), &types.LLMRequest{}, nil, tt.pods)
			scores := scorer.Score(ctx, tt.pods)

			for i, pod := range tt.pods {
				expectedScore := tt.expectedScoresPod[i]
				assert.InDelta(t, expectedScore, scores[pod], 0.0001, "Pod %d should have score %f", i, expectedScore)
			}
		})
	}
}
