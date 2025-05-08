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

func TestKvCacheScorer(t *testing.T) {
	tests := []struct {
		name              string
		pods              []types.Pod
		expectedScoresPod map[int]float64 // Map of pod index to expected score
	}{
		{
			name: "Different KV cache utilization",
			pods: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{}, Metrics: &backendmetrics.Metrics{KVCacheUsagePercent: 0.8}},
				&types.PodMetrics{Pod: &backend.Pod{}, Metrics: &backendmetrics.Metrics{KVCacheUsagePercent: 0.5}},
				&types.PodMetrics{Pod: &backend.Pod{}, Metrics: &backendmetrics.Metrics{KVCacheUsagePercent: 0.0}},
			},
			expectedScoresPod: map[int]float64{
				0: 0.2, // Highest KV cache usage (0.8) gets lowest score (1-0.8=0.2)
				1: 0.5, // Medium KV cache usage (0.5) gets medium score (1-0.5=0.5)
				2: 1.0, // No KV cache usage (0.0) gets highest score (1-0=1.0)
			},
		},
		{
			name: "Same KV cache utilization",
			pods: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{}, Metrics: &backendmetrics.Metrics{KVCacheUsagePercent: 0.6}},
				&types.PodMetrics{Pod: &backend.Pod{}, Metrics: &backendmetrics.Metrics{KVCacheUsagePercent: 0.6}},
			},
			expectedScoresPod: map[int]float64{
				0: 0.4, // Both get same score (1-0.6=0.4)
				1: 0.4,
			},
		},
		{
			name: "Zero KV cache utilization",
			pods: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{}, Metrics: &backendmetrics.Metrics{KVCacheUsagePercent: 0.0}},
				&types.PodMetrics{Pod: &backend.Pod{}, Metrics: &backendmetrics.Metrics{KVCacheUsagePercent: 0.0}},
			},
			expectedScoresPod: map[int]float64{
				0: 1.0, // No KV cache usage gets highest score
				1: 1.0,
			},
		},
		{
			name: "Full KV cache utilization",
			pods: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{}, Metrics: &backendmetrics.Metrics{KVCacheUsagePercent: 1.0}},
				&types.PodMetrics{Pod: &backend.Pod{}, Metrics: &backendmetrics.Metrics{KVCacheUsagePercent: 0.5}},
			},
			expectedScoresPod: map[int]float64{
				0: 0.0, // Full KV cache (1.0) gets lowest score (1-1=0)
				1: 0.5, // Half KV cache (0.5) gets medium score (1-0.5=0.5)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := types.NewSchedulingContext(context.Background(), &types.LLMRequest{}, nil, tt.pods)
			scorer := &KVCacheScorer{}
			scores := scorer.Score(ctx, tt.pods)

			for i, pod := range tt.pods {
				expectedScore := tt.expectedScoresPod[i]
				assert.InDelta(t, expectedScore, scores[pod], 0.0001, "Pod %d should have score %f", i, expectedScore)
			}
		})
	}
}
