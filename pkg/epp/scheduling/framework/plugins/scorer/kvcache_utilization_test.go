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
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	types "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestKvCacheUtilizationScorer(t *testing.T) {
	tests := []struct {
		name                   string
		normalizeScore         bool
		endpoints              []types.Endpoint
		expectedScoresEndpoint map[int]float64 // Map of endpoint index to expected score
	}{
		{
			name:           "Absolute: Different KV cache utilization",
			normalizeScore: false,
			endpoints: []types.Endpoint{
				&types.PodMetrics{EndpointMetadata: &fwkdl.EndpointMetadata{}, Metrics: &datalayer.Metrics{KVCacheUsagePercent: 0.8}},
				&types.PodMetrics{EndpointMetadata: &fwkdl.EndpointMetadata{}, Metrics: &datalayer.Metrics{KVCacheUsagePercent: 0.5}},
				&types.PodMetrics{EndpointMetadata: &fwkdl.EndpointMetadata{}, Metrics: &datalayer.Metrics{KVCacheUsagePercent: 0.0}},
			},
			expectedScoresEndpoint: map[int]float64{
				0: 0.2, // Highest KV cache usage (0.8) gets lowest score (1-0.8=0.2)
				1: 0.5, // Medium KV cache usage (0.5) gets medium score (1-0.5=0.5)
				2: 1.0, // No KV cache usage (0.0) gets highest score (1-0=1.0)
			},
		},
		{
			name:           "Absolute: Same KV cache utilization",
			normalizeScore: false,
			endpoints: []types.Endpoint{
				&types.PodMetrics{EndpointMetadata: &fwkdl.EndpointMetadata{}, Metrics: &datalayer.Metrics{KVCacheUsagePercent: 0.6}},
				&types.PodMetrics{EndpointMetadata: &fwkdl.EndpointMetadata{}, Metrics: &datalayer.Metrics{KVCacheUsagePercent: 0.6}},
			},
			expectedScoresEndpoint: map[int]float64{
				0: 0.4, // Both get same score (1-0.6=0.4)
				1: 0.4,
			},
		},
		{
			name:           "Absolute: Zero KV cache utilization",
			normalizeScore: false,
			endpoints: []types.Endpoint{
				&types.PodMetrics{EndpointMetadata: &fwkdl.EndpointMetadata{}, Metrics: &datalayer.Metrics{KVCacheUsagePercent: 0.0}},
				&types.PodMetrics{EndpointMetadata: &fwkdl.EndpointMetadata{}, Metrics: &datalayer.Metrics{KVCacheUsagePercent: 0.0}},
			},
			expectedScoresEndpoint: map[int]float64{
				0: 1.0, // No KV cache usage gets highest score
				1: 1.0,
			},
		},
		{
			name: "Absolute: Full KV cache utilization",
			endpoints: []types.Endpoint{
				&types.PodMetrics{EndpointMetadata: &fwkdl.EndpointMetadata{}, Metrics: &datalayer.Metrics{KVCacheUsagePercent: 1.0}},
				&types.PodMetrics{EndpointMetadata: &fwkdl.EndpointMetadata{}, Metrics: &datalayer.Metrics{KVCacheUsagePercent: 0.5}},
			},
			expectedScoresEndpoint: map[int]float64{
				0: 0.0, // Full KV cache (1.0) gets lowest score (1-1=0)
				1: 0.5, // Half KV cache (0.5) gets medium score (1-0.5=0.5)
			},
		},
		{
			name:           "Normalized: Different KV cache utilization",
			normalizeScore: true,
			endpoints: []types.Endpoint{
				&types.PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, Metrics: &datalayer.Metrics{KVCacheUsagePercent: 0.1}},
				&types.PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, Metrics: &datalayer.Metrics{KVCacheUsagePercent: 0.5}},
				&types.PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}, Metrics: &datalayer.Metrics{KVCacheUsagePercent: 0.9}},
			},
			expectedScoresEndpoint: map[int]float64{
				0: 1.0, // (0.9 - 0.1) / (0.9 - 0.1) = 1.0
				1: 0.5, // (0.9 - 0.5) / (0.9 - 0.1) = 0.5
				2: 0.0, // (0.9 - 0.9) / (0.9 - 0.1) = 0.0
			},
		},
		{
			name:           "Normalized: All same utilization",
			normalizeScore: true,
			endpoints: []types.Endpoint{
				&types.PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, Metrics: &datalayer.Metrics{KVCacheUsagePercent: 0.5}},
				&types.PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, Metrics: &datalayer.Metrics{KVCacheUsagePercent: 0.5}},
			},
			expectedScoresEndpoint: map[int]float64{
				0: 1.0,
				1: 1.0,
			},
		},
		{
			name:           "Normalized: Utilization smaller than tolerance",
			normalizeScore: true,
			endpoints: []types.Endpoint{
				&types.PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, Metrics: &datalayer.Metrics{KVCacheUsagePercent: 0.5}},
				&types.PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, Metrics: &datalayer.Metrics{KVCacheUsagePercent: 0.500001}},
			},
			expectedScoresEndpoint: map[int]float64{
				0: 1.0,
				1: 1.0,
			},
		},
		{
			name:                   "Normalized: Empty endpoints",
			normalizeScore:         true,
			endpoints:              []types.Endpoint{},
			expectedScoresEndpoint: map[int]float64{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scorer := NewKVCacheUtilizationScorer(test.normalizeScore)
			scores := scorer.Score(context.Background(), types.NewCycleState(), &types.LLMRequest{}, test.endpoints)

			for i, endpoint := range test.endpoints {
				expectedScore := test.expectedScoresEndpoint[i]
				assert.InDelta(t, expectedScore, scores[endpoint], 0.0001, "Endpoint %d should have score %f", i, expectedScore)
			}
		})
	}
}

func TestKvCacheUtilizationScorerFactory(t *testing.T) {
	tests := []struct {
		name     string
		args     json.RawMessage
		wantNorm bool
		wantErr  bool
	}{
		{
			name:     "Empty args (default)",
			args:     nil,
			wantNorm: false,
			wantErr:  false,
		},
		{
			name:     "Explicit normalize false",
			args:     json.RawMessage(`{"normalizeScore": false}`),
			wantNorm: false,
			wantErr:  false,
		},
		{
			name:     "Explicit normalize true",
			args:     json.RawMessage(`{"normalizeScore": true}`),
			wantNorm: true,
			wantErr:  false,
		},
		{
			name:    "Invalid JSON",
			args:    json.RawMessage(`{invalid}`),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := KvCacheUtilizationScorerFactory(KvCacheUtilizationScorerType, tt.args, nil)
			if (err != nil) != tt.wantErr {
				t.Errorf("KvCacheUtilizationScorerFactory() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				scorer := p.(*KVCacheUtilizationScorer)
				if scorer.normalizeScore != tt.wantNorm {
					t.Errorf("expected normalizeScore %v, got %v", tt.wantNorm, scorer.normalizeScore)
				}
				if scorer.TypedName().Name != KvCacheUtilizationScorerType {
					t.Errorf("expected name 'kv-cache-utilization-scorer', got %s", scorer.TypedName().Name)
				}
			}
		})
	}
}
