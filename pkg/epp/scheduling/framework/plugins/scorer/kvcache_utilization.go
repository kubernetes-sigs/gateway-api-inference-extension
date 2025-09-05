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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	KvCacheUtilizationScorerType = "kv-cache-utilization-scorer"
)

// compile-time type assertion
var _ framework.Scorer = &KVCacheUtilizationScorer{}

// KvCacheUtilizationScorerFactory defines the factory function for KVCacheUtilizationScorer.
func KvCacheUtilizationScorerFactory(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	return NewKVCacheUtilizationScorer().WithName(name), nil
}

// NewKVCacheUtilizationScorer initializes a new KVCacheUtilizationScorer and returns its pointer.
func NewKVCacheUtilizationScorer() *KVCacheUtilizationScorer {
	return &KVCacheUtilizationScorer{
		typedName: plugins.TypedName{Type: KvCacheUtilizationScorerType, Name: KvCacheUtilizationScorerType},
	}
}

// KVCacheUtilizationScorer scores list of candidate pods based on KV cache utilization.
type KVCacheUtilizationScorer struct {
	typedName plugins.TypedName
}

func (s *KVCacheUtilizationScorer) Dependencies() []plugins.TypedName {
	return []plugins.TypedName{} // No dependencies
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *KVCacheUtilizationScorer) TypedName() plugins.TypedName {
	return s.typedName
}

// WithName sets the name of the scorer.
func (s *KVCacheUtilizationScorer) WithName(name string) *KVCacheUtilizationScorer {
	s.typedName.Name = name
	return s
}

// Score returns the scoring result for the given list of pods based on context.
func (s *KVCacheUtilizationScorer) Score(_ context.Context, _ *types.CycleState, req *types.LLMRequest, pods []types.Pod) map[types.Pod]float64 {
	scores := make(map[types.Pod]float64, len(pods))

	if req.PredictorBasedScheduling {
		// If PredictorBasedScheduling is true, we skip queue-based scoring.
		// This is to avoid interference with latency-based scoring.
		for _, pod := range pods {
			scores[pod] = 0.0 // Neutral score
		}
		return scores
	}

	for _, pod := range pods {
		scores[pod] = 1 - pod.GetMetrics().KVCacheUsagePercent
	}
	return scores
}
