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
	DefaultKVCacheScorerWeight = 1
	KvCacheScorerType          = "kv-cache"
)

// compile-time type assertion
var _ framework.Scorer = &KVCacheScorer{}

// KvCacheScorerFactory defines the factory function for KVCacheScorer.
func KvCacheScorerFactory(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	return NewKVCacheScorer(), nil
}

// NewKVCacheScorer initializes a new KVCacheScorer and returns its pointer.
func NewKVCacheScorer() *KVCacheScorer {
	return &KVCacheScorer{}
}

// KVCacheScorer scores list of candidate pods based on KV cache utilization.
type KVCacheScorer struct{}

// Type returns the type of the scorer.
func (s *KVCacheScorer) Type() string {
	return KvCacheScorerType
}

// Score returns the scoring result for the given list of pods based on context.
func (s *KVCacheScorer) Score(_ context.Context, _ *plugins.CycleState, _ *types.LLMRequest, pods []types.Pod) map[types.Pod]float64 {
	scores := make(map[types.Pod]float64, len(pods))
	for _, pod := range pods {
		scores[pod] = 1 - pod.GetMetrics().KVCacheUsagePercent
	}
	return scores
}
