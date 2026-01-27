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
	"math"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
)

const (
	KvCacheUtilizationScorerType = "kv-cache-utilization-scorer"
	// normalizedKVCacheUtilizationTolerance is the epsilon used for floating point comparisons
	// Only NormalizeScore is enabled, checking if max and min KV cache utilization(0.0 to 1.0) are effectively equal.
	normalizedKVCacheUtilizationTolerance = 1e-5
)

// Config defines the configuration arguments for KVCacheUtilizationScorer.
type Config struct {
	// NormalizeScore indicates whether to normalize scores against other candidates.
	// If true, the score will be calculated as (max - current) / (max - min).
	// If false, the score will be calculated as (1 - current).
	NormalizeScore bool `json:"normalizeScore,omitempty"`
}

// compile-time type assertion
var _ framework.Scorer = &KVCacheUtilizationScorer{}

// KvCacheUtilizationScorerFactory defines the factory function for KVCacheUtilizationScorer.
func KvCacheUtilizationScorerFactory(name string, args json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config := &Config{NormalizeScore: false}
	if args != nil {
		if err := json.Unmarshal(args, config); err != nil {
			return nil, err
		}
	}
	return NewKVCacheUtilizationScorer(config.NormalizeScore).WithName(name), nil
}

// NewKVCacheUtilizationScorer initializes a new KVCacheUtilizationScorer and returns its pointer.
func NewKVCacheUtilizationScorer(normalizeScore bool) *KVCacheUtilizationScorer {
	return &KVCacheUtilizationScorer{
		typedName:      fwkplugin.TypedName{Type: KvCacheUtilizationScorerType, Name: KvCacheUtilizationScorerType},
		normalizeScore: normalizeScore,
	}
}

// KVCacheUtilizationScorer scores list of candidate endpoints based on KV cache utilization.
type KVCacheUtilizationScorer struct {
	typedName      fwkplugin.TypedName
	normalizeScore bool
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *KVCacheUtilizationScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *KVCacheUtilizationScorer) Category() framework.ScorerCategory {
	return framework.Distribution
}

// Consumes returns the list of data that is consumed by the plugin.
func (s *KVCacheUtilizationScorer) Consumes() map[string]any {
	return map[string]any{
		metrics.KVCacheUsagePercentKey: float64(0),
	}
}

// WithName sets the name of the scorer.
func (s *KVCacheUtilizationScorer) WithName(name string) *KVCacheUtilizationScorer {
	s.typedName.Name = name
	return s
}

// Score returns the scoring result for the given list of endpoints based on context.
func (s *KVCacheUtilizationScorer) Score(_ context.Context, _ *framework.CycleState, _ *framework.LLMRequest, endpoints []framework.Endpoint) map[framework.Endpoint]float64 {
	scores := make(map[framework.Endpoint]float64, len(endpoints))

	if s.normalizeScore {
		minKVCacheUsage := 1.0
		maxKVCacheUsage := 0.0

		for _, endpoint := range endpoints {
			kvCacheUsage := endpoint.GetMetrics().KVCacheUsagePercent
			if kvCacheUsage < minKVCacheUsage {
				minKVCacheUsage = kvCacheUsage
			}
			if kvCacheUsage > maxKVCacheUsage {
				maxKVCacheUsage = kvCacheUsage
			}
		}

		// endpointScoreFunc calculates the score based on the KV cache usage of each endpoint.
		// Higher KV cache usage gets a lower score.
		endpointScoreFunc := func(endpoint framework.Endpoint) float64 {
			if math.Abs(maxKVCacheUsage-minKVCacheUsage) < normalizedKVCacheUtilizationTolerance {
				// If all endpoints have roughly the same KV cache usage, return a neutral score
				return 1.0
			}
			return (maxKVCacheUsage - endpoint.GetMetrics().KVCacheUsagePercent) / (maxKVCacheUsage - minKVCacheUsage)
		}

		for _, endpoint := range endpoints {
			scores[endpoint] = endpointScoreFunc(endpoint)
		}
	} else {
		for _, endpoint := range endpoints {
			scores[endpoint] = 1 - endpoint.GetMetrics().KVCacheUsagePercent
		}
	}
	return scores
}
