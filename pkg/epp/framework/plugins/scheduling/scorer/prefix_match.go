/*
Copyright 2026 The Kubernetes Authors.

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

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

const (
	PrefixMatchScorerPluginType = "prefix-match-scorer"
)

type PrefixMatchScorer struct {
	typedName plugin.TypedName
}

// compile-time type assertion
var (
	_ framework.Scorer = &PrefixMatchScorer{}
)

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (p *PrefixMatchScorer) Category() framework.ScorerCategory {
	return framework.Affinity
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *PrefixMatchScorer) TypedName() plugin.TypedName {
	return p.typedName
}

// Consumes returns the data consumed by the plugin.
func (p *PrefixMatchScorer) Consumes() map[string]any {
	return map[string]any{attrprefix.PrefixCacheMatchInfoKey: attrprefix.PrefixCacheMatchInfo{}}
}

// Produces returns the data produced by the plugin.
func (p *PrefixMatchScorer) Produces() map[string]any {
	return map[string]any{}
}

// Score returns the scoring result for the given list of pods based on context.
func (p *PrefixMatchScorer) Score(ctx context.Context, _ *framework.CycleState, request *framework.LLMRequest, endpoints []framework.Endpoint) map[framework.Endpoint]float64 {
	// calculate the scores of endpoints
	scores := make(map[framework.Endpoint]float64, len(endpoints))
	logger := log.FromContext(ctx)

	for _, endpoint := range endpoints {
		// PrefixCacheMatchInfo is expected to be set by the datalayer plugin that calculates the prefix cache match info for the request and endpoints.
		info, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoKey)
		if !ok {
			logger.V(logging.DEFAULT).Error(nil, "PrefixCacheMatchInfo not found for endpoint, assigning score 0", "endpoint", endpoint)
			scores[endpoint] = 0.0
			continue
		}
		if prefixMatchInfo, ok := info.(*attrprefix.PrefixCacheMatchInfo); ok {
			if prefixMatchInfo.TotalBlocks() == 0 {
				logger.V(logging.DEFAULT).Info("TotalBlocks is 0 for endpoint, assigning score 0", "endpoint", endpoint)
				scores[endpoint] = 0.0
			} else {
				scores[endpoint] = float64(prefixMatchInfo.MatchBlocks()) / float64(prefixMatchInfo.TotalBlocks())
			}
		} else {
			logger.V(logging.DEFAULT).Error(nil, "PrefixCacheMatchInfo has unexpected type for endpoint, assigning score 0", "endpoint", endpoint)
			scores[endpoint] = 0.0
		}
	}
	return scores
}
