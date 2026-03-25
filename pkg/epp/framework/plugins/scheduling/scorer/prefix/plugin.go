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

package prefix

import (
	"context"
	"encoding/json"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
)

const (
	// vLLM default token block size is 16 tokens
	DefaultBlockSizeTokens = 16
	// The maximum number of blocks to match. Two long requests with the same prefix up to this
	// limit will be indistinguishable.
	// This parameter provides a trade-off between cache size, prefix matching speed and matching
	// accuracy. Use a small value if most requests are short to reduce cache size and speed up the
	// matching process. Use a large value if most requests are long to increase the matching accuracy.
	DefaultMaxPrefixBlocks = 256
	// The indexer is an approximation to the actual prefix LRU cache state on the model servers per server (pod).
	// A small capacity ensures a high accuracy of cache hit on the model server, but it will
	// increase the chance of false negatives. A high capacity does the opposite.
	// To properly size this, consider the sum of the total number of cache entries on all model
	// servers. Consider the llama3 8B model on a H100 80GB GPUs. The size of the model weight is
	// about 16GB. The remaining HBM used for caching prefixes is 64GB. Each
	// token is about 128KB in size, so we can cache 500K tokens. Using the default block size of 16
	// in vLLM, we will have 250K / 16 = 31.25K blocks.
	DefaultLRUCapacityPerServer = 31250
	// In P/D disaggregation mode, the prefill and decode are usually represented as two different scheduling profiles to pick
	// the prefill and decode endpoints. This constant defines the prefill profile name to ensure that the index is updated
	// for the prefill endpoint and not only for the primary endpoint that will initially handle the request.
	// This is hardcoded for now until we land on a canonical approach for plugins to identify prefill and decode endpoints
	// (See https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/2080)
	Experimental_DefaultPrefillProfile = "prefill"

	PrefixCachePluginType = "prefix-cache-scorer"

	PodActiveCheckInterval = 2 * time.Minute
)

// Plugin implements the prefix cache aware scoring logic.
type Plugin struct {
	typedName plugin.TypedName
}

// compile-time type assertions
var (
	_ framework.Scorer = &Plugin{}
)

type metricsReporter struct{}

func (m *metricsReporter) RecordPrefixCacheSize(size int64) {
	metrics.RecordPrefixCacheSize(size)
}

func (m *metricsReporter) RecordPrefixCacheMatch(matchedTokens, totalTokens int) {
	metrics.RecordPrefixCacheMatch(matchedTokens, totalTokens)
}

// PrefixCachePluginFactory defines the factory function for the Prefix plugin.
func PrefixCachePluginFactory(name string, _ json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	p, err := New(handle.Context())
	if err != nil {
		return nil, err
	}
	if name != "" {
		p = p.WithName(name)
	}
	return p, nil
}

// New initializes a new prefix Plugin.
func New(_ context.Context) (*Plugin, error) {
	return &Plugin{
		typedName: plugin.TypedName{
			Type: attrprefix.PrefixCachePluginType,
			Name: attrprefix.PrefixCachePluginType,
		},
	}, nil
}

// TypedName returns the type and name of this plugin instance.
func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// Category returns the preference the scorer applies (Affinity).
func (p *Plugin) Category() framework.ScorerCategory {
	return framework.Affinity
}

// WithName sets the name of the plugin instance.
func (p *Plugin) WithName(name string) *Plugin {
	p.typedName.Name = name
	return p
}

// Produces returns the data produced by the plugin.
func (p *Plugin) Produces() map[string]any {
	return map[string]any{attrprefix.PrefixCacheMatchInfoKey: attrprefix.PrefixCacheMatchInfo{}}
}

// Consumes returns the data consumed by the plugin.
func (p *Plugin) Consumes() map[string]any {
	return map[string]any{attrprefix.PrefixCacheMatchInfoKey: attrprefix.PrefixCacheMatchInfo{}}
}

// Score returns the scoring result for the given list of pods based on prefix cache match info.
func (p *Plugin) Score(ctx context.Context, _ *framework.CycleState, _ *framework.LLMRequest, endpoints []framework.Endpoint) map[framework.Endpoint]float64 {
	scores := make(map[framework.Endpoint]float64, len(endpoints))
	logger := log.FromContext(ctx)

	for _, endpoint := range endpoints {
		info, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoKey)
		if !ok {
			logger.V(logutil.DEFAULT).Error(nil, "PrefixCacheMatchInfo not found for endpoint, assigning score 0", "endpoint", endpoint)
			scores[endpoint] = 0.0
			continue
		}

		if prefixMatchInfo, ok := info.(*attrprefix.PrefixCacheMatchInfo); ok {
			if prefixMatchInfo.TotalBlocks() == 0 {
				scores[endpoint] = 0.0
			} else {
				scores[endpoint] = float64(prefixMatchInfo.MatchBlocks()) / float64(prefixMatchInfo.TotalBlocks())
			}
		} else {
			logger.V(logutil.DEFAULT).Error(nil, "PrefixCacheMatchInfo has unexpected type, assigning score 0", "endpoint", endpoint)
			scores[endpoint] = 0.0
		}
	}
	return scores
}
