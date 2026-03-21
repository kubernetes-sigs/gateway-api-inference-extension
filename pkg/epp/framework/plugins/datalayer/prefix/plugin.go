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

package prefix

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

var (
	_ requestcontrol.PrepareDataPlugin = &PrepareData{}
)

// PrepareData is a plugin that prepares data consumed by approx prefix cache aware scheduling.
type PrepareData struct {
	typedName plugin.TypedName
	config    Config
	indexer   Indexer
}

// TypedName returns the type and name of the plugin.
func (p *PrepareData) TypedName() plugin.TypedName {
	return p.typedName
}

// Consumes returns the data consumed by the plugin.
func (p *PrepareData) Consumes() map[string]any {
	return map[string]any{}
}

// Produces returns the data produced by the plugin.
func (p *PrepareData) Produces() map[string]any {
	return map[string]any{attrprefix.PrefixCacheMatchInfoKey: attrprefix.PrefixCacheMatchInfo{}}
}

// NewPrepareData initializes a new PrepareData and returns its pointer.
func NewPrepareData(ctx context.Context, config Config, indexer Indexer) (*PrepareData, error) {
	log.FromContext(ctx).V(logutil.DEFAULT).Info("Prefix PrepareData initialized", "config", config)
	return &PrepareData{
		typedName: plugin.TypedName{Type: attrprefix.PrefixCachePluginType, Name: ApproxPrefixCachePlugin},
		config:    config,
		indexer:   indexer,
	}, nil
}

// Indexer returns the shared indexer.
func (p *PrepareData) Indexer() Indexer {
	return p.indexer
}

// PrepareRequestData is called by the director before scheduling requests.
func (p *PrepareData) PrepareRequestData(ctx context.Context, request *framework.LLMRequest, pods []framework.Endpoint) error {
	blockSize := p.GetBlockSize(pods)
	hashes := HashPrompt(ctx, request, blockSize, p.config.MaxPrefixBlocksToMatch)
	total := len(hashes)
	prefixCacheServers := p.matchLongestPrefix(ctx, hashes)

	for _, pod := range pods {
		matchLen := prefixCacheServers[ServerID(pod.GetMetadata().NamespacedName)]
		pod.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(matchLen, total, blockSize))
	}

	state := &SchedulingContextState{
		PrefixHashes:       hashes,
		PrefixCacheServers: prefixCacheServers,
	}

	// Store the state in cycle state for later use in PreRequest.
	// NOTE: We use the prefix plugin's type name as part of the key so that the scorer can read it.
	plugin.NewPluginState(ctx).Write(request.RequestId, plugin.StateKey(attrprefix.PrefixCachePluginType), state)

	return nil
}

// matchLongestPrefix returns a map of servers and length of prefix that each server caches, prefix length is defined in blocks.
func (p *PrepareData) matchLongestPrefix(ctx context.Context, hashes []BlockHash) map[ServerID]int {
	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)
	res := make(map[ServerID]int)

	// Use a greedy strategy to search from the longest prefix.
	for _, hash := range hashes {
		cachedServers := p.indexer.Get(hash)
		if len(cachedServers) == 0 {
			break
		}
		loggerTrace.Info("Found cached servers", "cachedServers", cachedServers, "total # blocks", len(hashes))
		for server := range cachedServers {
			res[server]++
		}
	}
	return res
}

// GetBlockSize returns the block size in tokens, potentially auto-tuned from endpoint metrics.
func (p *PrepareData) GetBlockSize(endpoints []framework.Endpoint) int {
	if !p.config.AutoTune || len(endpoints) == 0 {
		return p.config.BlockSizeTokens
	}

	if endpoint := endpoints[0]; endpoint.GetMetrics() != nil {
		cacheBlockSize := endpoint.GetMetrics().CacheBlockSize
		if cacheBlockSize > 0 {
			return cacheBlockSize
		}
	}
	return p.config.BlockSizeTokens
}

// ApproxPrefixCacheFactory is the factory function for the prefix cache prepare data plugin.
func ApproxPrefixCacheFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := DefaultConfig
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to unmarshal prefix cache parameters: %w", err)
		}
	}

	// Share the indexer with the prefix scorer plugin.
	indexer := NewIndexer(handle.Context(), parameters.LRUCapacityPerServer)
	p, err := NewPrepareData(handle.Context(), parameters, indexer)
	if err != nil {
		return nil, err
	}

	return p, nil
}
