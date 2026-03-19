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

package preparedata

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

var (
	_ requestcontrol.PrepareDataPlugin = &ApproxPrefixCache{}
)

// ApproxPrefixCache is a plugin that prepares data consumed by approx prefix cache aware scheduling.
type ApproxPrefixCache struct {
	typedName    plugin.TypedName
	config       Config
	indexer      Indexer
	pluginState  plugin.PluginState
	wg           sync.WaitGroup
	prefixPlugin plugin.Plugin
}

// TypedName returns the type and name of the plugin.
func (p *ApproxPrefixCache) TypedName() plugin.TypedName {
	return plugin.TypedName{
		// Note: PrefixCachePluginType is used for backward compatibility since the parameters are passed to the prefix cache scorer plugin.
		Type: attrprefix.PrefixCachePluginType,
		Name: ApproxPrefixCachePlugin,
	}
}

// Consumes returns the data consumed by the plugin.
func (p *ApproxPrefixCache) Consumes() map[string]any {
	return map[string]any{}
}

// Produces returns the data produced by the plugin.
func (p *ApproxPrefixCache) Produces() map[string]any {
	return map[string]any{attrprefix.PrefixCacheMatchInfoKey: attrprefix.PrefixCacheMatchInfo{}}
}

// ApproxPrefixCacheFactory is the factory function for the ApproxPrefixCache plugin.
func ApproxPrefixCacheFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := DefaultConfig
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to unmarshal ApproxPrefixCache parameters: %w", err)
		}
	}

	// Share the indexer with the prefix scorer plugin.
	prefixScorer, err := getOrCreatePrefixScorer(handle, parameters)
	if err != nil {
		return nil, err
	}

	p, err := New(handle.Context(), parameters, prefixScorer.Indexer(), prefixScorer)
	if err != nil {
		return nil, err
	}

	return p, nil
}

func getOrCreatePrefixScorer(handle plugin.Handle, config Config) (PrefixScorer, error) {
	prefixScorer, err := plugin.PluginByType[PrefixScorer](handle, attrprefix.PrefixCachePluginType)
	if err == nil {
		if prefixScorer.Indexer() == nil {
			// Found but indexer is not initialized (e.g. created by PrefixCachePluginFactory)
			indexer := NewIndexer(handle.Context(), config.LRUCapacityPerServer)
			prefixScorer.SetIndexer(indexer)
		}
		return prefixScorer, nil
	}

	// Not found, create a new one.
	if ScorerFactory == nil {
		return nil, errors.New("ScorerFactory not initialized, make sure prefix package is imported")
	}

	indexer := NewIndexer(handle.Context(), config.LRUCapacityPerServer)
	prefixScorer, err = ScorerFactory(handle.Context(), config, indexer)
	if err != nil {
		return nil, fmt.Errorf("failed to create prefix scorer from factory: %w", err)
	}

	handle.AddPlugin(attrprefix.PrefixCachePluginType, prefixScorer)
	return prefixScorer, nil
}

// PrepareRequestData is called by the director before scheduling requests.
func (p *ApproxPrefixCache) PrepareRequestData(ctx context.Context, request *scheduling.LLMRequest, pods []scheduling.Endpoint) error {
	blockSize := GetBlockSize(pods, p.config)
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

	// Store the state in plugin state for later use in PreRequest.
	// NOTE: We use the prefix plugin's typed name so that it can read the state in its PreRequest.
	p.pluginState.Write(request.RequestId, plugin.StateKey(p.prefixPlugin.TypedName().String()), state)

	return nil
}

// matchLongestPrefix returns a map of servers and length of prefix that each server caches, prefix length is defined in blocks.
func (p *ApproxPrefixCache) matchLongestPrefix(ctx context.Context, hashes []BlockHash) map[ServerID]int {
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

// New initializes a new ApproxPrefixCache and returns its pointer.
func New(ctx context.Context, config Config, indexer Indexer, prefixPlugin plugin.Plugin) (*ApproxPrefixCache, error) {
	if err := validateConfig(config); err != nil {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(err, "invalid prefix plugin configuration")
		return nil, err
	}

	applyConfigDefaults(&config)

	log.FromContext(ctx).V(logutil.DEFAULT).Info("ApproxPrefixCache initialized", "config", config)
	return &ApproxPrefixCache{
		typedName:    plugin.TypedName{Type: attrprefix.PrefixCachePluginType, Name: ApproxPrefixCachePlugin},
		config:       config,
		pluginState:  *plugin.NewPluginState(ctx),
		indexer:      indexer,
		prefixPlugin: prefixPlugin,
	}, nil
}

func validateConfig(config Config) error {
	if config.BlockSize > 0 && config.BlockSizeTokens <= 0 {
		return errors.New("BlockSize is deprecated, use BlockSizeTokens instead, the value should be defined in tokens")
	}
	return nil
}

func applyConfigDefaults(config *Config) {
	if config.LRUCapacityPerServer <= 0 {
		config.LRUCapacityPerServer = DefaultLRUCapacityPerServer
	}
	if config.BlockSizeTokens <= 0 {
		config.BlockSizeTokens = DefaultBlockSizeTokens
	}
	if config.MaxPrefixBlocksToMatch <= 0 {
		config.MaxPrefixBlocksToMatch = DefaultMaxPrefixBlocks
	}
}

// GetBlockSize returns the block size in tokens.
func GetBlockSize(endpoints []scheduling.Endpoint, config Config) int {
	if !config.AutoTune || len(endpoints) == 0 {
		return config.BlockSizeTokens
	}

	// All endpoints from the same pool should have identical configs.
	if endpoint := endpoints[0]; endpoint.GetMetrics() != nil {
		cacheBlockSize := endpoint.GetMetrics().CacheBlockSize
		if cacheBlockSize > 0 {
			return cacheBlockSize
		}
	}
	return config.BlockSizeTokens
}
