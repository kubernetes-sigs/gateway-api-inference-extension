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

package approximateprefix

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
)

const (
	ApproxPrefixCachePluginType = "approx-prefix-cache-producer"
)

var (
	_ requestcontrol.PrepareDataPlugin = &prepareData{}
	_ requestcontrol.PreRequest        = &prepareData{}
)

// prepareData is a plugin that prepares data consumed by approx prefix cache aware scheduling.
type prepareData struct {
	typedName   plugin.TypedName
	config      config
	indexerInst indexerInterface
	pluginState *plugin.PluginState
	wg          sync.WaitGroup // Used for waiting on async cache updates in tests.
}

// TypedName returns the type and name of the plugin.
func (p *prepareData) TypedName() plugin.TypedName {
	return p.typedName
}

// Consumes returns the data consumed by the plugin.
func (p *prepareData) Consumes() map[string]any {
	return map[string]any{}
}

// Produces returns the data produced by the plugin.
func (p *prepareData) Produces() map[string]any {
	return map[string]any{attrprefix.PrefixCacheMatchInfoKey: attrprefix.PrefixCacheMatchInfo{}}
}

// newPrepareData returns a new PrepareData plugin.
func newPrepareData(ctx context.Context, config config, handle plugin.Handle) (*prepareData, error) {
	log.FromContext(ctx).V(logutil.DEFAULT).Info("Prefix PrepareData initialized", "config", config)

	//nolint:staticcheck // BlockSize is deprecated, but we check it here to provide a migration path for users.
	if config.BlockSize > 0 && config.BlockSizeTokens <= 0 {
		return nil, fmt.Errorf("invalid configuration: BlockSize (%d) is deprecated; please use BlockSizeTokens instead to define the cache block size in tokens", config.BlockSize)
	}

	if !config.AutoTune && config.BlockSizeTokens <= 0 {
		return nil, fmt.Errorf("invalid configuration: BlockSizeTokens must be > 0 when AutoTune is disabled (current value: %d)", config.BlockSizeTokens)
	}
	indexer := newIndexer(ctx, config.LRUCapacityPerServer)

	p := &prepareData{
		typedName: plugin.TypedName{
			Type: ApproxPrefixCachePluginType,
			Name: ApproxPrefixCachePluginType,
		},
		config:      config,
		indexerInst: indexer,
		pluginState: plugin.NewPluginState(ctx),
	}

	if handle != nil {
		go p.CleanUpInactivePods(ctx, handle)
	}

	return p, nil
}

// CleanUpInactivePods starts a goroutine that periodically removes inactive pods from the indexer.
func (p *prepareData) CleanUpInactivePods(ctx context.Context, handle plugin.Handle) {
	ticker := time.NewTicker(podActiveCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			podNames := handle.PodList()
			activePods := make(map[ServerID]struct{}, len(podNames))
			for _, nsn := range podNames {
				activePods[ServerID(nsn)] = struct{}{}
			}

			for _, pod := range p.indexerInst.Pods() {
				if _, ok := activePods[pod]; !ok {
					p.indexerInst.RemovePod(pod)
					log.FromContext(ctx).V(logutil.VERBOSE).Info("Removed pod not in active set", "pod", pod)
				}
			}
		}
	}
}

// indexer returns the shared indexer.
func (p *prepareData) indexer() indexerInterface {
	return p.indexerInst
}

// PluginState returns the shared plugin state.
func (p *prepareData) PluginState() *plugin.PluginState {
	return p.pluginState
}

// PrepareRequestData is called by the director before scheduling requests.
func (p *prepareData) PrepareRequestData(ctx context.Context, request *framework.LLMRequest, pods []framework.Endpoint) error {
	blockSize := p.GetBlockSize(pods)
	hashes := hashPrompt(ctx, request, blockSize, p.config.MaxPrefixBlocksToMatch)
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

	// Store the state in shared plugin state for later use in PreRequest.
	// NOTE: We use the prefix plugin's type name as part of the key so that the scorer can read it.
	p.pluginState.Write(request.RequestId, plugin.StateKey(ApproxPrefixCachePluginType), state)

	return nil
}

// PreRequest records in the shared indexer the result of the scheduling selection.
// It updates the indexer with the prefix hashes for the selected endpoint(s).
func (p *prepareData) PreRequest(ctx context.Context, request *framework.LLMRequest, schedulingResult *framework.SchedulingResult) {
	// Delete the state to avoid memory leak.
	defer p.pluginState.Delete(request.RequestId)
	primaryProfileResult := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if len(primaryProfileResult.TargetEndpoints) == 0 {
		return
	}

	targetEndpoint := primaryProfileResult.TargetEndpoints[0]
	servers := []server{p.makeserver(targetEndpoint)}

	// Also record for prefill node if present in P/D disaggregated mode.
	if pr, exists := schedulingResult.ProfileResults[experimentalDefaultPrefillProfile]; exists && len(pr.TargetEndpoints) > 0 {
		servers = append(servers, p.makeserver(pr.TargetEndpoints[0]))
	}

	// Read state saved during PrepareRequestData.
	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.pluginState, request.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	if err != nil {
		log.FromContext(ctx).Error(err, "failed to read prefix plugin state", "requestID", request.RequestId)
		return
	}

	// Update indexer asynchronously to avoid blocking the request path.
	p.wg.Go(func() {
		for _, s := range servers {
			p.indexerInst.Add(state.PrefixHashes, s)
		}
	})

	// Record metrics.
	total := len(state.PrefixHashes)
	matchLen := state.PrefixCacheServers[ServerID(targetEndpoint.GetMetadata().NamespacedName)]
	blockSize := p.GetBlockSize(primaryProfileResult.TargetEndpoints)
	avgChars := averageCharactersPerToken
	metrics.RecordPrefixCacheMatch(matchLen*blockSize*avgChars, total*blockSize*avgChars)
}

func (p *prepareData) makeserver(targetEndpoint framework.Endpoint) server {
	gpuBlocks := defaultLRUCapacityPerServer
	if p.config.AutoTune && targetEndpoint.GetMetrics().CacheNumBlocks > 0 {
		gpuBlocks = targetEndpoint.GetMetrics().CacheNumBlocks
	}
	return server{
		ServerID:       ServerID(targetEndpoint.GetMetadata().NamespacedName),
		NumOfGPUBlocks: gpuBlocks,
	}
}

// matchLongestPrefix returns a map of servers and length of prefix that each server caches, prefix length is defined in blocks.
func (p *prepareData) matchLongestPrefix(ctx context.Context, hashes []blockHash) map[ServerID]int {
	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)
	res := make(map[ServerID]int)

	// Use a greedy strategy to search from the longest prefix.
	for _, hash := range hashes {
		cachedServers := p.indexerInst.Get(hash)
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
func (p *prepareData) GetBlockSize(endpoints []framework.Endpoint) int {
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
	parameters := defaultConfig
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to unmarshal prefix cache parameters: %w", err)
		}
	}

	// pluginState will be initialized by newPrepareData as we pass nil here.
	p, err := newPrepareData(handle.Context(), parameters, handle)
	if err != nil {
		return nil, err
	}

	return p, nil
}
