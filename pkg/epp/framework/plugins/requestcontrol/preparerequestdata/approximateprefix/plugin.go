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
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

var (
	_ requestcontrol.PrepareDataPlugin = &PrepareData{}
	_ requestcontrol.PreRequest        = &PrepareData{}
)

// PrepareData is a plugin that prepares data consumed by approx prefix cache aware scheduling.
type PrepareData struct {
	typedName   plugin.TypedName
	config      Config
	indexer     Indexer
	pluginState *plugin.PluginState
	metrics     MetricsReporter
	wg          sync.WaitGroup // Used for waiting on async cache updates in tests.
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

// NewPrepareData returns a new PrepareData plugin.
func NewPrepareData(ctx context.Context, config Config, handle plugin.Handle, pluginState *plugin.PluginState, metrics MetricsReporter) (*PrepareData, error) {
	log.FromContext(ctx).V(logutil.DEFAULT).Info("Prefix PrepareData initialized", "config", config)

	//nolint:staticcheck // BlockSize is deprecated, but we check it here to provide a migration path for users.
	if config.BlockSize > 0 && config.BlockSizeTokens <= 0 {
		return nil, fmt.Errorf("invalid configuration: BlockSize (%d) is deprecated; please use BlockSizeTokens instead to define the cache block size in tokens", config.BlockSize)
	}

	if !config.AutoTune && config.BlockSizeTokens <= 0 {
		return nil, fmt.Errorf("invalid configuration: BlockSizeTokens must be > 0 when AutoTune is disabled (current value: %d)", config.BlockSizeTokens)
	}

	indexer := NewIndexer(ctx, config.LRUCapacityPerServer, metrics)

	// If pluginState is nil, we initialize it here. This ensures that the state object
	// (and its background cleanup goroutine) is created exactly once during the plugin's construction.
	if pluginState == nil {
		pluginState = plugin.NewPluginState(ctx)
	}

	p := &PrepareData{
		typedName: plugin.TypedName{
			Type: attrprefix.PrefixCachePluginType,
			Name: ApproxPrefixCachePlugin,
		},
		config:      config,
		indexer:     indexer,
		pluginState: pluginState,
		metrics:     metrics,
	}

	if handle != nil {
		go p.CleanUpInactivePods(ctx, handle)
	}

	return p, nil
}

// CleanUpInactivePods starts a goroutine that periodically removes inactive pods from the indexer.
func (p *PrepareData) CleanUpInactivePods(ctx context.Context, handle plugin.Handle) {
	ticker := time.NewTicker(PodActiveCheckInterval)
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

			for _, pod := range p.indexer.Pods() {
				if _, ok := activePods[pod]; !ok {
					p.indexer.RemovePod(pod)
					log.FromContext(ctx).V(logutil.VERBOSE).Info("Removed pod not in active set", "pod", pod)
				}
			}
		}
	}
}

// Indexer returns the shared indexer.
func (p *PrepareData) Indexer() Indexer {
	return p.indexer
}

// PluginState returns the shared plugin state.
func (p *PrepareData) PluginState() *plugin.PluginState {
	return p.pluginState
}

// SetMetricsReporter sets the metrics reporter for the plugin.
func (p *PrepareData) SetMetricsReporter(reporter MetricsReporter) {
	p.metrics = reporter
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

	// Store the state in shared plugin state for later use in PreRequest.
	// NOTE: We use the prefix plugin's type name as part of the key so that the scorer can read it.
	p.pluginState.Write(request.RequestId, plugin.StateKey(attrprefix.PrefixCachePluginType), state)

	return nil
}

// PreRequest records in the shared indexer the result of the scheduling selection.
// It updates the indexer with the prefix hashes for the selected endpoint(s).
func (p *PrepareData) PreRequest(ctx context.Context, request *framework.LLMRequest, schedulingResult *framework.SchedulingResult) {
	primaryProfileResult := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if len(primaryProfileResult.TargetEndpoints) == 0 {
		return
	}

	targetEndpoint := primaryProfileResult.TargetEndpoints[0]
	servers := []Server{p.makeServer(targetEndpoint)}

	// Also record for prefill node if present in P/D disaggregated mode.
	if pr, exists := schedulingResult.ProfileResults[Experimental_DefaultPrefillProfile]; exists && len(pr.TargetEndpoints) > 0 {
		servers = append(servers, p.makeServer(pr.TargetEndpoints[0]))
	}

	// Read state saved during PrepareRequestData.
	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.pluginState, request.RequestId, plugin.StateKey(attrprefix.PrefixCachePluginType))
	p.pluginState.Delete(request.RequestId)
	if err != nil {
		log.FromContext(ctx).Error(err, "failed to read prefix plugin state", "requestID", request.RequestId)
		return
	}

	// Update indexer asynchronously to avoid blocking the request path.
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		for _, s := range servers {
			p.indexer.Add(state.PrefixHashes, s)
		}
	}()

	// Record metrics.
	if p.metrics != nil {
		total := len(state.PrefixHashes)
		matchLen := state.PrefixCacheServers[ServerID(targetEndpoint.GetMetadata().NamespacedName)]
		blockSize := p.GetBlockSize(primaryProfileResult.TargetEndpoints)
		avgChars := AverageCharactersPerToken()
		p.metrics.RecordPrefixCacheMatch(matchLen*blockSize*avgChars, total*blockSize*avgChars)
	}
}

func (p *PrepareData) makeServer(targetEndpoint framework.Endpoint) Server {
	gpuBlocks := DefaultLRUCapacityPerServer
	if p.config.AutoTune && targetEndpoint.GetMetrics().CacheNumGPUBlocks > 0 {
		gpuBlocks = targetEndpoint.GetMetrics().CacheNumGPUBlocks
	}
	return Server{
		ServerID:       ServerID(targetEndpoint.GetMetadata().NamespacedName),
		NumOfGPUBlocks: gpuBlocks,
	}
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

	// pluginState will be initialized by NewPrepareData as we pass nil here.
	p, err := NewPrepareData(handle.Context(), parameters, handle, nil, nil)
	if err != nil {
		return nil, err
	}

	return p, nil
}
