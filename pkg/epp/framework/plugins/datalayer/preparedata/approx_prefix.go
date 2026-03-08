/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may not use this file except in compliance with the License.
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

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	prefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/scorer/prefix"
)

var _ requestcontrol.PrepareDataPlugin = &ApproxPrefixCache{}

const (
	// PrepareDataPluginType is the type for plugins that prepare data for scheduling.
	PrepareDataPluginType   = "prepare-data-plugin"
	ApproxPrefixCachePlugin = "approx-prefix-cache"
)

// ApproxPrefixCache is a plugin that prepares data consumed by approx prefix cache aware scheduling.
type ApproxPrefixCache struct {
	typedName plugin.TypedName
	config    prefix.Config
	indexer   prefix.Indexer
}

// New returns a new ApproxPrefixCache plugin.
func New() *ApproxPrefixCache {
	return &ApproxPrefixCache{}
}

// TypedName returns the type and name of the plugin.
func (p *ApproxPrefixCache) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: PrepareDataPluginType,
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

// PrepareRequestData is called by the director before scheduling requests.
func (p *ApproxPrefixCache) PrepareRequestData(ctx context.Context, request *scheduling.LLMRequest, pods []scheduling.Endpoint) error {
	blockSize := prefix.GetBlockSize(pods, p.config)
	hashes := prefix.HashPrompt(ctx, request, blockSize, p.config.MaxPrefixBlocksToMatch)
	total := len(hashes)
	prefixCacheServers := p.matchLongestPrefix(ctx, hashes)
	for _, pod := range pods {
		matchLen := prefixCacheServers[prefix.ServerID(pod.GetMetadata().NamespacedName)]
		pod.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(matchLen, total, blockSize))
	}

	return nil
}

// matchLongestPrefix returns a map of servers and length of prefix that each server caches, prefix length is defined in blocks.
func (p *ApproxPrefixCache) matchLongestPrefix(ctx context.Context, hashes []prefix.BlockHash) map[prefix.ServerID]int {
	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)
	res := make(map[prefix.ServerID]int)
	// Use a greedy strategy to search from the longest prefix.
	// NOTE: It's possible to further optimize this with a binary search.
	for i, hash := range hashes {
		cachedServers := p.indexer.Get(hash)
		if len(cachedServers) == 0 {
			break
		} else {
			loggerTrace.Info("Found cached servers", "cachedServers", cachedServers, "total # blocks", len(hashes), "longest prefix", i)
			for server := range cachedServers {
				// Update servers with their longest prefix match, prefix length is in blocks.
				res[server]++
			}
		}
	}
	return res
}
