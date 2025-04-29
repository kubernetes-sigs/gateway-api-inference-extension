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
	"encoding/binary"
	"fmt"

	"github.com/cespare/xxhash/v2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	// Attempt to return DefaultNumServersToMatch servers with their longest prefix match length.
	// Why not just return the server with longest prefix match?
	// It may not be the optimal choice, e.g., it may have a high queue depth.
	// We optimistically search more than one to give more candidates for the scheduler to choose.
	DefaultNumServersToMatch = 2
	// vLLM default token block size is 16, and a good guess of average characters per token is 4.
	DefaultCacheBlockSize  = 64
	DefaultMaxPrefixBlocks = 128
	// Assume each request reaches DefaultMaxPrefixBlocks = 128, and each BlockHash is cached onto 2
	// servers due to load balancing, then it requires 256 entries per request.
	// According to the estimates in indexer.estimateEntrySize(), the size of each entry is 348 bytes.
	// So each request will cost 89,088 bytes ~ 90KB.
	// Therefore, to cache 50k requests, we need 50K * 90KB = 4.5GB. Assuming 500 requests per
	// second, a 4.5 GB cache can hold at least last 100 seconds of requests.
	// Note in practice, the size of each entry will be much smaller (shorter NamespacedNames,
	// shorter prompt). And due to the prefix cache hit, the number of unique cache entries will be
	// much smaller per request. Therefore the actual cache size will be much smaller.
	// TODO: Add some guidance for choosing the right size.
	DefaultLRUIndexerCapacity = 50000
)

type Config struct {
	// The input prompt is broken into sizes of HashBlockSize to calculate block hashes . Requests
	// with length shorter than the block size will be ignored.
	HashBlockSize int
	// MaxPrefixBlocksToMatch is the maximum number of prefix blocks to match. Input beyond this limit will
	// be ignored.
	MaxPrefixBlocksToMatch int
	// Max (approximate) size of the LRU indexer in number of entries.
	LRUIndexerCapacity int
}

var DefaultConfig = Config{
	HashBlockSize:          DefaultCacheBlockSize,
	MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
	LRUIndexerCapacity:     DefaultLRUIndexerCapacity,
}

type plugin struct {
	Config
	indexer Indexer
}

type Indexer interface {
	Get(hash types.BlockHash) map[types.ServerID]bool
	Add(hashes []types.BlockHash, server types.ServerID)
}

func New(config Config) *plugin {
	m := &plugin{
		Config:  config,
		indexer: newIndexer(config.LRUIndexerCapacity),
	}
	return m
}

func (m *plugin) Name() string {
	return "prefixCache"
}

func (m *plugin) PreSchedule(ctx *types.SchedulingContext) {
	ctx.PrefixHashes = hashPrompt(ctx, m.HashBlockSize, m.MaxPrefixBlocksToMatch)
	ctx.PrefixCacheServers = m.matchLongestPrefix(ctx, DefaultNumServersToMatch)
	ctx.Logger.V(logutil.DEBUG).Info(fmt.Sprintf("PreSchedule, cached servers: %+v", ctx.PrefixCacheServers), "hashes", ctx.PrefixHashes)
}

// If a request was routed to a server, record it in the cache:
func (m *plugin) PostSchedule(ctx *types.SchedulingContext, res *types.Result) {
	targetPod := res.TargetPod.GetPod()
	m.indexer.Add(ctx.PrefixHashes, types.ServerID(targetPod.NamespacedName))
	total := len(ctx.PrefixHashes)
	matchLen := ctx.PrefixCacheServers[types.ServerID(targetPod.NamespacedName)]
	metrics.RecordPrefixCacheMatch(matchLen*m.HashBlockSize, total*m.HashBlockSize)
}

func (m *plugin) Score(ctx *types.SchedulingContext, pods []types.Pod) map[types.Pod]float64 {
	total := len(ctx.PrefixHashes)
	podScoreFunc := func(ctx *types.SchedulingContext, pod types.Pod) float64 {
		if total == 0 {
			return 0
		}
		matchLen := ctx.PrefixCacheServers[types.ServerID(pod.GetPod().NamespacedName)]
		return float64(matchLen) / float64(total)
	}

	scores := make(map[types.Pod]float64, len(pods))
	for _, pod := range pods {
		scores[pod] = podScoreFunc(ctx, pod)
	}
	return scores
}

// matchLongestPrefix returns a map of servers and length of prefix that each server caches.
func (m *plugin) matchLongestPrefix(ctx *types.SchedulingContext, numServers int) map[types.ServerID]int {
	if numServers > len(ctx.PodsSnapshot) {
		numServers = len(ctx.PodsSnapshot)
	}
	res := make(map[types.ServerID]int)
	// Use a greedy strategy to search from the longest prefix.
	// NOTE: It's possible to further optimize this with a binary search.
	for i := len(ctx.PrefixHashes) - 1; i >= 0 && len(res) < numServers; i-- {
		hash := ctx.PrefixHashes[i]
		cachedServers := m.indexer.Get(hash)
		if len(cachedServers) > 0 {
			ctx.Logger.V(logutil.VERBOSE).Info("Found cached servers", "cachedServers", cachedServers, "total # blocks", len(ctx.PrefixHashes), "longest prefix", i)
			for server := range cachedServers {
				// Update servers with their longest prefix match.
				// If we already found this server with longer prefix match, don't update it.
				if _, ok := res[server]; !ok {
					res[server] = i + 1
				}
			}
		}
	}
	return res
}

// hashPrompt divides the prompt into blocks and calculate the prefix cache for each block.
// hash(0) is the hash of the model name, since different models generally don't share prefix cache.
// For block i, hash(i) = hash(block i content, hash(i-1)).
func hashPrompt(ctx *types.SchedulingContext, cacheBlockSize int, maxPrefixBlocks int) []types.BlockHash {
	prompt := []byte(ctx.Req.Prompt)
	if len(prompt) < cacheBlockSize {
		ctx.Logger.V(logutil.DEBUG).Info("Request body too small for prefix cache", "size", len(prompt), "block size", cacheBlockSize)
		return nil
	}
	if len(prompt) > cacheBlockSize*maxPrefixBlocks {
		ctx.Logger.V(logutil.DEBUG).Info("Truncating input", "size", len(prompt), "max prefix blocks", maxPrefixBlocks, "block size", cacheBlockSize)
		prompt = prompt[:maxPrefixBlocks*cacheBlockSize]
	}
	// Split the body into blocks of size cacheBlockSize. The +1 is to account for the model.
	// If the last block is smaller than cacheBlockSize, it will be ignored.
	res := make([]types.BlockHash, 0, 1+len(prompt)/cacheBlockSize)
	// Add the model to the first block hash so that different models have different hashes even with the same body.
	res = append(res, types.BlockHash(xxhash.Sum64String(ctx.Req.ResolvedTargetModel)))
	for i := 0; i+cacheBlockSize <= len(prompt); i += cacheBlockSize {
		block := prompt[i : i+cacheBlockSize]
		prevBlockHash := res[len(res)-1]
		toHash := append(block, toBytes(prevBlockHash)...)
		res = append(res, types.BlockHash(xxhash.Sum64(toHash)))
	}
	return res
}

func toBytes(i types.BlockHash) []byte {
	bytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(bytes, uint64(i))
	return bytes
}
