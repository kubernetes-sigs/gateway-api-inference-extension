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
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	DefaultScorerWeight = 1
	// Attempt to return DefaultNumServersToMatch servers with their longest prefix match length.
	// Why not just return the server with longest prefix match?
	// It may not be the optimal choice, e.g., it may have a high queue depth.
	// We optimistically search more than one to give more candidates for the scheduler to choose.
	DefaultNumServersToMatch = 2
	// vLLM default token block size is 16, and a good guess of average characters per token is 4.
	DefaultHashBlockSize = 64
	// The maximum number of blocks to match. Two long requests with the same prefix up to this
	// limit will be indistinguishable.
	// This parameter provides a trade-off between cache size, prefix matching speed and matching
	// accuracy. Use a small value if most requests are short to reduce cache size and speed up the
	// matching process. Use a large value if most requests are long to increase the matching accuracy.
	DefaultMaxPrefixBlocks = 128
	// The indexer is an approximation to the actual prefix cache state on the model servers.
	// A small capacity ensures a high accuracy of cache hit on the model server, but it will
	// increase the chance of false negatives. A high capacity does the opposite.
	// To properly size this, consider the sum of the total number of cache entries on all model
	// servers. Consider the llama3 8B model on 3 H100 80GB GPUs. The size of the model weight is
	// about 16GB. Assume 50% of the remaining HBM is used for caching prefixes, we have 32GB. Each
	// token is about 128KB in size, so we can cache 250K tokens. Using the default block size of 16
	// in vLLM, we will have 250K / 16 = 15.6K blocks. In total we have 15.6K * 3 = 46.8K blocks, or
	// roughly 50K.
	// How much memory space does it require to hold the 50K block hashes?
	// According to the estimates in indexer.estimateEntrySize(), the size of each entry is
	// approximately 348 bytes. So in total we have 50K * 348 = 17.4MB.
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

type Plugin struct {
	Config
	indexer Indexer
}

type Indexer interface {
	Get(hash BlockHash) map[ServerID]bool
	Add(hashes []BlockHash, server ServerID)
}

// BlockHash is a hash of the block of request body.
type BlockHash uint64

type ServerID k8stypes.NamespacedName

func (s ServerID) String() string {
	return k8stypes.NamespacedName(s).String()
}

var _ types.StateData = &schedulingContextState{}

// This is the state of this plugin to be used during a scheduling cycle.
type schedulingContextState struct {
	// PrefixHashes is a list of prefix hashes of the request prompt broken into blocks.
	PrefixHashes []BlockHash
	// A map of server to its longest prefix cache match length.
	PrefixCacheServers map[ServerID]int
}

func (s *schedulingContextState) Clone() types.StateData {
	prefixHashes := make([]BlockHash, len(s.PrefixHashes))
	copy(prefixHashes, s.PrefixHashes)
	prefixCacheServers := make(map[ServerID]int, len(s.PrefixCacheServers))
	for key, value := range s.PrefixCacheServers {
		prefixCacheServers[key] = value
	}

	return &schedulingContextState{
		PrefixHashes:       prefixHashes,
		PrefixCacheServers: prefixCacheServers,
	}
}

func New(config Config) *Plugin {
	m := &Plugin{
		Config:  config,
		indexer: newIndexer(config.LRUIndexerCapacity),
	}
	return m
}

func (m *Plugin) Name() string {
	return "prefix-cache"
}

func (m *Plugin) PreSchedule(ctx *types.SchedulingContext) {
	hashes := hashPrompt(ctx, m.HashBlockSize, m.MaxPrefixBlocksToMatch)
	state := &schedulingContextState{
		PrefixHashes:       hashes,
		PrefixCacheServers: m.matchLongestPrefix(ctx, hashes, DefaultNumServersToMatch),
	}

	ctx.CycleState.Write(types.StateKey(m.Name()), state)
	ctx.Logger.V(logutil.DEBUG).Info(fmt.Sprintf("PreSchedule, cached servers: %+v", state.PrefixCacheServers), "hashes", state.PrefixHashes)
}

// If a request was routed to a server, record it in the cache:
func (m *Plugin) PostSchedule(ctx *types.SchedulingContext, res *types.Result) {
	targetPod := res.TargetPod.GetPod()
	state, err := m.getPrefixState(ctx.CycleState)
	if err != nil {
		ctx.Logger.Error(err, "failed to read prefix plugin cycle state")
		return
	}
	m.indexer.Add(state.PrefixHashes, ServerID(targetPod.NamespacedName))
	total := len(state.PrefixHashes)
	matchLen := state.PrefixCacheServers[ServerID(targetPod.NamespacedName)]
	metrics.RecordPrefixCacheMatch(matchLen*m.HashBlockSize, total*m.HashBlockSize)
}

func (m *Plugin) Score(ctx *types.SchedulingContext, pods []types.Pod) map[types.Pod]float64 {
	scores := make(map[types.Pod]float64, len(pods))

	state, err := m.getPrefixState(ctx.CycleState)
	if err != nil {
		ctx.Logger.Error(err, "failed to read prefix plugin cycle state")
		return scores
	}

	total := len(state.PrefixHashes)
	podScoreFunc := func(pod types.Pod) float64 {
		if total == 0 {
			return 0
		}
		matchLen := state.PrefixCacheServers[ServerID(pod.GetPod().NamespacedName)]
		return float64(matchLen) / float64(total)
	}

	for _, pod := range pods {
		scores[pod] = podScoreFunc(pod)
	}
	return scores
}

// matchLongestPrefix returns a map of servers and length of prefix that each server caches.
func (m *Plugin) matchLongestPrefix(ctx *types.SchedulingContext, hashes []BlockHash, numServers int) map[ServerID]int {
	if numServers > len(ctx.PodsSnapshot) {
		numServers = len(ctx.PodsSnapshot)
	}
	res := make(map[ServerID]int)
	// Use a greedy strategy to search from the longest prefix.
	// NOTE: It's possible to further optimize this with a binary search.
	for i := len(hashes) - 1; i >= 0 && len(res) < numServers; i-- {
		hash := hashes[i]
		cachedServers := m.indexer.Get(hash)
		if len(cachedServers) > 0 {
			ctx.Logger.V(logutil.DEBUG).Info("Found cached servers", "cachedServers", cachedServers, "total # blocks", len(hashes), "longest prefix", i)
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

func (m *Plugin) getPrefixState(cycleState *types.CycleState) (*schedulingContextState, error) {
	prefixStateKey := types.StateKey(m.Name())
	state, err := cycleState.Read(prefixStateKey)
	if err != nil {
		return nil, fmt.Errorf("failed reading %q from CycleState: %w", prefixStateKey, err)
	}

	prefixSchedulingState, ok := state.(*schedulingContextState)
	if !ok {
		return nil, fmt.Errorf("invalid Prefix state, got type %T", state)
	}

	return prefixSchedulingState, nil
}

// hashPrompt divides the prompt into blocks and calculate the prefix cache for each block.
// hash(0) is the hash of the model name, since different models generally don't share prefix cache.
// For block i, hash(i) = hash(block i content, hash(i-1)).
func hashPrompt(ctx *types.SchedulingContext, cacheBlockSize int, maxPrefixBlocks int) []BlockHash {
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
	res := make([]BlockHash, 0, 1+len(prompt)/cacheBlockSize)
	// Add the model to the first block hash so that different models have different hashes even with the same body.
	res = append(res, BlockHash(xxhash.Sum64String(ctx.Req.TargetModel)))
	for i := 0; i+cacheBlockSize <= len(prompt); i += cacheBlockSize {
		block := prompt[i : i+cacheBlockSize]
		prevBlockHash := res[len(res)-1]
		block = append(block, toBytes(prevBlockHash)...)
		res = append(res, BlockHash(xxhash.Sum64(block)))
	}
	return res
}

func toBytes(i BlockHash) []byte {
	bytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(bytes, uint64(i))
	return bytes
}
