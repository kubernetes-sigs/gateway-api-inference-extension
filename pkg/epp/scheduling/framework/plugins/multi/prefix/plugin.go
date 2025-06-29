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
	"encoding/binary"
	"encoding/json"
	"fmt"

	"github.com/cespare/xxhash/v2"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	DefaultScorerWeight = 1
	// vLLM default token block size is 16, and a good guess of average characters per token is 4.
	DefaultHashBlockSize = 64
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

	PrefixCachePluginType = "prefix-cache"
)

type Config struct {
	// The input prompt is broken into sizes of HashBlockSize to calculate block hashes . Requests
	// with length shorter than the block size will be ignored.
	HashBlockSize int `json:"hashBlockSize"`
	// MaxPrefixBlocksToMatch is the maximum number of prefix blocks to match. Input beyond this limit will
	// be ignored.
	MaxPrefixBlocksToMatch int `json:"maxPrefixBlocksToMatch"`
	// Max capacity size of the LRU indexer in number of entries per server (pod).
	LRUCapacityPerServer int `json:"lruCapacityPerServer"`
}

type Plugin struct {
	Config
	plugins.TypedName
	indexer Indexer
}

// podSet holds an pods servers that may have a specific prefix hash.
type podSet map[ServerID]struct{}

type Indexer interface {
	Get(hash BlockHash) podSet
	Add(hashes []BlockHash, server ServerID)
}

// BlockHash is a hash of the block of request body.
type BlockHash uint64

type ServerID k8stypes.NamespacedName

func (s ServerID) String() string {
	return k8stypes.NamespacedName(s).String()
}

// compile-time type validation
var _ types.StateData = &SchedulingContextState{}

// SchedulingContextState is the state of this plugin to be used during a scheduling cycle.
type SchedulingContextState struct {
	// PrefixHashes is a list of prefix hashes of the request prompt broken into blocks.
	PrefixHashes []BlockHash
	// A map of server to its longest prefix cache match length.
	PrefixCacheServers map[ServerID]int
}

func (s *SchedulingContextState) Clone() types.StateData {
	prefixHashes := make([]BlockHash, len(s.PrefixHashes))
	copy(prefixHashes, s.PrefixHashes)
	prefixCacheServers := make(map[ServerID]int, len(s.PrefixCacheServers))
	for key, value := range s.PrefixCacheServers {
		prefixCacheServers[key] = value
	}

	return &SchedulingContextState{
		PrefixHashes:       prefixHashes,
		PrefixCacheServers: prefixCacheServers,
	}
}

// compile-time type assertion
var _ framework.Scorer = &Plugin{}
var _ framework.PostCycle = &Plugin{}

// PrefixCachePluginFactory defines the factory function for Prefix plugin.
func PrefixCachePluginFactory(name string, rawParameters json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	parameters := Config{
		HashBlockSize:          DefaultHashBlockSize,
		MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}
	if err := json.Unmarshal(rawParameters, &parameters); err != nil {
		return nil, fmt.Errorf("failed to parse the parameters of the %s plugin. Error: %s", PrefixCachePluginType, err)
	}

	return New(parameters).WithName(name), nil
}

// New initializes a new prefix Plugin and returns its pointer.
func New(config Config) *Plugin {
	capacity := config.LRUCapacityPerServer
	if capacity <= 0 {
		capacity = DefaultLRUCapacityPerServer
		log.FromContext(context.TODO()).V(logutil.DEFAULT).Info(
			"LRUCapacityPerServer is not positive, using default value",
			"defaultCapacity", DefaultLRUCapacityPerServer,
		)
	}

	return &Plugin{
		TypedName: plugins.NewTypedName(PrefixCachePluginType, PrefixCachePluginType),
		Config:    config,
		indexer:   newIndexer(capacity),
	}
}

// WithName sets the name of the plugin.
func (m *Plugin) WithName(name string) *Plugin {
	m.TypedName.SetName(name)
	return m
}

// Score returns the scoring result for the given list of pods based on context.
func (m *Plugin) Score(ctx context.Context, cycleState *types.CycleState, request *types.LLMRequest, pods []types.Pod) map[types.Pod]float64 {
	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)
	// pre score step, hashing prompt and find longest prefix match.
	hashes := hashPrompt(ctx, request, m.HashBlockSize, m.MaxPrefixBlocksToMatch)
	state := &SchedulingContextState{
		PrefixHashes:       hashes,
		PrefixCacheServers: m.matchLongestPrefix(ctx, hashes),
	}

	cycleState.Write(types.StateKey(m.Type()), state)
	loggerTrace.Info(fmt.Sprintf("cached servers: %+v", state.PrefixCacheServers), "hashes", state.PrefixHashes)
	// calculate the scores of pods
	scores := make(map[types.Pod]float64, len(pods))

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

// PostCycle records in the plugin cache the result of the scheduling selection.
func (m *Plugin) PostCycle(ctx context.Context, cycleState *types.CycleState, res *types.ProfileRunResult) {
	targetPod := res.TargetPod.GetPod()
	state, err := types.ReadCycleStateKey[*SchedulingContextState](cycleState, PrefixCachePluginType)
	if err != nil {
		log.FromContext(ctx).Error(err, "failed to read prefix plugin cycle state")
		return
	}

	m.indexer.Add(state.PrefixHashes, ServerID(targetPod.NamespacedName))

	total := len(state.PrefixHashes)
	matchLen := state.PrefixCacheServers[ServerID(targetPod.NamespacedName)]
	metrics.RecordPrefixCacheMatch(matchLen*m.HashBlockSize, total*m.HashBlockSize)
}

// matchLongestPrefix returns a map of servers and length of prefix that each server caches.
func (m *Plugin) matchLongestPrefix(ctx context.Context, hashes []BlockHash) map[ServerID]int {
	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)
	res := make(map[ServerID]int)
	// Use a greedy strategy to search from the longest prefix.
	// NOTE: It's possible to further optimize this with a binary search.
	for i := 0; i < len(hashes); i++ {
		hash := hashes[i]
		cachedServers := m.indexer.Get(hash)
		if len(cachedServers) == 0 {
			break
		} else {
			loggerTrace.Info("Found cached servers", "cachedServers", cachedServers, "total # blocks", len(hashes), "longest prefix", i)
			for server := range cachedServers {
				// Update servers with their longest prefix match.
				res[server]++

			}
		}
	}
	return res
}

// hashPrompt divides the prompt into blocks and calculate the prefix cache for each block.
// hash(0) is the hash of the model name, since different models generally don't share prefix cache.
// For block i, hash(i) = hash(block i content, hash(i-1)).
func hashPrompt(ctx context.Context, request *types.LLMRequest, cacheBlockSize int, maxPrefixBlocks int) []BlockHash {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	prompt := []byte(request.Prompt)
	if len(prompt) < cacheBlockSize {
		loggerDebug.Info("Request body too small for prefix cache", "size", len(prompt), "block size", cacheBlockSize)
		return nil
	}
	if len(prompt) > cacheBlockSize*maxPrefixBlocks {
		loggerDebug.Info("Truncating input", "size", len(prompt), "max prefix blocks", maxPrefixBlocks, "block size", cacheBlockSize)
		prompt = prompt[:maxPrefixBlocks*cacheBlockSize]
	}
	// Split the body into blocks of size cacheBlockSize. The +1 is to account for the model.
	// If the last block is smaller than cacheBlockSize, it will be ignored.
	res := make([]BlockHash, 0, 1+len(prompt)/cacheBlockSize)
	// Add the model to the first block hash so that different models have different hashes even with the same body.
	res = append(res, BlockHash(xxhash.Sum64String(request.TargetModel)))
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
