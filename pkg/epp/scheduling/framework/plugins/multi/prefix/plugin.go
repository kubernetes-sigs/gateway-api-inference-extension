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
	"sync"
	"time"

	"github.com/cespare/xxhash/v2"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	// vLLM default token block size is 16, and a good guess of average characters per token is 4.
	DefaultBlockSize = 64
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

	PrefixCachePluginType = "prefix-cache-scorer"
)

const (
	PodActiveCheckInterval = 2 * time.Minute

	// An estimated average characters per token, used since the request we cached is not tokenized.
	averageCharactersPerToken = 4
)

var DefaultConfig = Config{
	DefaultBlockSize:       DefaultBlockSize,
	MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
	LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
}

type Config struct {
	// The input prompt is broken into sizes of BlockSize to calculate block hashes . Requests
	// with length shorter than the block size will be ignored.
	DefaultBlockSize int `json:"blockSize"`
	// MaxPrefixBlocksToMatch is the maximum number of prefix blocks to match. Input beyond this limit will
	// be ignored.
	MaxPrefixBlocksToMatch int `json:"maxPrefixBlocksToMatch"`
	// Max capacity size of the LRU indexer in number of entries per server (pod).
	LRUCapacityPerServer int `json:"lruCapacityPerServer"`
}

type Plugin struct {
	typedName   plugins.TypedName
	config      Config
	pluginState *plugins.PluginState
	indexer     Indexer
	wg          sync.WaitGroup
}

// podSet holds an pods servers that may have a specific prefix hash.
type podSet map[ServerID]struct{}

type Indexer interface {
	Get(hash BlockHash) podSet
	Add(hashes []BlockHash, server Server)
	RemovePod(server ServerID)
	Pods() []ServerID
}

// BlockHash is a hash of the block of request body.
type BlockHash uint64

type Server struct {
	ServerID
	numOfGPUBlocks int
}

type ServerID k8stypes.NamespacedName

func (s ServerID) String() string {
	return k8stypes.NamespacedName(s).String()
}

// compile-time type validation
var _ plugins.StateData = &SchedulingContextState{}

// SchedulingContextState is the state of this plugin to be used during a scheduling cycle.
type SchedulingContextState struct {
	// PrefixHashes is a list of prefix hashes of the request prompt broken into blocks.
	PrefixHashes []BlockHash
	// A map of server to its longest prefix cache match length.
	PrefixCacheServers map[ServerID]int
}

func (s *SchedulingContextState) Clone() plugins.StateData {
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
var (
	_ framework.Scorer          = &Plugin{}
	_ requestcontrol.PreRequest = &Plugin{}
)

// PrefixCachePluginFactory defines the factory function for Prefix plugin.
func PrefixCachePluginFactory(name string, rawParameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	parameters := Config{
		DefaultBlockSize:       DefaultBlockSize,
		MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}

	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the %s plugin. Error: %s", PrefixCachePluginType, err)
		}
	}

	p := New(handle.Context(), parameters).WithName(name)
	go p.CleanUpInactivePods(handle.Context(), handle)
	return p, nil
}

// New initializes a new prefix Plugin and returns its pointer.
func New(ctx context.Context, config Config) *Plugin {
	capacity := config.LRUCapacityPerServer
	if capacity <= 0 {
		capacity = DefaultLRUCapacityPerServer
		log.FromContext(ctx).V(logutil.DEFAULT).Info(
			"LRUCapacityPerServer is not positive, using default value",
			"defaultCapacity", DefaultLRUCapacityPerServer,
		)
	}

	return &Plugin{
		typedName:   plugins.TypedName{Type: PrefixCachePluginType, Name: PrefixCachePluginType},
		config:      config,
		pluginState: plugins.NewPluginState(ctx),
		indexer:     newIndexer(ctx, capacity),
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *Plugin) TypedName() plugins.TypedName {
	return p.typedName
}

// WithName sets the name of the plugin.
func (p *Plugin) WithName(name string) *Plugin {
	p.typedName.Name = name
	return p
}

// Score returns the scoring result for the given list of pods based on context.
func (p *Plugin) Score(ctx context.Context, cycleState *types.CycleState, request *types.LLMRequest, pods []types.Pod) map[types.Pod]float64 {
	// pre score step, hashing prompt and find longest prefix match.
	hashes := hashPrompt(ctx, request, getBlockSize(pods, p.config.DefaultBlockSize), p.config.MaxPrefixBlocksToMatch)
	state := &SchedulingContextState{
		PrefixHashes:       hashes,
		PrefixCacheServers: p.matchLongestPrefix(ctx, hashes),
	}

	cycleState.Write(plugins.StateKey(p.TypedName().String()), state)
	p.pluginState.Write(request.RequestId, plugins.StateKey(p.TypedName().String()), state)
	log.FromContext(ctx).V(logutil.TRACE).Info("prefix cached state", "cached-servers", state.PrefixCacheServers, "hashes", state.PrefixHashes)
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

// PreRequest records in the plugin cache the result of the scheduling selection.
func (p *Plugin) PreRequest(ctx context.Context, request *types.LLMRequest, schedulingResult *types.SchedulingResult) {
	primaryProfileResult := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	targetPod := primaryProfileResult.TargetPods[0].GetPod() // get the first pod of the primary profile
	gpuBlocks := primaryProfileResult.TargetPods[0].GetMetrics().CacheNumGPUBlocks

	state, err := plugins.ReadPluginStateKey[*SchedulingContextState](p.pluginState, request.RequestId, plugins.StateKey(p.TypedName().String()))
	p.pluginState.Delete(request.RequestId) // delete the state explicitly after completing using it
	if err != nil {
		log.FromContext(ctx).Error(err, "failed to read prefix plugin state", "requestID", request.RequestId)
		return
	}

	// This function is just adding data, it does not need to block other operations.
	// TODO: look into making this entire function async, none of this needs to be done in-band
	// The PR that introduces this change is meant as a cherrypick, so it was minimally invasive.
	// WaitGroup is added to the Plugin struct to allow waiting in tests.
	p.wg.Add(1)
	go func() {
		p.indexer.Add(state.PrefixHashes, Server{
			ServerID(targetPod.NamespacedName),
			gpuBlocks,
		})
		p.wg.Done()
	}()

	total := len(state.PrefixHashes)
	matchLen := state.PrefixCacheServers[ServerID(targetPod.NamespacedName)]

	blockSize := getBlockSize(primaryProfileResult.TargetPods, p.config.DefaultBlockSize)
	metrics.RecordPrefixCacheMatch(matchLen*blockSize, total*blockSize)
}

// matchLongestPrefix returns a map of servers and length of prefix that each server caches.
func (p *Plugin) matchLongestPrefix(ctx context.Context, hashes []BlockHash) map[ServerID]int {
	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)
	res := make(map[ServerID]int)
	// Use a greedy strategy to search from the longest prefix.
	// NOTE: It's possible to further optimize this with a binary search.
	for i := 0; i < len(hashes); i++ {
		hash := hashes[i]
		cachedServers := p.indexer.Get(hash)
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

// CleanUpInactivePods starts a goroutine that watches for inactive pods.
func (m *Plugin) CleanUpInactivePods(ctx context.Context, handle plugins.Handle) {
	logger := log.FromContext(ctx).V(logutil.VERBOSE)
	ticker := time.NewTicker(PodActiveCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			activePodMetrics := handle.PodList(func(_ backendmetrics.PodMetrics) bool { return true })
			activePods := make(map[ServerID]struct{}, len(activePodMetrics))
			for _, pm := range activePodMetrics {
				activePods[ServerID(pm.GetPod().NamespacedName)] = struct{}{}
			}

			for _, pod := range m.indexer.Pods() {
				if _, ok := activePods[pod]; !ok {
					m.indexer.RemovePod(pod)
					logger.Info("Removed pod not in active set", "pod", pod)
				}
			}
		}
	}
}

// hashPrompt divides the prompt into blocks and calculate the prefix cache for each block.
// hash[0] is calculated including the model name and cache_salt(if provided), since different models generally don't share prefix cache.
// For block i, hash(i) = hash(block i content, hash(i-1)).
func hashPrompt(ctx context.Context, request *types.LLMRequest, cacheBlockSize int, maxPrefixBlocks int) []BlockHash {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	if request == nil || request.Body == nil {
		loggerDebug.Info("Request or request data is nil, skipping hashing")
		return nil
	}

	userInput, err := getUserInputBytes(request)
	if err != nil {
		loggerDebug.Error(err, "Failed to get user input bytes")
		return nil
	}

	if len(userInput) < cacheBlockSize {
		loggerDebug.Info("Request body too small for prefix cache", "size", len(userInput), "block size", cacheBlockSize)
		return nil
	}
	if len(userInput) > cacheBlockSize*maxPrefixBlocks {
		loggerDebug.Info("Truncating input", "size", len(userInput), "max prefix blocks", maxPrefixBlocks, "block size", cacheBlockSize)
		userInput = userInput[:maxPrefixBlocks*cacheBlockSize]
	}
	// Split the body into blocks of size cacheBlockSize.
	// If the last block is smaller than cacheBlockSize, it will be ignored.
	res := make([]BlockHash, 0, len(userInput)/cacheBlockSize)
	// Add the model to the first block hash so that different models have different hashes even with the same body.
	h := xxhash.New()
	_, _ = h.Write([]byte(request.TargetModel))
	if cacheSalt := request.Body.CacheSalt(); cacheSalt != "" {
		_, _ = h.Write([]byte(cacheSalt))
	}

	prevBlockHash := BlockHash(h.Sum64())
	for i := 0; i+cacheBlockSize <= len(userInput); i += cacheBlockSize {
		h.Reset()
		_, _ = h.Write(userInput[i : i+cacheBlockSize])
		_, _ = h.Write(toBytes(prevBlockHash))
		res = append(res, BlockHash(h.Sum64()))

		prevBlockHash = res[len(res)-1]
	}
	return res
}

func toBytes(i BlockHash) []byte {
	bytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(bytes, uint64(i))
	return bytes
}

func getUserInputBytes(request *types.LLMRequest) ([]byte, error) {
	if request.Body.Completions != nil { // assumed to be valid if not nil
		return []byte(request.Body.Completions.Prompt), nil
	}

	// must be chat-completions request at this point, return bytes of entire messages
	return json.Marshal(request.Body.ChatCompletions.Messages)
}

func getBlockSize(pods []types.Pod, defaultBlockSize int) int {
	if len(pods) == 0 {
		return defaultBlockSize
	}

	// Since all PODs originate from the same inference pool, they are considered to have identical configurations.
	// Therefore, using the CacheBlockSize value from the first POD suffices.
	if pod := pods[0]; pod.GetMetrics() != nil {
		cacheBlockSize := pod.GetMetrics().CacheBlockSize
		if cacheBlockSize > 0 {
			return cacheBlockSize * averageCharactersPerToken
		}
	}
	return defaultBlockSize
}
