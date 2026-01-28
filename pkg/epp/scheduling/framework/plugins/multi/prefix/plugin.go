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
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/cespare/xxhash/v2"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/util/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer/plugins/approximateprefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
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
)

const (
	PodActiveCheckInterval = 2 * time.Minute

	// An estimated average characters per token, used since the request we cached is not tokenized.
	averageCharactersPerToken = 4
)

var DefaultConfig = Config{
	AutoTune:               true,
	BlockSize:              0,
	BlockSizeTokens:        0,
	MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
	LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
}

type Config struct {
	// If set to true, the plugin will automatically adjust the configuration based on various
	// metrics from the model servers.
	AutoTune bool `json:"autoTune"`
	// The input prompt is broken into sizes of BlockSizeTokens to calculate block hashes. Requests
	// with length shorter than the block size will be ignored.
	BlockSizeTokens int `json:"blockSizeTokens"`
	// Depricated: Legacy block size defined in number of characters.
	// In case only BlockSize is defined in the configuration - plugin initialization will fail.
	// In case both BlockSize and BlockSizeTokens are defined - BlockSizeTokens is used.
	BlockSize int `json:"blockSize"`
	// MaxPrefixBlocksToMatch is the maximum number of prefix blocks to match. Input beyond this limit will
	// be ignored.
	MaxPrefixBlocksToMatch int `json:"maxPrefixBlocksToMatch"`
	// Max capacity size of the LRU indexer in number of entries per server (pod).
	LRUCapacityPerServer int `json:"lruCapacityPerServer"`
}

type Plugin struct {
	typedName   plugin.TypedName
	config      Config
	pluginState *plugin.PluginState
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
var _ plugin.StateData = &SchedulingContextState{}

// SchedulingContextState is the state of this plugin to be used during a scheduling cycle.
type SchedulingContextState struct {
	// PrefixHashes is a list of prefix hashes of the request prompt broken into blocks.
	PrefixHashes []BlockHash
	// A map of server to its longest prefix cache match length in blocks.
	PrefixCacheServers map[ServerID]int
}

func (s *SchedulingContextState) Clone() plugin.StateData {
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
func PrefixCachePluginFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := DefaultConfig

	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the %s plugin. Error: %s", PrefixCachePluginType, err)
		}
	}

	p, err := New(handle.Context(), parameters)
	if err != nil {
		return nil, err
	}

	p.WithName(name)
	go p.CleanUpInactivePods(handle.Context(), handle)
	return p, nil
}

// New initializes a new prefix Plugin and returns its pointer.
func New(ctx context.Context, config Config) (*Plugin, error) {
	// invalid configuration: only BlockSize is defined
	if config.BlockSize > 0 && config.BlockSizeTokens <= 0 {
		err := errors.New("BlockSize is depricated, use BlockSizeTokens instead, the value should be defined in tokens")
		log.FromContext(ctx).V(logutil.DEFAULT).Error(err, "invalid prefix plugin configuration")
		return nil, err
	}

	if config.LRUCapacityPerServer <= 0 {
		config.LRUCapacityPerServer = DefaultLRUCapacityPerServer
		log.FromContext(ctx).V(logutil.DEFAULT).Info(
			"LRUCapacityPerServer is not positive, using default value",
			"defaultCapacity", DefaultLRUCapacityPerServer,
		)
	}

	if config.BlockSizeTokens <= 0 {
		config.BlockSizeTokens = DefaultBlockSizeTokens
		log.FromContext(ctx).V(logutil.DEFAULT).Info("BlockSize is not positive, using default value",
			"default", DefaultBlockSizeTokens)
	}

	if config.MaxPrefixBlocksToMatch <= 0 {
		config.MaxPrefixBlocksToMatch = DefaultMaxPrefixBlocks
		log.FromContext(ctx).V(logutil.DEFAULT).Info("MaxPrefixBlocksToMatch is not positive, using default value",
			"default", DefaultMaxPrefixBlocks)
	}

	log.FromContext(ctx).V(logutil.DEFAULT).Info("PrefixCachePlugin initialized", "config", config)
	return &Plugin{
		typedName:   plugin.TypedName{Type: PrefixCachePluginType, Name: PrefixCachePluginType},
		config:      config,
		pluginState: plugin.NewPluginState(ctx),
		indexer:     newIndexer(ctx, config.LRUCapacityPerServer),
	}, nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (p *Plugin) Category() framework.ScorerCategory {
	return framework.Affinity
}

// WithName sets the name of the plugin.
func (p *Plugin) WithName(name string) *Plugin {
	p.typedName.Name = name
	return p
}

func (p *Plugin) Produces() map[string]any {
	return map[string]any{approximateprefix.PrefixCacheMatchInfoKey: approximateprefix.PrefixCacheMatchInfo{}}
}

func (p *Plugin) Consumes() map[string]any {
	return map[string]any{}
}

// PrepareRequestData hashes prompt, finds longest prefix match and stores it in endpoint as attribute.
func (p *Plugin) PrepareRequestData(ctx context.Context, request *framework.LLMRequest, endpoints []framework.Endpoint) error {
	blockSize := getBlockSize(endpoints, p.config)
	hashes := hashPrompt(ctx, request, blockSize, p.config.MaxPrefixBlocksToMatch)
	state := &SchedulingContextState{
		PrefixHashes:       hashes,
		PrefixCacheServers: p.matchLongestPrefix(ctx, hashes),
	}
	total := len(state.PrefixHashes)

	for _, endpoint := range endpoints {
		matchLen := state.PrefixCacheServers[ServerID(endpoint.GetMetadata().NamespacedName)]
		endpoint.Put(approximateprefix.PrefixCacheMatchInfoKey, approximateprefix.NewPrefixCacheMatchInfo(matchLen, total, blockSize))
	}

	// Store the state in plugin state for later use.
	p.pluginState.Write(request.RequestId, plugin.StateKey(p.TypedName().String()), state)
	return nil
}

// Score returns the scoring result for the given list of pods based on context.
func (p *Plugin) Score(ctx context.Context, cycleState *framework.CycleState, request *framework.LLMRequest, endpoints []framework.Endpoint) map[framework.Endpoint]float64 {
	// pre score step, hashing prompt and find longest prefix match.
	blockSize := getBlockSize(endpoints, p.config)
	hashes := hashPrompt(ctx, request, blockSize, p.config.MaxPrefixBlocksToMatch)
	state := &SchedulingContextState{
		PrefixHashes:       hashes,
		PrefixCacheServers: p.matchLongestPrefix(ctx, hashes),
	}

	cycleState.Write(plugin.StateKey(p.TypedName().String()), state)

	// store the state in plugin state for later use in PreRequest. This may go away once we default to prepare request data plugin hook.
	p.pluginState.Write(request.RequestId, plugin.StateKey(p.TypedName().String()), state)
	log.FromContext(ctx).V(logutil.TRACE).Info("prefix cached state", "cached-servers", state.PrefixCacheServers, "hashes", state.PrefixHashes)
	// calculate the scores of endpoints
	scores := make(map[framework.Endpoint]float64, len(endpoints))

	// total prefix length in tokens
	total := len(state.PrefixHashes)
	endpointScoreFunc := func(endpoint framework.Endpoint) float64 {
		if total == 0 {
			return 0
		}
		matchLen := state.PrefixCacheServers[ServerID(endpoint.GetMetadata().NamespacedName)]
		return float64(matchLen) / float64(total)
	}

	for _, endpoint := range endpoints {
		scores[endpoint] = endpointScoreFunc(endpoint)
	}
	return scores
}

// PreRequest records in the plugin cache the result of the scheduling selection.
func (p *Plugin) PreRequest(ctx context.Context, request *framework.LLMRequest, schedulingResult *framework.SchedulingResult) {
	primaryProfileResult := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	targetEndpoint := primaryProfileResult.TargetEndpoints[0] // get the first endpoint of the primary profile
	servers := []Server{p.makeServer(targetEndpoint)}

	if pr, exists := schedulingResult.ProfileResults[Experimental_DefaultPrefillProfile]; exists && len(pr.TargetEndpoints) > 0 {
		servers = append(servers, p.makeServer(pr.TargetEndpoints[0]))
	}

	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.pluginState, request.RequestId, plugin.StateKey(p.TypedName().String()))
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
		for _, s := range servers {
			p.indexer.Add(state.PrefixHashes, s)
		}
		p.wg.Done()
	}()

	total := len(state.PrefixHashes)
	matchLen := state.PrefixCacheServers[ServerID(targetEndpoint.GetMetadata().NamespacedName)]

	blockSize := getBlockSize(primaryProfileResult.TargetEndpoints, p.config)
	// report matched and total prefix length in chars
	metrics.RecordPrefixCacheMatch(matchLen*blockSize*averageCharactersPerToken, total*blockSize*averageCharactersPerToken)
}

func (p *Plugin) makeServer(targetEndpoint framework.Endpoint) Server {
	gpuBlocks := p.config.LRUCapacityPerServer
	if p.config.AutoTune && targetEndpoint.GetMetrics().CacheNumGPUBlocks > 0 {
		gpuBlocks = targetEndpoint.GetMetrics().CacheNumGPUBlocks
	}
	return Server{
		ServerID(targetEndpoint.GetMetadata().NamespacedName),
		gpuBlocks,
	}
}

// matchLongestPrefix returns a map of servers and length of prefix that each server caches, prefix length is defined in blocks.
func (p *Plugin) matchLongestPrefix(ctx context.Context, hashes []BlockHash) map[ServerID]int {
	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)
	res := make(map[ServerID]int)
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

// CleanUpInactivePods starts a goroutine that watches for inactive pods.
func (m *Plugin) CleanUpInactivePods(ctx context.Context, handle plugin.Handle) {
	logger := log.FromContext(ctx).V(logutil.VERBOSE)
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
func hashPrompt(ctx context.Context, request *framework.LLMRequest, blockSizeTokens int, maxPrefixBlocks int) []BlockHash {
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

	// convert block size from tokens to characters
	cacheBlockSizeChars := blockSizeTokens * averageCharactersPerToken

	if len(userInput) < cacheBlockSizeChars {
		loggerDebug.Info("Request body too small for prefix cache", "size", len(userInput), "block size in chars", cacheBlockSizeChars)
		return nil
	}
	if len(userInput) > cacheBlockSizeChars*maxPrefixBlocks {
		loggerDebug.Info("Truncating input", "size", len(userInput), "max prefix blocks", maxPrefixBlocks, "block size in chars", cacheBlockSizeChars)
		userInput = userInput[:maxPrefixBlocks*cacheBlockSizeChars]
	}
	// Split the body into blocks of size cacheBlockSize.
	// If the last block is smaller than cacheBlockSize, it will be ignored.
	res := make([]BlockHash, 0, len(userInput)/cacheBlockSizeChars)
	// Add the model to the first block hash so that different models have different hashes even with the same body.
	h := xxhash.New()
	_, _ = h.Write([]byte(request.TargetModel))
	if cacheSalt := request.Body.CacheSalt(); cacheSalt != "" {
		_, _ = h.Write([]byte(cacheSalt))
	}

	prevBlockHash := BlockHash(h.Sum64())
	for i := 0; i+cacheBlockSizeChars <= len(userInput); i += cacheBlockSizeChars {
		h.Reset()
		_, _ = h.Write(userInput[i : i+cacheBlockSizeChars])
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

func getUserInputBytes(request *framework.LLMRequest) ([]byte, error) {
	if request.Body.Completions != nil { // assumed to be valid if not nil
		return []byte(request.Body.Completions.Prompt), nil
	}

	// must be chat-completions request at this point, return bytes of entire messages
	return json.Marshal(request.Body.ChatCompletions.Messages)
}

// getBlockSize returns the block size in tokens.
// In case of auto-tune uses the block size from the first endpoint, otherwise uses the block size from the configuration
func getBlockSize(endpoints []framework.Endpoint, config Config) int {
	if !config.AutoTune {
		return config.BlockSizeTokens
	}

	// Fallback to BlockSize if no metrics are available.
	if len(endpoints) == 0 {
		return config.BlockSizeTokens
	}

	// Since all Endpoints originate from the same inference pool, they are considered to have identical configurations.
	// Therefore, using the CacheBlockSize value from the first Endpoint suffices.
	if endpoint := endpoints[0]; endpoint.GetMetrics() != nil {
		cacheBlockSize := endpoint.GetMetrics().CacheBlockSize
		if cacheBlockSize > 0 {
			return cacheBlockSize
		}
	}
	return config.BlockSizeTokens
}
