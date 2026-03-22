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
	"time"

	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// Indexer maintains an LRU cache of prompt prefix hashes and the server(s) that might have that
// prefix cached.
type Indexer interface {
	Get(hash BlockHash) PodSet
	Add(hashes []BlockHash, server Server)
	RemovePod(server ServerID)
	Pods() []ServerID
}

// PodSet holds a set of pods that may have a specific prefix hash.
type PodSet map[ServerID]struct{}

// BlockHash is a hash of a block of request data.
type BlockHash uint64

// Server contains information about a specific server/pod and its cache capacity.
type Server struct {
	ServerID
	NumOfGPUBlocks int
}

// ServerID is a unique identifier for a server, based on its NamespacedName.
type ServerID k8stypes.NamespacedName

func (s ServerID) String() string {
	return k8stypes.NamespacedName(s).String()
}

// SchedulingContextState is the state of this plugin to be used during a scheduling cycle.
type SchedulingContextState struct {
	// PrefixHashes is a list of prefix hashes of the request prompt broken into blocks.
	PrefixHashes []BlockHash
	// A map of server to its longest prefix cache match length in blocks.
	PrefixCacheServers map[ServerID]int
}

// Clone creates a deep copy of the SchedulingContextState.
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

const (
	// Experimental_DefaultPrefillProfile is a hardcoded profile name for prefill nodes.
	Experimental_DefaultPrefillProfile = "prefill"

	// ApproxPrefixCachePlugin is the name of the data preparation plugin.
	ApproxPrefixCachePlugin = "approx-prefix-cache"

	// PodActiveCheckInterval is the interval at which we check if pods are still active.
	PodActiveCheckInterval = 2 * time.Minute

	// DefaultBlockSizeTokens is the default token block size (vLLM default is 16).
	DefaultBlockSizeTokens = 16

	// DefaultMaxPrefixBlocks is the maximum number of blocks to match.
	DefaultMaxPrefixBlocks = 256

	// DefaultLRUCapacityPerServer is the default capacity of the LRU indexer per server.
	DefaultLRUCapacityPerServer = 31250

	// averageCharactersPerToken is an estimated average characters per token.
	averageCharactersPerToken = 4
)

// AverageCharactersPerToken returns the estimated average characters per token.
func AverageCharactersPerToken() int {
	return averageCharactersPerToken
}

// Config defines the configuration for the prefix cache plugins.
type Config struct {
	// If set to true, the plugin will automatically adjust the configuration based on various
	// metrics from the model servers.
	AutoTune bool `json:"autoTune"`
	// The input prompt is broken into sizes of BlockSizeTokens to calculate block hashes.
	BlockSizeTokens int `json:"blockSizeTokens"`
	// Deprecated: Legacy block size defined in number of characters.
	BlockSize int `json:"blockSize"`
	// MaxPrefixBlocksToMatch is the maximum number of prefix blocks to match.
	MaxPrefixBlocksToMatch int `json:"maxPrefixBlocksToMatch"`
	// Max capacity size of the LRU indexer in number of entries per server (pod).
	LRUCapacityPerServer int `json:"lruCapacityPerServer"`
}

// DefaultConfig provides sensible defaults for the prefix cache plugins.
var DefaultConfig = Config{
	AutoTune:               true,
	BlockSize:              0,
	BlockSizeTokens:        DefaultBlockSizeTokens,
	MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
	LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
}
