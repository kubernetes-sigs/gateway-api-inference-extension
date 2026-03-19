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
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// mockIndexer is a mock implementation of the Indexer for testing.
type mockIndexer struct {
	cache map[BlockHash]PodSet
}

// Get returns the servers that have the given block hash.
func (m *mockIndexer) Get(hash BlockHash) PodSet {
	if servers, ok := m.cache[hash]; ok {
		return servers
	}
	return nil
}

// Put adds a block hash to the cache for a given server.
func (m *mockIndexer) Put(server ServerID, hash BlockHash) {
	if m.cache == nil {
		m.cache = make(map[BlockHash]PodSet)
	}
	if _, ok := m.cache[hash]; !ok {
		m.cache[hash] = make(PodSet)
	}
	m.cache[hash][server] = struct{}{}
}

func (m *mockIndexer) Add(hashes []BlockHash, server Server) {
	for _, hash := range hashes {
		m.Put(server.ServerID, hash)
	}
}

func (m *mockIndexer) RemovePod(server ServerID) {}
func (m *mockIndexer) Pods() []ServerID {
	return nil
}

// mockEndpoint is a mock implementation of the scheduling.Endpoint for testing.
type mockEndpoint struct {
	data    map[string]datalayer.Cloneable
	meta    *datalayer.EndpointMetadata
	metrics *datalayer.Metrics
}

func newMockEndpoint(name string) *mockEndpoint {
	return &mockEndpoint{
		data: make(map[string]datalayer.Cloneable),
		meta: &datalayer.EndpointMetadata{
			NamespacedName: types.NamespacedName{Name: name, Namespace: "default"},
		},
		metrics: &datalayer.Metrics{},
	}
}

func (m *mockEndpoint) Get(key string) (datalayer.Cloneable, bool) {
	val, ok := m.data[key]
	return val, ok
}

func (m *mockEndpoint) Put(key string, val datalayer.Cloneable) {
	m.data[key] = val
}

func (m *mockEndpoint) GetMetadata() *datalayer.EndpointMetadata {
	return m.meta
}

func (m *mockEndpoint) GetMetrics() *datalayer.Metrics {
	return m.metrics
}

func (m *mockEndpoint) String() string {
	return m.meta.NamespacedName.String()
}

func (m *mockEndpoint) Keys() []string {
	keys := make([]string, 0, len(m.data))
	for k := range m.data {
		keys = append(keys, k)
	}
	return keys
}

type mockPrefixScorer struct {
	indexer Indexer
}

func (m *mockPrefixScorer) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: attrprefix.PrefixCachePluginType, Name: attrprefix.PrefixCachePluginType}
}
func (m *mockPrefixScorer) Indexer() Indexer       { return m.indexer }
func (m *mockPrefixScorer) SetIndexer(i Indexer)  { m.indexer = i }
func (m *mockPrefixScorer) Produces() map[string]any { return nil }
func (m *mockPrefixScorer) Consumes() map[string]any { return nil }

func createTestPlugins(ctx context.Context, config Config, indexer Indexer) (*ApproxPrefixCache, PrefixScorer) {
	prefixScorer := &mockPrefixScorer{indexer: indexer}
	p, _ := New(ctx, config, indexer, prefixScorer)
	return p, prefixScorer
}

func TestNew(t *testing.T) {
	config := Config{
		AutoTune:               true,
		BlockSizeTokens:        16,
		MaxPrefixBlocksToMatch: 256,
		LRUCapacityPerServer:   31250,
	}
	p, err := New(context.Background(), config, &mockIndexer{}, nil)
	assert.NoError(t, err)
	assert.NotNil(t, p)
	assert.IsType(t, &ApproxPrefixCache{}, p)
}

func TestTypedName(t *testing.T) {
	config := Config{
		AutoTune:               true,
		BlockSizeTokens:        16,
		MaxPrefixBlocksToMatch: 256,
		LRUCapacityPerServer:   31250,
	}
	p, _ := New(context.Background(), config, &mockIndexer{}, nil)
	expected := plugin.TypedName{
		Type: attrprefix.PrefixCachePluginType,
		Name: ApproxPrefixCachePlugin,
	}
	assert.Equal(t, expected, p.TypedName())
}

func TestMatchLongestPrefix(t *testing.T) {
	server1 := ServerID(types.NamespacedName{Name: "server1", Namespace: "default"})
	server2 := ServerID(types.NamespacedName{Name: "server2", Namespace: "default"})

	hash1 := BlockHash(1)
	hash2 := BlockHash(2)
	hash3 := BlockHash(3)

	testCases := []struct {
		name         string
		hashes       []BlockHash
		indexerCache map[BlockHash]PodSet
		expected     map[ServerID]int
	}{
		{
			name:         "no match",
			hashes:       []BlockHash{hash1, hash2, hash3},
			indexerCache: map[BlockHash]PodSet{},
			expected:     map[ServerID]int{},
		},
		{
			name:   "partial match",
			hashes: []BlockHash{hash1, hash2, hash3},
			indexerCache: map[BlockHash]PodSet{
				hash1: {server1: {}},
			},
			expected: map[ServerID]int{server1: 1},
		},
		{
			name:   "full match",
			hashes: []BlockHash{hash1, hash2, hash3},
			indexerCache: map[BlockHash]PodSet{
				hash1: {server1: {}},
				hash2: {server1: {}},
				hash3: {server1: {}},
			},
			expected: map[ServerID]int{server1: 3},
		},
		{
			name:   "multiple servers",
			hashes: []BlockHash{hash1, hash2, hash3},
			indexerCache: map[BlockHash]PodSet{
				hash1: {server1: {}, server2: {}},
				hash2: {server1: {}},
			},
			expected: map[ServerID]int{server1: 2, server2: 1},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			p := &ApproxPrefixCache{
				indexer: &mockIndexer{cache: tc.indexerCache},
			}
			result := p.matchLongestPrefix(context.Background(), tc.hashes)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestPrepareRequestData(t *testing.T) {
	server1 := ServerID(types.NamespacedName{Name: "server1", Namespace: "default"})

	request := &scheduling.LLMRequest{
		Body: &scheduling.LLMRequestBody{
			Completions: &scheduling.CompletionsRequest{
				Prompt: "This is a prompt",
			},
		},
	}

	hashes := HashPrompt(context.Background(), request, 4, 10)
	hash1 := hashes[0]

	p, _ := createTestPlugins(context.Background(), Config{
		BlockSizeTokens:        4,
		MaxPrefixBlocksToMatch: 10,
	}, &mockIndexer{
		cache: map[BlockHash]PodSet{
			hash1: {server1: {}},
		},
	})

	pods := []scheduling.Endpoint{
		newMockEndpoint("server1"),
		newMockEndpoint("server2"),
	}

	err := p.PrepareRequestData(context.Background(), request, pods)
	assert.NoError(t, err)

	for _, pod := range pods {
		info, ok := pod.Get(attrprefix.PrefixCacheMatchInfoKey)
		assert.True(t, ok)
		prefixInfo, ok := info.(*attrprefix.PrefixCacheMatchInfo)
		assert.True(t, ok)

		podName := pod.GetMetadata().NamespacedName.Name
		if podName == "server1" {
			assert.Equal(t, 1, prefixInfo.MatchBlocks())
			assert.Equal(t, len(hashes), prefixInfo.TotalBlocks())
			assert.Equal(t, 4, prefixInfo.BlockSizeTokens())
		} else if podName == "server2" {
			assert.Equal(t, 0, prefixInfo.MatchBlocks())
			assert.Equal(t, len(hashes), prefixInfo.TotalBlocks())
			assert.Equal(t, 4, prefixInfo.BlockSizeTokens())
		} else {
			t.Errorf("unexpected pod name: %s", podName)
		}
		fmt.Printf("pod: %s, info: %+v\n", pod.GetMetadata().NamespacedName.Name, prefixInfo)
	}
}

func TestApproxPrefixCacheValidation(t *testing.T) {
	validConfigs := []Config{{
		AutoTune:               false,
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}, {
		AutoTune:               false,
		BlockSize:              1,
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}}
	invalidConfigs := []Config{{
		AutoTune:               false,
		BlockSize:              1,
		MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}}

	for _, config := range validConfigs {
		_, err := New(context.Background(), config, &mockIndexer{}, nil)
		assert.NoError(t, err)
	}

	for _, config := range invalidConfigs {
		_, err := New(context.Background(), config, &mockIndexer{}, nil)
		assert.Error(t, err)
	}
}

func TestNew_InvalidConfigFallbacks(t *testing.T) {
	tests := []struct {
		name           string
		config         Config
		expectBlock    int
		expectMaxMatch int
		expectCapacity int
	}{
		{
			name: "all zero",
			config: Config{
				BlockSizeTokens:        0,
				MaxPrefixBlocksToMatch: 0,
				LRUCapacityPerServer:   0,
			},
			expectBlock:    DefaultBlockSizeTokens,
			expectMaxMatch: DefaultMaxPrefixBlocks,
			expectCapacity: DefaultLRUCapacityPerServer,
		},
		{
			name: "negative values",
			config: Config{
				BlockSizeTokens:        -5,
				MaxPrefixBlocksToMatch: -10,
				LRUCapacityPerServer:   -100,
			},
			expectBlock:    DefaultBlockSizeTokens,
			expectMaxMatch: DefaultMaxPrefixBlocks,
			expectCapacity: DefaultLRUCapacityPerServer,
		},
		{
			name: "mixed valid and invalid",
			config: Config{
				BlockSizeTokens:        32,    // valid
				MaxPrefixBlocksToMatch: -1,    // invalid
				LRUCapacityPerServer:   50000, // valid
			},
			expectBlock:    32,
			expectMaxMatch: DefaultMaxPrefixBlocks,
			expectCapacity: 50000,
		},
		{
			name: "all valid",
			config: Config{
				BlockSizeTokens:        64,
				MaxPrefixBlocksToMatch: 200,
				LRUCapacityPerServer:   30000,
			},
			expectBlock:    64,
			expectMaxMatch: 200,
			expectCapacity: 30000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := New(context.Background(), tt.config, &mockIndexer{}, nil)

			assert.NoError(t, err)
			assert.NotEmpty(t, p)
			assert.NotNil(t, p.indexer)
			assert.Equal(t, tt.expectBlock, p.config.BlockSizeTokens)
			assert.Equal(t, tt.expectMaxMatch, p.config.MaxPrefixBlocksToMatch)
			assert.Equal(t, tt.expectCapacity, p.config.LRUCapacityPerServer)
		})
	}
}

func TestApproxPrefixCacheAutoTune(t *testing.T) {
	// Setup common test data
	podName := "pod-autotune"
	endpoint := &mockEndpoint{
		meta: &datalayer.EndpointMetadata{NamespacedName: types.NamespacedName{Name: podName, Namespace: "default"}},
		metrics: &datalayer.Metrics{
			CacheBlockSize:    16,   // 16 tokens * 4 chars/token = 64 chars per block
			CacheNumGPUBlocks: 1000, // 1000 blocks capacity
		},
		data: make(map[string]datalayer.Cloneable),
	}
	endpoints := []scheduling.Endpoint{endpoint}

	req := &scheduling.LLMRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &scheduling.LLMRequestBody{
			Completions: &scheduling.CompletionsRequest{
				// Length 128 chars.
				// If AutoTune=true (block size 64): 2 blocks
				// If AutoTune=false (block size 32): 4 blocks
				Prompt: strings.Repeat("a", 128),
			},
		},
	}

	t.Run("AutoTune Enabled", func(t *testing.T) {
		config := Config{
			AutoTune:               true,
			BlockSizeTokens:        32, // Should be ignored in favor of pod metrics (64)
			MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
			// Should be ignored in favor of pod metrics (1000)
			LRUCapacityPerServer: 1,
		}
		p, _ := createTestPlugins(context.Background(), config, &mockIndexer{})

		// Verify PrepareRequestData uses pod metrics for block size
		err := p.PrepareRequestData(context.Background(), req, endpoints)
		assert.NoError(t, err)

		info, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoKey)
		assert.True(t, ok)
		prefixInfo := info.(*attrprefix.PrefixCacheMatchInfo)
		// Block size from pod is 16 tokens * 4 = 64 chars.
		// Prompt is 128 chars.
		// Expected blocks: 128/64 = 2
		assert.Equal(t, 2, prefixInfo.TotalBlocks(), "Should use pod block size (16 tokens) -> 2 blocks")
	})

	t.Run("AutoTune Disabled", func(t *testing.T) {
		config := Config{
			AutoTune:               false,
			BlockSizeTokens:        8, // Should be used (32 chars, 8 tokens)
			MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
			LRUCapacityPerServer:   1,
		}
		p, _ := createTestPlugins(context.Background(), config, &mockIndexer{})

		// Verify PrepareRequestData uses config BlockSize
		err := p.PrepareRequestData(context.Background(), req, endpoints)
		assert.NoError(t, err)

		info, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoKey)
		assert.True(t, ok)
		prefixInfo := info.(*attrprefix.PrefixCacheMatchInfo)
		// Block size from config is 8 tokens (32 chars).
		// Prompt is 128 chars.
		// 128 / 32 = 4 blocks.
		assert.Equal(t, 4, prefixInfo.TotalBlocks(), "Should use config block size (8 tokens) -> 4 blocks")
	})
}

func TestHashPromptChatCompletions(t *testing.T) {
	config := Config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}
	_, _ = createTestPlugins(context.Background(), config, &mockIndexer{})

	// Test with chat completions request
	req1 := &scheduling.LLMRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &scheduling.LLMRequestBody{
			ChatCompletions: &scheduling.ChatCompletionsRequest{
				Messages: []scheduling.Message{
					{Role: "user", Content: scheduling.Content{Raw: "hello world"}},
					{Role: "assistant", Content: scheduling.Content{Raw: "hi there"}},
				},
			},
		},
	}
	hashes := HashPrompt(context.Background(), req1, 1, DefaultMaxPrefixBlocks)
	// Should have some hashes for the JSON-encoded messages
	assert.Greater(t, len(hashes), 1, "should have hashes for chat completions")
}

func BenchmarkApproxPrefixCacheStress(b *testing.B) {
	maxPrefixBlocks := 50000
	config := Config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: maxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}

	p, _ := createTestPlugins(context.Background(), config, &mockIndexer{})
	var promptLen []int
	for i := 1; i <= 1024; {
		promptLen = append(promptLen, i)
		i += 10
	}
	promptLen = append(promptLen, 2048, 4096, 8192, 10000, 20000, 50000)

	for i, v := range promptLen {
		b.Run(fmt.Sprintf("messages_%d_length_%d", i, v), func(b *testing.B) {
			// Generate increasing-length random prompts
			prompt := randomPrompt(4 + v)
			endpoint := newMockEndpoint(fmt.Sprintf("random-pod-%d", v))

			endpoints := []scheduling.Endpoint{endpoint}
			req := &scheduling.LLMRequest{
				RequestId:   uuid.NewString(),
				TargetModel: "model-stress",
				Body: &scheduling.LLMRequestBody{
					Completions: &scheduling.CompletionsRequest{
						Prompt: prompt,
					},
				},
			}

			b.ResetTimer()
			for j := 0; j < b.N; j++ {
				_ = p.PrepareRequestData(context.Background(), req, endpoints)
			}
		})
	}
}

func BenchmarkApproxPrefixCacheChatCompletionsStress(b *testing.B) {
	maxPrefixBlocks := 50000
	config := Config{
		BlockSizeTokens:        2,
		MaxPrefixBlocksToMatch: maxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}
	p, _ := createTestPlugins(context.Background(), config, &mockIndexer{})

	// Test scenarios: varying number of messages and message lengths
	scenarios := []struct {
		messageCount  int
		messageLength int
	}{
		{2, 50},   // Short conversation, short messages
		{2, 500},  // Short conversation, long messages
		{5, 100},  // Medium conversation, medium messages
		{10, 200}, // Long conversation, medium messages
		{20, 100}, // Very long conversation, medium messages
		{50, 50},  // Very long conversation, short messages
		{100, 25}, // Extremely long conversation, very short messages
	}

	for _, scenario := range scenarios {
		b.Run(fmt.Sprintf("messages_%d_length_%d", scenario.messageCount, scenario.messageLength), func(b *testing.B) {
			// Generate messages for this scenario
			messages := make([]scheduling.Message, scenario.messageCount)
			messages[0] = scheduling.Message{Role: "system", Content: scheduling.Content{Raw: "You are a helpful assistant."}}

			for i := 1; i < scenario.messageCount; i++ {
				role := "user"
				if i%2 == 0 {
					role = "assistant"
				}
				content := randomPrompt(scenario.messageLength)
				messages[i] = scheduling.Message{Role: role, Content: scheduling.Content{Raw: content}}
			}

			endpoint := newMockEndpoint(fmt.Sprintf("chat-pod-%d-%d", scenario.messageCount, scenario.messageLength))
			endpoints := []scheduling.Endpoint{endpoint}

			req := &scheduling.LLMRequest{
				RequestId:   uuid.NewString(),
				TargetModel: "chat-model-stress",
				Body: &scheduling.LLMRequestBody{
					ChatCompletions: &scheduling.ChatCompletionsRequest{
						Messages: messages,
					},
				},
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = p.PrepareRequestData(context.Background(), req, endpoints)
			}
		})
	}
}

// randomPrompt generates a pseudo-random string of length n using lowercase letters.
func randomPrompt(n int) string {
	runes := []rune("abcdefghijklmnopqrstuvwxyz")
	var sb strings.Builder
	for i := 0; i < n; i++ {
		sb.WriteRune(runes[rand.Intn(len(runes))])
	}
	return sb.String()
}
