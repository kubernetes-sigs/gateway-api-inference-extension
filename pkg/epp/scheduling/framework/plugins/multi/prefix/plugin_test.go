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
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	dplugins "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer/plugins/approximateprefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// static check to ensure Plugin implements the PrepareDataPlugin interface.
var _ requestcontrol.PrepareDataPlugin = &Plugin{}

func TestPrefixPluginCompletion(t *testing.T) {
	config := Config{
		BlockSize:              4,
		MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}
	plugin := New(context.Background(), config)

	pod1 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, MetricsState: backendmetrics.NewMetricsState()}
	pod2 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, MetricsState: backendmetrics.NewMetricsState()}
	pods := []types.Pod{pod1, pod2}

	// First request.
	req1 := &types.LLMRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: "aaaaaa",
			},
		},
	}
	scores := plugin.Score(context.Background(), types.NewCycleState(), req1, pods)
	state, err := plugins.ReadPluginStateKey[*SchedulingContextState](plugin.pluginState, req1.RequestId, plugins.StateKey(plugin.TypedName().String()))
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 6, hash block size is 4, the last 2 characters are ignored.
	// Total hashes = 1 (the first one is for the prefix with model)
	assert.Equal(t, 1, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 0, len(state.PrefixCacheServers), "there shouldn't be any cached servers")
	assert.Equal(t, float64(0), scores[pod1], "score for pod1")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	// Simulate pod1 was picked.
	schedulingResult := &types.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*types.ProfileRunResult{
			"default": {TargetPods: []types.Pod{pod1}},
		},
	}
	plugin.PreRequest(context.Background(), req1, schedulingResult)
	plugin.wg.Wait()

	// Second request doesn't share any prefix with first one. It should be added to the cache but
	// the pod score should be 0.
	req2 := &types.LLMRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model2",
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: "bbbbbb",
			},
		},
	}
	scores = plugin.Score(context.Background(), types.NewCycleState(), req2, pods)
	state, err = plugins.ReadPluginStateKey[*SchedulingContextState](plugin.pluginState, req2.RequestId, plugins.StateKey(plugin.TypedName().String()))
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 6, hash block size is 4, the last 2 characters are ignored.
	// Total hashes = 1 (the first one is for the prefix with model)
	assert.Equal(t, 1, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 0, len(state.PrefixCacheServers), "there shouldn't be any cached servers")
	assert.Equal(t, float64(0), scores[pod1], "score for pod1")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	// Simulate pod2 was picked.
	schedulingResult = &types.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*types.ProfileRunResult{
			"default": {TargetPods: []types.Pod{pod2}},
		},
	}
	plugin.PreRequest(context.Background(), req2, schedulingResult)
	plugin.wg.Wait()

	// Third request shares partial prefix with first one.
	req3 := &types.LLMRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: "aaaabbbb",
			},
		},
	}
	scores = plugin.Score(context.Background(), types.NewCycleState(), req3, pods)
	state, err = plugins.ReadPluginStateKey[*SchedulingContextState](plugin.pluginState, req3.RequestId, plugins.StateKey(plugin.TypedName().String()))
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 8, hash block size is 4, so 2 hashes will be calculated.
	// Total hashes = 2 (the first one is for the prefix with model)
	assert.Equal(t, 2, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 1, len(state.PrefixCacheServers), "pod1 should have cached the aaaa prefix")
	assert.Equal(t, 0.5, scores[pod1], "score should be 0.5 - the model and the first prefix block match")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	schedulingResult = &types.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*types.ProfileRunResult{
			"default": {TargetPods: []types.Pod{pod1}},
		},
	}
	plugin.PreRequest(context.Background(), req3, schedulingResult)
	plugin.wg.Wait()

	// 4th request is same as req3 except the model is different, still no match.
	req4 := &types.LLMRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model-new",
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: "aaaabbbb",
			},
		},
	}
	scores = plugin.Score(context.Background(), types.NewCycleState(), req4, pods)
	state, err = plugins.ReadPluginStateKey[*SchedulingContextState](plugin.pluginState, req4.RequestId, plugins.StateKey(plugin.TypedName().String()))
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 8, hash block size is 4, so 2 hashes will be calculated.
	// Total hashes = 2 (the first one is for the prefix with model)
	assert.Equal(t, 2, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 0, len(state.PrefixCacheServers), "pod1 should have cached the aaaa prefix")
	assert.Equal(t, float64(0), scores[pod1], "score for pod1")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	schedulingResult = &types.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*types.ProfileRunResult{
			"default": {TargetPods: []types.Pod{pod1}},
		},
	}
	plugin.PreRequest(context.Background(), req4, schedulingResult)
	plugin.wg.Wait()

	// 5th request shares partial prefix with 3rd one.
	req5 := &types.LLMRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: "aaaabbbbcccc",
			},
		},
	}
	scores = plugin.Score(context.Background(), types.NewCycleState(), req5, pods)
	state, err = plugins.ReadPluginStateKey[*SchedulingContextState](plugin.pluginState, req5.RequestId, plugins.StateKey(plugin.TypedName().String()))
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 12, hash block size is 4, so 3 hashes will be calculated.
	// Total hashes = 3 (the first one is for the prefix with model)
	assert.Equal(t, 3, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 1, len(state.PrefixCacheServers), "pod1 should have cached the aaaa prefix")
	assert.Equal(t, 2./3, scores[pod1], "score should be 2./3 - the model and the first 2 prefix blocks match")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	schedulingResult = &types.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*types.ProfileRunResult{
			"default": {TargetPods: []types.Pod{pod1}},
		},
	}
	plugin.PreRequest(context.Background(), req5, schedulingResult)
	plugin.wg.Wait()
}

func TestPrefixPluginChatCompletions(t *testing.T) {
	config := Config{
		BlockSize:              4,
		MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}
	plugin := New(context.Background(), config)

	pod1 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, MetricsState: &backendmetrics.MetricsState{}}
	pods := []types.Pod{pod1}

	// Test with chat completions request
	req1 := &types.LLMRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &types.LLMRequestBody{
			ChatCompletions: &types.ChatCompletionsRequest{
				Messages: []types.Message{
					{Role: "user", Content: types.Content{Raw: "hello world"}},
					{Role: "assistant", Content: types.Content{Raw: "hi there"}},
				},
			},
		},
	}
	scores := plugin.Score(context.Background(), types.NewCycleState(), req1, pods)
	state, err := plugins.ReadPluginStateKey[*SchedulingContextState](plugin.pluginState, req1.RequestId, plugins.StateKey(plugin.TypedName().String()))
	assert.NoError(t, err)
	t.Logf("Chat completions - Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Should have some hashes for the JSON-encoded messages
	assert.Greater(t, len(state.PrefixHashes), 1, "should have hashes for chat completions")
	assert.Equal(t, 0, len(state.PrefixCacheServers), "there shouldn't be any cached servers initially")
	assert.Equal(t, float64(0), scores[pod1], "score for pod1")
}

func TestPrefixPluginChatCompletionsGrowth(t *testing.T) {
	config := Config{
		BlockSize:              8, // Use larger block size for more predictable JSON marshaling
		MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}
	plugin := New(context.Background(), config)

	pod1 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, MetricsState: &backendmetrics.MetricsState{}}
	pod2 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, MetricsState: &backendmetrics.MetricsState{}}
	pods := []types.Pod{pod1, pod2}

	// First request with initial conversation
	req1 := &types.LLMRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &types.LLMRequestBody{
			ChatCompletions: &types.ChatCompletionsRequest{
				Messages: []types.Message{
					{Role: "system", Content: types.Content{Raw: "You are a helpful assistant"}},
					{Role: "user", Content: types.Content{Raw: "Hello, how are you?"}},
				},
			},
		},
	}
	scores := plugin.Score(context.Background(), types.NewCycleState(), req1, pods)
	state, err := plugins.ReadPluginStateKey[*SchedulingContextState](plugin.pluginState, req1.RequestId, plugins.StateKey(plugin.TypedName().String()))
	assert.NoError(t, err)
	t.Logf("Initial conversation - Hashes %+v, cached servers: %+v", len(state.PrefixHashes), state.PrefixCacheServers)
	initialHashCount := len(state.PrefixHashes)
	assert.Greater(t, initialHashCount, 1, "should have hashes for chat completions")
	assert.Equal(t, 0, len(state.PrefixCacheServers), "there shouldn't be any cached servers initially")
	assert.Equal(t, float64(0), scores[pod1], "score for pod1")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	// Simulate pod1 was picked
	schedulingResult := &types.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*types.ProfileRunResult{
			"default": {TargetPods: []types.Pod{pod1}},
		},
	}
	plugin.PreRequest(context.Background(), req1, schedulingResult)
	plugin.wg.Wait()

	// Second request adds assistant response and new user message (conversation grows)
	req2 := &types.LLMRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &types.LLMRequestBody{
			ChatCompletions: &types.ChatCompletionsRequest{
				Messages: []types.Message{
					{Role: "system", Content: types.Content{Raw: "You are a helpful assistant"}},
					{Role: "user", Content: types.Content{Raw: "Hello, how are you?"}},
					{Role: "assistant", Content: types.Content{Raw: "I'm doing well, thank you! How can I help you today?"}},
					{Role: "user", Content: types.Content{Raw: "Can you explain how prefix caching works?"}},
				},
			},
		},
	}
	scores = plugin.Score(context.Background(), types.NewCycleState(), req2, pods)
	state, err = plugins.ReadPluginStateKey[*SchedulingContextState](plugin.pluginState, req2.RequestId, plugins.StateKey(plugin.TypedName().String()))
	assert.NoError(t, err)
	t.Logf("Extended conversation - Hashes %+v, cached servers: %+v", len(state.PrefixHashes), state.PrefixCacheServers)
	extendedHashCount := len(state.PrefixHashes)
	assert.Greater(t, extendedHashCount, initialHashCount, "extended conversation should have more hashes")
	assert.Greater(t, len(state.PrefixCacheServers), 0, "should have cached servers from prefix match")

	// Calculate expected score - pod1 should have cached the initial prefix
	cachedBlocks := state.PrefixCacheServers[ServerID(pod1.GetPod().NamespacedName)]
	expectedScore := float64(cachedBlocks) / float64(extendedHashCount)
	assert.Equal(t, expectedScore, scores[pod1], "pod1 should have prefix cache hit")
	assert.Equal(t, float64(0), scores[pod2], "pod2 should have no cache hit")

	// Simulate pod1 was picked again
	plugin.PreRequest(context.Background(), req2, schedulingResult)
	plugin.wg.Wait()

	// Third request continues the conversation even further
	req3 := &types.LLMRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &types.LLMRequestBody{
			ChatCompletions: &types.ChatCompletionsRequest{
				Messages: []types.Message{
					{Role: "system", Content: types.Content{Raw: "You are a helpful assistant"}},
					{Role: "user", Content: types.Content{Raw: "Hello, how are you?"}},
					{Role: "assistant", Content: types.Content{Raw: "I'm doing well, thank you! How can I help you today?"}},
					{Role: "user", Content: types.Content{Raw: "Can you explain how prefix caching works?"}},
					{Role: "assistant", Content: types.Content{Raw: "Prefix caching is a technique where..."}},
					{Role: "user", Content: types.Content{Raw: "That's very helpful, thank you!"}},
				},
			},
		},
	}
	scores = plugin.Score(context.Background(), types.NewCycleState(), req3, pods)
	state, err = plugins.ReadPluginStateKey[*SchedulingContextState](plugin.pluginState, req3.RequestId, plugins.StateKey(plugin.TypedName().String()))
	assert.NoError(t, err)
	t.Logf("Long conversation - Hashes %+v, cached servers: %+v", len(state.PrefixHashes), state.PrefixCacheServers)
	longHashCount := len(state.PrefixHashes)
	assert.Greater(t, longHashCount, extendedHashCount, "long conversation should have even more hashes")
	assert.Greater(t, len(state.PrefixCacheServers), 0, "should have cached servers from prefix match")

	// pod1 should have an even higher cache hit rate now
	cachedBlocks = state.PrefixCacheServers[ServerID(pod1.GetPod().NamespacedName)]
	expectedScore = float64(cachedBlocks) / float64(longHashCount)
	assert.Equal(t, expectedScore, scores[pod1], "pod1 should have higher prefix cache hit")
	assert.Greater(t, scores[pod1], float64(0.5), "cache hit rate should be substantial for growing conversation")
	assert.Equal(t, float64(0), scores[pod2], "pod2 should still have no cache hit")
}

// TestPrefixPluginStress is a stress test for the prefix scoring plugin, using prompts of increasing length.
func BenchmarkPrefixPluginStress(b *testing.B) {
	blockSize := 4
	maxPrefixBlocks := 50000
	config := Config{
		BlockSize:              blockSize,
		MaxPrefixBlocksToMatch: maxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}

	plugin := New(context.Background(), config)
	types.NewCycleState()
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
			pod := &types.PodMetrics{
				Pod: &backend.Pod{
					NamespacedName: k8stypes.NamespacedName{
						Name: fmt.Sprintf("random-pod-%d", v),
					},
				},
			}

			pods := []types.Pod{pod}
			req := &types.LLMRequest{
				RequestId:   uuid.NewString(),
				TargetModel: "model-stress",
				Body: &types.LLMRequestBody{
					Completions: &types.CompletionsRequest{
						Prompt: prompt,
					},
				},
			}

			b.ResetTimer()
			// Benchmark the scoring operation
			scores := plugin.Score(context.Background(), types.NewCycleState(), req, pods)
			_ = scores // Use the result to prevent optimization

			// Clean up state for next iteration
			plugin.pluginState.Delete(req.RequestId)
		})

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
				BlockSize:              0,
				MaxPrefixBlocksToMatch: 0,
				LRUCapacityPerServer:   0,
			},
			expectBlock:    DefaultBlockSize,
			expectMaxMatch: DefaultMaxPrefixBlocks,
			expectCapacity: DefaultLRUCapacityPerServer,
		},
		{
			name: "negative values",
			config: Config{
				BlockSize:              -5,
				MaxPrefixBlocksToMatch: -10,
				LRUCapacityPerServer:   -100,
			},
			expectBlock:    DefaultBlockSize,
			expectMaxMatch: DefaultMaxPrefixBlocks,
			expectCapacity: DefaultLRUCapacityPerServer,
		},
		{
			name: "mixed valid and invalid",
			config: Config{
				BlockSize:              32,    // valid
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
				BlockSize:              64,
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

			plugin := New(context.Background(), tt.config)

			assert.NotEmpty(t, plugin)
			assert.NotEmpty(t, plugin.indexer)
			assert.Equal(t, tt.expectBlock, plugin.config.BlockSize)
			assert.Equal(t, tt.expectMaxMatch, plugin.config.MaxPrefixBlocksToMatch)
			assert.Equal(t, tt.expectCapacity, plugin.config.LRUCapacityPerServer)
		})
	}
}

func TestPrefixPluginAutoTune(t *testing.T) {
	// Setup common test data
	podName := "pod-autotune"
	pod := &types.PodMetrics{
		Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: podName}},
		MetricsState: &backendmetrics.MetricsState{
			CacheBlockSize:    16,   // 16 tokens * 4 chars/token = 64 chars per block
			CacheNumGPUBlocks: 1000, // 1000 blocks capacity
		},
	}
	pods := []types.Pod{pod}

	req := &types.LLMRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
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
			BlockSize:              32, // Should be ignored in favor of pod metrics (64)
			MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
			// Should be ignored in favor of pod metrics (1000)
			LRUCapacityPerServer: 1,
		}
		plugin := New(context.Background(), config)

		// 1. Verify Score uses pod metrics for block size
		scores := plugin.Score(context.Background(), types.NewCycleState(), req, pods)
		_ = scores

		state, err := plugins.ReadPluginStateKey[*SchedulingContextState](plugin.pluginState, req.RequestId, plugins.StateKey(plugin.TypedName().String()))
		assert.NoError(t, err)
		// Block size from pod is 16 tokens * 4 = 64 chars.
		// Prompt is 128 chars.
		// Expected blocks: 128/64 = 2 hashes (model hash is used as seed but not returned as a block)
		assert.Equal(t, 2, len(state.PrefixHashes), "Should use pod block size (64 chars) -> 2 body blocks")

		// 2. Verify PreRequest uses pod metrics for capacity
		schedulingResult := &types.SchedulingResult{
			PrimaryProfileName: "default",
			ProfileResults: map[string]*types.ProfileRunResult{
				"default": {TargetPods: []types.Pod{pod}},
			},
		}
		plugin.PreRequest(context.Background(), req, schedulingResult)
		plugin.wg.Wait()

		// Check indexer state
		assert.Contains(t, plugin.indexer.Pods(), ServerID(pod.GetPod().NamespacedName))
	})

	t.Run("AutoTune Disabled", func(t *testing.T) {
		config := Config{
			AutoTune:               false,
			BlockSize:              32, // Should be used (32 chars)
			MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
			LRUCapacityPerServer:   1, // Should be used, and the first hash should be evicted due to the small
		}
		plugin := New(context.Background(), config)

		// 1. Verify Score uses config BlockSize
		req.RequestId = uuid.NewString() // New request ID
		scores := plugin.Score(context.Background(), types.NewCycleState(), req, pods)
		_ = scores

		state, err := plugins.ReadPluginStateKey[*SchedulingContextState](plugin.pluginState, req.RequestId, plugins.StateKey(plugin.TypedName().String()))
		assert.NoError(t, err)
		// Block size from config is 32 chars.
		// Prompt is 128 chars.
		// 128 / 32 = 4 chunks.
		assert.Equal(t, 4, len(state.PrefixHashes), "Should use config block size (32 chars) -> 4 body blocks")

		// 2. Verify PreRequest uses config LRUCapacityPerServer
		schedulingResult := &types.SchedulingResult{
			PrimaryProfileName: "default",
			ProfileResults: map[string]*types.ProfileRunResult{
				"default": {TargetPods: []types.Pod{pod}},
			},
		}
		plugin.PreRequest(context.Background(), req, schedulingResult)
		plugin.wg.Wait()

		assert.Contains(t, plugin.indexer.Pods(), ServerID(pod.GetPod().NamespacedName))
	})
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

func TestPrepareRequestData(t *testing.T) {
	config := Config{
		BlockSize:              4,
		MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}
	plugin := New(context.Background(), config)

	pod1 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, MetricsState: backendmetrics.NewMetricsState(), AttributeMap: datalayer.NewAttributes()}
	pod2 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, MetricsState: backendmetrics.NewMetricsState(), AttributeMap: datalayer.NewAttributes()}
	pods := []types.Pod{pod1, pod2}

	// First request to populate cache.
	req1 := &types.LLMRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: "aaaabbbb",
			},
		},
	}
	_ = plugin.Score(context.Background(), types.NewCycleState(), req1, pods)
	schedulingResult := &types.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*types.ProfileRunResult{
			"default": {TargetPods: []types.Pod{pod1}},
		},
	}
	plugin.PreRequest(context.Background(), req1, schedulingResult)
	plugin.wg.Wait()

	// Second request that shares a prefix.
	req2 := &types.LLMRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: "aaaacccc",
			},
		},
	}

	err := plugin.PrepareRequestData(context.Background(), req2, pods)
	assert.NoError(t, err)

	// Verify pod1 has the correct prefix match info
	info1, ok := pod1.Get(dplugins.PrefixCacheMatchInfoKey)
	assert.True(t, ok)
	prefixInfo1 := info1.(*dplugins.PrefixCacheMatchInfo)
	assert.Equal(t, 1, prefixInfo1.MatchLength()) // "aaaa" matches
	assert.Equal(t, 2, prefixInfo1.TotalLength()) // "aaaacccc" -> 2 blocks

	// Verify pod2 has no match info
	info2, ok := pod2.Get(dplugins.PrefixCacheMatchInfoKey)
	assert.True(t, ok)
	prefixInfo2 := info2.(*dplugins.PrefixCacheMatchInfo)
	assert.Equal(t, 0, prefixInfo2.MatchLength()) // No match for pod2
	assert.Equal(t, 2, prefixInfo2.TotalLength())
}

// BenchmarkPrefixPluginChatCompletionsStress is a stress test for chat completions with varying message counts and lengths
func BenchmarkPrefixPluginChatCompletionsStress(b *testing.B) {
	blockSize := 8
	maxPrefixBlocks := 50000
	config := Config{
		BlockSize:              blockSize,
		MaxPrefixBlocksToMatch: maxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}
	plugin := New(context.Background(), config)

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
			messages := make([]types.Message, scenario.messageCount)
			messages[0] = types.Message{Role: "system", Content: types.Content{Raw: "You are a helpful assistant."}}

			for i := 1; i < scenario.messageCount; i++ {
				role := "user"
				if i%2 == 0 {
					role = "assistant"
				}
				content := randomPrompt(scenario.messageLength)
				messages[i] = types.Message{Role: role, Content: types.Content{Raw: content}}
			}

			pod := &types.PodMetrics{
				Pod: &backend.Pod{
					NamespacedName: k8stypes.NamespacedName{
						Name: fmt.Sprintf("chat-pod-%d-%d", scenario.messageCount, scenario.messageLength),
					},
				},
			}
			pods := []types.Pod{pod}

			req := &types.LLMRequest{
				RequestId:   uuid.NewString(),
				TargetModel: "chat-model-stress",
				Body: &types.LLMRequestBody{
					ChatCompletions: &types.ChatCompletionsRequest{
						Messages: messages,
					},
				},
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Benchmark the scoring operation
				scores := plugin.Score(context.Background(), types.NewCycleState(), req, pods)
				_ = scores // Use the result to prevent optimization

				// Clean up state for next iteration
				plugin.pluginState.Delete(req.RequestId)
			}
		})
	}
}
