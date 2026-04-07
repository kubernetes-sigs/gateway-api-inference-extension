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
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func TestPrepareRequestData(t *testing.T) {
	config := config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	// Test the "initialize if nil" pattern
	p, err := newPrepareData(context.Background(), config, nil)
	assert.NoError(t, err)
	assert.NotNil(t, p.PluginState())

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{endpoint1, endpoint2}

	// First request to populate cache.
	req1 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwksched.InferenceRequestBody{
			Completions: &fwksched.CompletionsRequest{
				Prompt: fwksched.Prompt{Raw: "aaaabbbb"},
			},
		},
	}

	// We need to simulate the PreRequest logic since PrepareRequestData only reads from the indexer.
	// But first let's see if PrepareRequestData correctly handles an empty indexer.
	err = p.PrepareRequestData(context.Background(), req1, endpoints)
	assert.NoError(t, err)

	// Verify state was written to PluginState
	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req1.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	assert.NoError(t, err)
	assert.NotNil(t, state)
	assert.Equal(t, 2, len(state.PrefixHashes)) // "aaaabbbb" with blockSize 4 (1 token * 4 chars) -> 2 blocks

	// Verify pod match info was set (should be 0 match since indexer is empty)
	for _, ep := range endpoints {
		info, ok := ep.Get(attrprefix.PrefixCacheMatchInfoKey)
		assert.True(t, ok)
		prefixInfo := info.(*attrprefix.PrefixCacheMatchInfo)
		assert.Equal(t, 0, prefixInfo.MatchBlocks())
		assert.Equal(t, 2, prefixInfo.TotalBlocks())
	}
}

func TestPreRequest(t *testing.T) {
	config := config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, _ := newPrepareData(context.Background(), config, nil)

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1", Namespace: "default"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	req1 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwksched.InferenceRequestBody{
			Completions: &fwksched.CompletionsRequest{
				Prompt: fwksched.Prompt{Raw: "aaaabbbb"},
			},
		},
	}

	// 1. Prepare data (this saves state)
	_ = p.PrepareRequestData(context.Background(), req1, []fwksched.Endpoint{endpoint1})

	// 2. Simulate scheduling result
	res := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {
				TargetEndpoints: []fwksched.Endpoint{endpoint1},
			},
		},
	}

	// 3. Call PreRequest
	p.PreRequest(context.Background(), req1, res)

	// Wait for async update
	p.wg.Wait()

	// 4. Verify indexer was updated
	hashes := hashPrompt(context.Background(), req1, 4, defaultMaxPrefixBlocks)
	for _, hash := range hashes {
		pods := p.indexer().Get(hash)
		assert.Contains(t, pods, ServerID(endpoint1.GetMetadata().NamespacedName))
	}
}

func TestPrepareDataValidation(t *testing.T) {
	validConfigs := []config{{
		AutoTune:        false,
		BlockSizeTokens: 1,
	}, {
		AutoTune:        false,
		BlockSize:       1,
		BlockSizeTokens: 1,
	}, {
		AutoTune:        true,
		BlockSizeTokens: 0,
	}}
	invalidConfigs := []config{{
		AutoTune:  false,
		BlockSize: 1,
	}, {
		AutoTune:        false,
		BlockSizeTokens: 0,
	}}

	for _, config := range validConfigs {
		_, err := newPrepareData(context.Background(), config, nil)
		assert.NoError(t, err)
	}

	for _, config := range invalidConfigs {
		_, err := newPrepareData(context.Background(), config, nil)
		assert.Error(t, err)
	}
}

func TestPrefixPluginCompletion(t *testing.T) {
	config := config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, _ := newPrepareData(context.Background(), config, nil)

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoint3 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{endpoint1, endpoint2, endpoint3}

	// First request.
	req1 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwksched.InferenceRequestBody{
			Completions: &fwksched.CompletionsRequest{
				Prompt: fwksched.Prompt{Raw: "aaaaaa"},
			},
		},
	}
	_ = p.PrepareRequestData(context.Background(), req1, endpoints)
	state, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req1.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	// Input size is 6, block size is 4, so 1 body block. Total hashes = 1 (model only is not a block)
	assert.Equal(t, 1, len(state.PrefixHashes))

	// Simulate pod1 was picked and pod3 was picked as a prefill node.
	schedulingResult := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default":                         {TargetEndpoints: []fwksched.Endpoint{endpoint1}},
			experimentalDefaultPrefillProfile: {TargetEndpoints: []fwksched.Endpoint{endpoint3}},
		},
	}
	p.PreRequest(context.Background(), req1, schedulingResult)
	p.wg.Wait()

	// Third request shares partial prefix with first one.
	req3 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwksched.InferenceRequestBody{
			Completions: &fwksched.CompletionsRequest{
				Prompt: fwksched.Prompt{Raw: "aaaabbbb"},
			},
		},
	}
	_ = p.PrepareRequestData(context.Background(), req3, endpoints)

	// Verify pod1 has the correct prefix match info
	info1, _ := endpoint1.Get(attrprefix.PrefixCacheMatchInfoKey)
	prefixInfo1 := info1.(*attrprefix.PrefixCacheMatchInfo)
	assert.Equal(t, 1, prefixInfo1.MatchBlocks()) // one block ("aaaa") matches
	assert.Equal(t, 2, prefixInfo1.TotalBlocks()) // "aaaabbbb" -> 2 blocks

	// Verify pod3 (prefill node) also has the match
	info3, _ := endpoint3.Get(attrprefix.PrefixCacheMatchInfoKey)
	prefixInfo3 := info3.(*attrprefix.PrefixCacheMatchInfo)
	assert.Equal(t, 1, prefixInfo3.MatchBlocks())

	// Verify pod2 has no match info
	info2, _ := endpoint2.Get(attrprefix.PrefixCacheMatchInfoKey)
	prefixInfo2 := info2.(*attrprefix.PrefixCacheMatchInfo)
	assert.Equal(t, 0, prefixInfo2.MatchBlocks())
}

func TestPrefixPluginChatCompletionsGrowth(t *testing.T) {
	config := config{
		BlockSizeTokens:        2, // Use larger block size
		AutoTune:               false,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, _ := newPrepareData(context.Background(), config, nil)

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{endpoint1}

	// First request with initial conversation
	req1 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwksched.InferenceRequestBody{
			ChatCompletions: &fwksched.ChatCompletionsRequest{
				Messages: []fwksched.Message{
					{Role: "system", Content: fwksched.Content{Raw: "You are a helpful assistant"}},
					{Role: "user", Content: fwksched.Content{Raw: "Hello, how are you?"}},
				},
			},
		},
	}
	_ = p.PrepareRequestData(context.Background(), req1, endpoints)
	state1, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req1.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	initialHashCount := len(state1.PrefixHashes)
	assert.Greater(t, initialHashCount, 0)

	// Simulate pod1 was picked
	schedulingResult := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint1}},
		},
	}
	p.PreRequest(context.Background(), req1, schedulingResult)
	p.wg.Wait()

	// Second request adds assistant response and new user message
	req2 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwksched.InferenceRequestBody{
			ChatCompletions: &fwksched.ChatCompletionsRequest{
				Messages: []fwksched.Message{
					{Role: "system", Content: fwksched.Content{Raw: "You are a helpful assistant"}},
					{Role: "user", Content: fwksched.Content{Raw: "Hello, how are you?"}},
					{Role: "assistant", Content: fwksched.Content{Raw: "I'm doing well, thank you! How can I help you today?"}},
					{Role: "user", Content: fwksched.Content{Raw: "Can you explain how prefix caching works?"}},
				},
			},
		},
	}
	_ = p.PrepareRequestData(context.Background(), req2, endpoints)
	state2, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req2.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	extendedHashCount := len(state2.PrefixHashes)
	assert.Greater(t, extendedHashCount, initialHashCount)

	info, _ := endpoint1.Get(attrprefix.PrefixCacheMatchInfoKey)
	prefixInfo := info.(*attrprefix.PrefixCacheMatchInfo)
	assert.Greater(t, prefixInfo.MatchBlocks(), 0, "should have prefix cache hit")
	assert.Equal(t, extendedHashCount, prefixInfo.TotalBlocks())
}

func TestPrefixPluginAutoTune(t *testing.T) {
	podName := "pod-autotune"
	endpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: podName}},
		&fwkdl.Metrics{
			CacheBlockSize: 16,   // 16 tokens * 4 chars/token = 64 chars per block
			CacheNumBlocks: 1000, // 1000 blocks capacity
		}, fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{endpoint}

	req := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwksched.InferenceRequestBody{
			Completions: &fwksched.CompletionsRequest{
				// Length 128 chars.
				// If block size is 64 chars: 2 blocks
				Prompt: fwksched.Prompt{Raw: strings.Repeat("a", 128)},
			},
		},
	}

	config := config{
		AutoTune:               true,
		BlockSizeTokens:        32, // Should be ignored in favor of pod metrics (16)
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   1,
	}
	p, _ := newPrepareData(context.Background(), config, nil)

	_ = p.PrepareRequestData(context.Background(), req, endpoints)
	state, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	// 128 chars / (16 tokens * 4 chars/token) = 2 blocks
	assert.Equal(t, 2, len(state.PrefixHashes), "Should use pod block size (16 tokens) -> 2 body blocks")

	schedulingResult := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}
	p.PreRequest(context.Background(), req, schedulingResult)
	p.wg.Wait()

	// Check indexer state - should be in tracked pods
	assert.Contains(t, p.indexer().Pods(), ServerID(endpoint.GetMetadata().NamespacedName))
}

func TestMaxPrefixTokensToMatch(t *testing.T) {
	// BlockSizeTokens=1 means each block is 4 chars (1 token * 4 chars/token).
	// With MaxPrefixTokensToMatch=2, maxBlocks = 2/1 = 2, so only the first
	// 2 blocks (8 chars) of the prompt should be hashed.
	cfg := config{
		BlockSizeTokens:        1,
		MaxPrefixTokensToMatch: 2,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, err := newPrepareData(context.Background(), cfg, nil)
	assert.NoError(t, err)

	endpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes(),
	)

	// Prompt is 16 chars = 4 blocks at blockSize 4 chars, but should be capped to 2.
	req := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwksched.InferenceRequestBody{
			Completions: &fwksched.CompletionsRequest{
				Prompt: fwksched.Prompt{Raw: "aaaabbbbccccdddd"},
			},
		},
	}

	err = p.PrepareRequestData(context.Background(), req, []fwksched.Endpoint{endpoint})
	assert.NoError(t, err)

	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	assert.NoError(t, err)
	assert.Equal(t, 2, len(state.PrefixHashes), "should cap at MaxPrefixTokensToMatch/BlockSizeTokens = 2 blocks")

	// When MaxPrefixTokensToMatch is 0 (unset), fall back to MaxPrefixBlocksToMatch.
	cfg2 := config{
		BlockSizeTokens:        1,
		MaxPrefixTokensToMatch: 0,
		MaxPrefixBlocksToMatch: 3,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p2, err := newPrepareData(context.Background(), cfg2, nil)
	assert.NoError(t, err)

	req2 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwksched.InferenceRequestBody{
			Completions: &fwksched.CompletionsRequest{
				Prompt: fwksched.Prompt{Raw: "aaaabbbbccccdddd"},
			},
		},
	}

	err = p2.PrepareRequestData(context.Background(), req2, []fwksched.Endpoint{endpoint})
	assert.NoError(t, err)

	state2, err := plugin.ReadPluginStateKey[*SchedulingContextState](p2.PluginState(), req2.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	assert.NoError(t, err)
	assert.Equal(t, 3, len(state2.PrefixHashes), "should fall back to MaxPrefixBlocksToMatch when MaxPrefixTokensToMatch is 0")
}

// BenchmarkPrefixPluginStress is a stress test using prompts of increasing length.
func BenchmarkPrefixPluginStress(b *testing.B) {
	config := config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: 50000,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, _ := newPrepareData(context.Background(), config, nil)

	promptLen := []int{1024, 4096, 10000, 50000}

	for _, v := range promptLen {
		b.Run(fmt.Sprintf("length_%d", v), func(b *testing.B) {
			prompt := randomPrompt(v)
			endpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
				NamespacedName: k8stypes.NamespacedName{Name: "pod1"},
			}, nil, fwkdl.NewAttributes())
			endpoints := []fwksched.Endpoint{endpoint}
			req := &fwksched.InferenceRequest{
				RequestId:   uuid.NewString(),
				TargetModel: "model-stress",
				Body: &fwksched.InferenceRequestBody{
					Completions: &fwksched.CompletionsRequest{
						Prompt: fwksched.Prompt{Raw: prompt},
					},
				},
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = p.PrepareRequestData(context.Background(), req, endpoints)
				p.PluginState().Delete(req.RequestId)
			}
		})
	}
}

func randomPrompt(n int) string {
	runes := []rune("abcdefghijklmnopqrstuvwxyz")
	var sb strings.Builder
	for range n {
		sb.WriteRune(runes[rand.Intn(len(runes))])
	}
	return sb.String()
}
