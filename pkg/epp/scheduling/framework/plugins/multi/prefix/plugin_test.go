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
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

func TestPrefixPlugin(t *testing.T) {

	config := Config{
		HashBlockSize:          4,
		MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}
	plugin := New(config)

	pod1 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}}
	pod2 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}}
	pods := []types.Pod{pod1, pod2}

	// First request.
	req1 := &types.LLMRequest{
		TargetModel: "test-model1",
		Prompt:      "aaaaaa",
	}
	cycleState1 := types.NewCycleState()
	scores := plugin.Score(context.Background(), cycleState1, req1, pods)
	state, err := types.ReadCycleStateKey[*SchedulingContextState](cycleState1, PrefixCachePluginType)
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 6, hash block size is 4, the last 2 characters are ignored.
	// Total hashes = 2 (the first one is for the model)
	assert.Equal(t, 2, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 0, len(state.PrefixCacheServers), "there shouldn't be any cached servers")
	assert.Equal(t, float64(0), scores[pod1], "score for pod1")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	// Simulate pod1 was picked - use PostResponse instead
	response1 := &requestcontrol.Response{
		RequestId: "req1",
		Headers:   map[string]string{"content-type": "application/json"},
	}
	plugin.PostResponse(context.Background(), req1, response1, pod1.Pod)

	// Second request doesn't share any prefix with first one. It should be added to the cache but
	// the pod score should be 0.
	req2 := &types.LLMRequest{
		TargetModel: "test-model2",
		Prompt:      "bbbbbb",
	}
	cycleState2 := types.NewCycleState()
	scores = plugin.Score(context.Background(), cycleState2, req2, pods)
	state, err = types.ReadCycleStateKey[*SchedulingContextState](cycleState2, PrefixCachePluginType)
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 6, hash block size is 4, the last 2 characters are ignored.
	// Total hashes = 2 (the first one is for the model)
	assert.Equal(t, 2, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 0, len(state.PrefixCacheServers), "there shouldn't be any cached servers")
	assert.Equal(t, float64(0), scores[pod1], "score for pod1")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	// Simulate pod2 was picked - use PostResponse instead
	response2 := &requestcontrol.Response{
		RequestId: "req2",
		Headers:   map[string]string{"content-type": "application/json"},
	}
	plugin.PostResponse(context.Background(), req2, response2, pod2.Pod)

	// Third request shares partial prefix with first one.
	req3 := &types.LLMRequest{
		TargetModel: "test-model1",
		Prompt:      "aaaabbbb",
	}
	cycleState3 := types.NewCycleState()
	scores = plugin.Score(context.Background(), cycleState3, req3, pods)
	state, err = types.ReadCycleStateKey[*SchedulingContextState](cycleState3, PrefixCachePluginType)
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 8, hash block size is 4, so 2 hashes will be calculated.
	// Total hashes = 3 (the first one is for the model)
	assert.Equal(t, 3, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 1, len(state.PrefixCacheServers), "pod1 should have cached the aaaa prefix")
	assert.Equal(t, float64(2)/float64(3), scores[pod1], "score should be 2/3 - the model and the first prefix block match")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	// Simulate pod1 was picked - use PostResponse instead
	response3 := &requestcontrol.Response{
		RequestId: "req3",
		Headers:   map[string]string{"content-type": "application/json"},
	}
	plugin.PostResponse(context.Background(), req3, response3, pod1.Pod)

	// 4th request is same as req3 except the model is different, still no match.
	req4 := &types.LLMRequest{
		TargetModel: "test-model-new",
		Prompt:      "aaaabbbb",
	}
	cycleState4 := types.NewCycleState()
	scores = plugin.Score(context.Background(), cycleState4, req4, pods)
	state, err = types.ReadCycleStateKey[*SchedulingContextState](cycleState4, PrefixCachePluginType)
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 8, hash block size is 4, so 2 hashes will be calculated.
	// Total hashes = 3 (the first one is for the model)
	assert.Equal(t, 3, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 0, len(state.PrefixCacheServers), "pod1 should have cached the aaaa prefix")
	assert.Equal(t, float64(0), scores[pod1], "score for pod1")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	// Simulate pod1 was picked - use PostResponse instead
	response4 := &requestcontrol.Response{
		RequestId: "req4",
		Headers:   map[string]string{"content-type": "application/json"},
	}
	plugin.PostResponse(context.Background(), req4, response4, pod1.Pod)

	// 5th request shares partial prefix with 3rd one.
	req5 := &types.LLMRequest{
		TargetModel: "test-model1",
		Prompt:      "aaaabbbbcccc",
	}
	cycleState5 := types.NewCycleState()
	scores = plugin.Score(context.Background(), cycleState5, req5, pods)
	state, err = types.ReadCycleStateKey[*SchedulingContextState](cycleState5, PrefixCachePluginType)
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 12, hash block size is 4, so 3 hashes will be calculated.
	// Total hashes = 4 (the first one is for the model)
	assert.Equal(t, 4, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 1, len(state.PrefixCacheServers), "pod1 should have cached the aaaa prefix")
	assert.Equal(t, 0.75, scores[pod1], "score should be 0.75 - the model and the first 2 prefix blocks match")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	// Simulate pod1 was picked - use PostResponse instead
	response5 := &requestcontrol.Response{
		RequestId: "req5",
		Headers:   map[string]string{"content-type": "application/json"},
	}
	plugin.PostResponse(context.Background(), req5, response5, pod1.Pod)
}

// TestPrefixPluginPostResponse tests the PostResponse method functionality
func TestPrefixPluginPostResponse(t *testing.T) {
	config := Config{
		HashBlockSize:          4,
		MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}
	plugin := New(config)

	pod1 := &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}
	pod2 := &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}

	// First request - use PostResponse to cache prefix
	req1 := &types.LLMRequest{
		TargetModel: "test-model1",
		Prompt:      "aaaaaa",
	}
	response1 := &requestcontrol.Response{
		RequestId: "req1",
		Headers:   map[string]string{"content-type": "application/json"},
	}

	// Call PostResponse to cache the prefix for pod1
	plugin.PostResponse(context.Background(), req1, response1, pod1)

	// Second request with same prefix - should get cache hit on pod1
	req2 := &types.LLMRequest{
		TargetModel: "test-model1",
		Prompt:      "aaaabbbb",
	}

	// Test scoring to verify cache hit
	pods := []types.Pod{
		&types.PodMetrics{Pod: pod1},
		&types.PodMetrics{Pod: pod2},
	}
	cycleState := types.NewCycleState()
	scores := plugin.Score(context.Background(), cycleState, req2, pods)

	// pod1 should have a higher score due to prefix cache hit
	assert.Greater(t, scores[pods[0]], scores[pods[1]], "pod1 should have higher score due to prefix cache")
	assert.Greater(t, scores[pods[0]], float64(0), "pod1 should have non-zero score")
	assert.Equal(t, float64(0), scores[pods[1]], "pod2 should have zero score")

	// Use PostResponse for the second request on pod1
	response2 := &requestcontrol.Response{
		RequestId: "req2",
		Headers:   map[string]string{"content-type": "application/json"},
	}
	plugin.PostResponse(context.Background(), req2, response2, pod1)

	// Third request with different model - should not get cache hit
	req3 := &types.LLMRequest{
		TargetModel: "test-model2", // Different model
		Prompt:      "aaaaaa",
	}
	cycleState3 := types.NewCycleState()
	scores3 := plugin.Score(context.Background(), cycleState3, req3, pods)

	// Both pods should have zero score since model is different
	assert.Equal(t, float64(0), scores3[pods[0]], "pod1 should have zero score for different model")
	assert.Equal(t, float64(0), scores3[pods[1]], "pod2 should have zero score for different model")
}

// TestPrefixPluginStress is a stress test for the prefix scoring plugin, using prompts of increasing length.
func BenchmarkPrefixPluginStress(b *testing.B) {
	blockSize := 4
	maxPrefixBlocks := 50000
	config := Config{
		HashBlockSize:          blockSize,
		MaxPrefixBlocksToMatch: maxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}

	plugin := New(config)
	types.NewCycleState()
	var promptLen []int
	for i := 1; i <= 1024; i++ {
		promptLen = append(promptLen, i)
	}
	promptLen = append(promptLen, 2048, 4096, 8192, 10000, 20000, 50000)

	for _, i := range promptLen {
		// Generate increasing-length random prompts
		prompt := randomPrompt(4 + i)
		pod := &types.PodMetrics{
			Pod: &backend.Pod{
				NamespacedName: k8stypes.NamespacedName{
					Name: fmt.Sprintf("random-pod-%d", i),
				},
			},
		}

		pods := []types.Pod{pod}
		req := &types.LLMRequest{
			TargetModel: "model-stress",
			Prompt:      prompt,
		}

		// First cycle: simulate scheduling and insert prefix info into the cache
		cycleState := types.NewCycleState()
		plugin.Score(context.Background(), cycleState, req, pods)
		// Use PostResponse instead of PostCycle
		response := &requestcontrol.Response{
			RequestId: fmt.Sprintf("req-%d", i),
			Headers:   map[string]string{"content-type": "application/json"},
		}
		plugin.PostResponse(context.Background(), req, response, pod.Pod)

		// Second cycle: validate internal state
		state, err := types.ReadCycleStateKey[*SchedulingContextState](cycleState, PrefixCachePluginType)
		assert.NoError(b, err)
		expectedHashes := int(math.Min(float64(maxPrefixBlocks+1), float64(len(req.Prompt)/blockSize+1))) // the extra one is for the model.
		assert.Equal(b, expectedHashes, len(state.PrefixHashes), "number of hashes is incorrect")
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
