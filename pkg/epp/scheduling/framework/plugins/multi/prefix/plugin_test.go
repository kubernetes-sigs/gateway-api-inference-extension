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
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

func TestPrefixPlugin(t *testing.T) {
	config := Config{
		HashBlockSize:          4,
		MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
		LRUIndexerCapacity:     DefaultLRUIndexerCapacity,
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
	scores := plugin.Score(context.Background(), req1, cycleState1, pods)
	state, err := plugin.getPrefixState(cycleState1)
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 6, hash block size is 4, the last 2 characters are ignored.
	// Total hashes = 2 (the first one is for the model)
	assert.Equal(t, 2, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 0, len(state.PrefixCacheServers), "there shouldn't be any cached servers")
	assert.Equal(t, float64(0), scores[pod1], "score for pod1")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	// Simulate pod1 was picked.
	plugin.PostCycle(context.Background(), cycleState1, &types.ProfileRunResult{TargetPod: pod1})

	// Second request doesn't share any prefix with first one. It should be added to the cache but
	// the pod score should be 0.
	req2 := &types.LLMRequest{
		TargetModel: "test-model2",
		Prompt:      "bbbbbb",
	}
	cycleState2 := types.NewCycleState()
	scores = plugin.Score(context.Background(), req2, cycleState2, pods)
	state, err = plugin.getPrefixState(cycleState2)
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 6, hash block size is 4, the last 2 characters are ignored.
	// Total hashes = 2 (the first one is for the model)
	assert.Equal(t, 2, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 0, len(state.PrefixCacheServers), "there shouldn't be any cached servers")
	assert.Equal(t, float64(0), scores[pod1], "score for pod1")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	// Simulate pod2 was picked.
	plugin.PostCycle(context.Background(), cycleState2, &types.ProfileRunResult{TargetPod: pod2})

	// Third request shares partial prefix with first one.
	req3 := &types.LLMRequest{
		TargetModel: "test-model1",
		Prompt:      "aaaabbbb",
	}
	cycleState3 := types.NewCycleState()
	scores = plugin.Score(context.Background(), req3, cycleState3, pods)
	state, err = plugin.getPrefixState(cycleState3)
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 8, hash block size is 4, so 2 hashes will be calculated.
	// Total hashes = 3 (the first one is for the model)
	assert.Equal(t, 3, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 1, len(state.PrefixCacheServers), "pod1 should have cached the aaaa prefix")
	assert.Equal(t, float64(2)/float64(3), scores[pod1], "score should be 2/3 - the model and the first prefix block match")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	plugin.PostCycle(context.Background(), cycleState3, &types.ProfileRunResult{TargetPod: pod1})

	// 4th request is same as req3 except the model is different, still no match.
	req4 := &types.LLMRequest{
		TargetModel: "test-model-new",
		Prompt:      "aaaabbbb",
	}
	cycleState4 := types.NewCycleState()
	scores = plugin.Score(context.Background(), req4, cycleState4, pods)
	state, err = plugin.getPrefixState(cycleState4)
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 8, hash block size is 4, so 2 hashes will be calculated.
	// Total hashes = 3 (the first one is for the model)
	assert.Equal(t, 3, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 0, len(state.PrefixCacheServers), "pod1 should have cached the aaaa prefix")
	assert.Equal(t, float64(0), scores[pod1], "score for pod1")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	plugin.PostCycle(context.Background(), cycleState4, &types.ProfileRunResult{TargetPod: pod1})

	// 5th request shares partial prefix with 3rd one.
	req5 := &types.LLMRequest{
		TargetModel: "test-model1",
		Prompt:      "aaaabbbbcccc",
	}
	cycleState5 := types.NewCycleState()
	scores = plugin.Score(context.Background(), req5, cycleState5, pods)
	state, err = plugin.getPrefixState(cycleState5)
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 12, hash block size is 4, so 3 hashes will be calculated.
	// Total hashes = 4 (the first one is for the model)
	assert.Equal(t, 4, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 1, len(state.PrefixCacheServers), "pod1 should have cached the aaaa prefix")
	assert.Equal(t, 0.75, scores[pod1], "score should be 0.75 - the model and the first 2 prefix blocks match")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	plugin.PostCycle(context.Background(), cycleState5, &types.ProfileRunResult{TargetPod: pod1})
}
