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
	ctx := types.NewSchedulingContext(context.Background(), req1, nil, pods)
	plugin.PreSchedule(ctx)
	state, err := plugin.getPrefixState(ctx.CycleState)
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 6, hash block size is 4, the last 2 characters are ignored.
	// Total hashes = 2 (the first one is for the model)
	assert.Equal(t, 2, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 0, len(state.PrefixCacheServers), "there shouldn't be any cached servers")

	// Updated to use the new Score method signature
	scores := plugin.Score(ctx, pods)
	assert.Equal(t, float64(0), scores[pod1], "score for pod1")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	// Simulate pod1 was picked.
	plugin.PostSchedule(ctx, &types.Result{TargetPod: pod1})

	// Second request doesn't share any prefix with first one. It should be added to the cache but
	// the pod score should be 0.
	req2 := &types.LLMRequest{
		TargetModel: "test-model2",
		Prompt:      "bbbbbb",
	}
	ctx = types.NewSchedulingContext(context.Background(), req2, nil, pods)
	plugin.PreSchedule(ctx)
	state, err = plugin.getPrefixState(ctx.CycleState)
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 6, hash block size is 4, the last 2 characters are ignored.
	// Total hashes = 2 (the first one is for the model)
	assert.Equal(t, 2, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 0, len(state.PrefixCacheServers), "there shouldn't be any cached servers")

	// Updated to use the new Score method signature
	scores = plugin.Score(ctx, pods)
	assert.Equal(t, float64(0), scores[pod1], "score for pod1")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	// Simulate pod2 was picked.
	plugin.PostSchedule(ctx, &types.Result{TargetPod: pod2})

	// Third request shares partial prefix with first one.
	req3 := &types.LLMRequest{
		TargetModel: "test-model1",
		Prompt:      "aaaabbbb",
	}
	ctx = types.NewSchedulingContext(context.Background(), req3, nil, pods)
	plugin.PreSchedule(ctx)
	state, err = plugin.getPrefixState(ctx.CycleState)
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 8, hash block size is 4, so 2 hashes will be calculated.
	// Total hashes = 3 (the first one is for the model)
	assert.Equal(t, 3, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 1, len(state.PrefixCacheServers), "pod1 should have cached the aaaa prefix")

	// Updated to use the new Score method signature
	scores = plugin.Score(ctx, pods)
	assert.Equal(t, float64(2)/float64(3), scores[pod1], "score should be 2/3 - the model and the first prefix block match")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	plugin.PostSchedule(ctx, &types.Result{TargetPod: pod1})

	// 4th request is same as req3 except the model is different, still no match.
	req4 := &types.LLMRequest{
		TargetModel: "test-model-new",
		Prompt:      "aaaabbbb",
	}
	ctx = types.NewSchedulingContext(context.Background(), req4, nil, pods)
	plugin.PreSchedule(ctx)
	state, err = plugin.getPrefixState(ctx.CycleState)
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 8, hash block size is 4, so 2 hashes will be calculated.
	// Total hashes = 3 (the first one is for the model)
	assert.Equal(t, 3, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 0, len(state.PrefixCacheServers), "pod1 should have cached the aaaa prefix")

	// Updated to use the new Score method signature
	scores = plugin.Score(ctx, pods)
	assert.Equal(t, float64(0), scores[pod1], "score for pod1")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	plugin.PostSchedule(ctx, &types.Result{TargetPod: pod1})

	// 5th request shares partial prefix with 3rd one.
	req5 := &types.LLMRequest{
		TargetModel: "test-model1",
		Prompt:      "aaaabbbbcccc",
	}
	ctx = types.NewSchedulingContext(context.Background(), req5, nil, pods)
	plugin.PreSchedule(ctx)
	state, err = plugin.getPrefixState(ctx.CycleState)
	assert.NoError(t, err)
	t.Logf("Hashes %+v, cached servers: %+v", state.PrefixHashes, state.PrefixCacheServers)
	// Input size is 12, hash block size is 4, so 3 hashes will be calculated.
	// Total hashes = 4 (the first one is for the model)
	assert.Equal(t, 4, len(state.PrefixHashes), "number of hashes is incorrect")
	assert.Equal(t, 1, len(state.PrefixCacheServers), "pod1 should have cached the aaaa prefix")

	// Updated to use the new Score method signature
	scores = plugin.Score(ctx, pods)
	assert.Equal(t, 0.75, scores[pod1], "score should be 0.75 - the model and the first 2 prefix blocks match")
	assert.Equal(t, float64(0), scores[pod2], "score for pod2")

	plugin.PostSchedule(ctx, &types.Result{TargetPod: pod1})
}
