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

package vtc

import (
	"context"
	"encoding/json"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol/mocks"
)

func TestVTC_TypedName(t *testing.T) {
	t.Parallel()

	p, err := newVTC("my-vtc", nil)
	require.NoError(t, err)
	assert.Equal(t, VTCFairnessPolicyType, p.TypedName().Type)
	assert.Equal(t, "my-vtc", p.TypedName().Name)
}

func TestVTC_TypedName_Default(t *testing.T) {
	t.Parallel()

	p, err := newVTC("", nil)
	require.NoError(t, err)
	assert.Equal(t, VTCFairnessPolicyType, p.TypedName().Name)
}

func TestVTC_Factory(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		params    json.RawMessage
		expectErr bool
	}{
		{
			name:   "nil params uses defaults",
			params: nil,
		},
		{
			name:   "empty params uses defaults",
			params: json.RawMessage{},
		},
		{
			name:   "valid params",
			params: json.RawMessage(`{"weights": {"a": 2.0}, "defaultWeight": 0.5}`),
		},
		{
			name:   "zero defaultWeight gets corrected to 1.0",
			params: json.RawMessage(`{"defaultWeight": 0}`),
		},
		{
			name:      "invalid JSON",
			params:    json.RawMessage(`{invalid`),
			expectErr: true,
		},
		{
			name:      "negative weight",
			params:    json.RawMessage(`{"weights": {"a": -1.0}}`),
			expectErr: true,
		},
		{
			name:      "zero weight",
			params:    json.RawMessage(`{"weights": {"a": 0}}`),
			expectErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			plugin, err := VTCFairnessPolicyFactory("test", tc.params, nil)
			if tc.expectErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, plugin)
		})
	}
}

// newMockQueueWithCost creates a mock queue with a head item of the given byte size.
func newMockQueueWithCost(key flowcontrol.FlowKey, queueLen int, headByteSize uint64) *frameworkmocks.MockFlowQueueAccessor {
	var peekHead flowcontrol.QueueItemAccessor
	if queueLen > 0 {
		peekHead = frameworkmocks.NewMockQueueItemAccessor(headByteSize, "req-"+key.ID, key)
	}
	return &frameworkmocks.MockFlowQueueAccessor{
		LenV:      queueLen,
		PeekHeadV: peekHead,
		FlowKeyV:  key,
	}
}

// buildBand creates a MockPriorityBandAccessor from a list of queues and a policy state.
func buildBand(state any, queues ...*frameworkmocks.MockFlowQueueAccessor) *frameworkmocks.MockPriorityBandAccessor {
	queueMap := make(map[string]*frameworkmocks.MockFlowQueueAccessor, len(queues))
	keys := make([]flowcontrol.FlowKey, 0, len(queues))
	for _, q := range queues {
		queueMap[q.FlowKeyV.ID] = q
		keys = append(keys, q.FlowKeyV)
	}

	return &frameworkmocks.MockPriorityBandAccessor{
		PolicyStateV: state,
		FlowKeysFunc: func() []flowcontrol.FlowKey { return keys },
		QueueFunc: func(id string) flowcontrol.FlowQueueAccessor {
			if q, ok := queueMap[id]; ok {
				return q
			}
			return nil
		},
		IterateQueuesFunc: func(callback func(flow flowcontrol.FlowQueueAccessor) bool) {
			for _, q := range queues {
				if !callback(q) {
					return
				}
			}
		},
	}
}

func TestVTC_Pick_NilBand(t *testing.T) {
	t.Parallel()

	p, err := newVTC("test", nil)
	require.NoError(t, err)

	q, err := p.Pick(context.Background(), nil)
	require.NoError(t, err)
	assert.Nil(t, q)
}

func TestVTC_Pick_EmptyBand(t *testing.T) {
	t.Parallel()

	p, err := newVTC("test", nil)
	require.NoError(t, err)
	state := p.NewState(context.Background())

	band := &frameworkmocks.MockPriorityBandAccessor{
		PolicyStateV: state,
		FlowKeysFunc: func() []flowcontrol.FlowKey { return nil },
	}

	q, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	assert.Nil(t, q)
}

func TestVTC_Pick_AllQueuesEmpty(t *testing.T) {
	t.Parallel()

	p, err := newVTC("test", nil)
	require.NoError(t, err)
	state := p.NewState(context.Background())

	qA := newMockQueueWithCost(flowcontrol.FlowKey{ID: "a"}, 0, 0)
	qB := newMockQueueWithCost(flowcontrol.FlowKey{ID: "b"}, 0, 0)
	band := buildBand(state, qA, qB)

	q, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	assert.Nil(t, q)
}

func TestVTC_Pick_SingleFlow(t *testing.T) {
	t.Parallel()

	p, err := newVTC("test", nil)
	require.NoError(t, err)
	state := p.NewState(context.Background())

	keyA := flowcontrol.FlowKey{ID: "a"}
	qA := newMockQueueWithCost(keyA, 5, 100)
	band := buildBand(state, qA)

	q, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	require.NotNil(t, q)
	assert.Equal(t, "a", q.FlowKey().ID)
}

func TestVTC_Pick_LowestCounterSelected(t *testing.T) {
	t.Parallel()

	p, err := newVTC("test", nil)
	require.NoError(t, err)

	// Pre-seed counters: b has lower counter than a.
	state := &vtcState{
		counters: map[string]float64{
			"a": 500,
			"b": 100,
		},
	}

	keyA := flowcontrol.FlowKey{ID: "a"}
	keyB := flowcontrol.FlowKey{ID: "b"}
	qA := newMockQueueWithCost(keyA, 3, 100)
	qB := newMockQueueWithCost(keyB, 3, 100)
	band := buildBand(state, qA, qB)

	q, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	require.NotNil(t, q)
	assert.Equal(t, "b", q.FlowKey().ID, "flow with lower counter should be selected")
}

func TestVTC_Pick_EqualWeightBalancedDistribution(t *testing.T) {
	t.Parallel()

	p, err := newVTC("test", nil)
	require.NoError(t, err)
	state := p.NewState(context.Background())

	keyA := flowcontrol.FlowKey{ID: "a"}
	keyB := flowcontrol.FlowKey{ID: "b"}

	// Both flows send 100-byte requests with equal weight.
	counts := map[string]int{"a": 0, "b": 0}

	for range 20 {
		qA := newMockQueueWithCost(keyA, 5, 100)
		qB := newMockQueueWithCost(keyB, 5, 100)
		band := buildBand(state, qA, qB)

		q, err := p.Pick(context.Background(), band)
		require.NoError(t, err)
		require.NotNil(t, q)
		counts[q.FlowKey().ID]++
	}

	assert.Equal(t, 10, counts["a"], "equal-weight flows should each get half the dispatches")
	assert.Equal(t, 10, counts["b"], "equal-weight flows should each get half the dispatches")
}

func TestVTC_Pick_HigherWeightGetsMoreTurns(t *testing.T) {
	t.Parallel()

	// Flow a: weight=2.0, flow b: weight=1.0, same request size (100 bytes).
	// Per pick: a advances by 100/2=50, b advances by 100/1=100.
	// Over 30 picks: a should get ~20, b should get ~10.
	params := json.RawMessage(`{"weights": {"a": 2.0, "b": 1.0}}`)
	p, err := newVTC("test", params)
	require.NoError(t, err)
	state := p.NewState(context.Background())

	keyA := flowcontrol.FlowKey{ID: "a"}
	keyB := flowcontrol.FlowKey{ID: "b"}

	counts := map[string]int{"a": 0, "b": 0}

	for range 30 {
		qA := newMockQueueWithCost(keyA, 5, 100)
		qB := newMockQueueWithCost(keyB, 5, 100)
		band := buildBand(state, qA, qB)

		q, err := p.Pick(context.Background(), band)
		require.NoError(t, err)
		require.NotNil(t, q)
		counts[q.FlowKey().ID]++
	}

	assert.Equal(t, 20, counts["a"], "weight-2 flow should get 2/3 of dispatches")
	assert.Equal(t, 10, counts["b"], "weight-1 flow should get 1/3 of dispatches")
}

func TestVTC_Pick_LargeRequestAdvancesCounterFaster(t *testing.T) {
	t.Parallel()

	p, err := newVTC("test", nil)
	require.NoError(t, err)
	state := p.NewState(context.Background())

	keyA := flowcontrol.FlowKey{ID: "a"} // 1000-byte requests
	keyB := flowcontrol.FlowKey{ID: "b"} // 100-byte requests

	counts := map[string]int{"a": 0, "b": 0}

	for range 11 {
		qA := newMockQueueWithCost(keyA, 5, 1000)
		qB := newMockQueueWithCost(keyB, 5, 100)
		band := buildBand(state, qA, qB)

		q, err := p.Pick(context.Background(), band)
		require.NoError(t, err)
		require.NotNil(t, q)
		counts[q.FlowKey().ID]++
	}

	// After a's first pick (counter=1000), b needs 10 picks of 100 to catch up.
	// Then a picks again. Total: a=2, b=10 over 12 picks... but we only do 11.
	// So: a=1, b=10.
	assert.Equal(t, 1, counts["a"], "large-request flow should get fewer turns")
	assert.Equal(t, 10, counts["b"], "small-request flow should get more turns")
}

func TestVTC_Pick_NewFlowJoinsAtMinCounter(t *testing.T) {
	t.Parallel()

	p, err := newVTC("test", nil)
	require.NoError(t, err)
	state := p.NewState(context.Background())

	keyA := flowcontrol.FlowKey{ID: "a"}

	// Run a few picks for flow a to build up its counter.
	for range 5 {
		qA := newMockQueueWithCost(keyA, 5, 100)
		band := buildBand(state, qA)

		_, err := p.Pick(context.Background(), band)
		require.NoError(t, err)
	}

	// Now add flow b. It should start at the current minimum (a's counter = 500).
	// Both have counter=500, so the first (by sort order) wins. "a" < "b", so a wins.
	// After a wins (counter=600), b's turn (counter=500 < 600).
	keyB := flowcontrol.FlowKey{ID: "b"}

	qA := newMockQueueWithCost(keyA, 5, 100)
	qB := newMockQueueWithCost(keyB, 5, 100)
	band := buildBand(state, qA, qB)

	// First pick with both: a wins (tie-break by sort, a < b)
	q, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	require.NotNil(t, q)
	assert.Equal(t, "a", q.FlowKey().ID)

	// Second pick: b has counter=500, a has counter=600 -> b wins
	qA = newMockQueueWithCost(keyA, 5, 100)
	qB = newMockQueueWithCost(keyB, 5, 100)
	band = buildBand(state, qA, qB)

	q, err = p.Pick(context.Background(), band)
	require.NoError(t, err)
	require.NotNil(t, q)
	assert.Equal(t, "b", q.FlowKey().ID, "new flow should get a turn after joining at min counter")
}

func TestVTC_Pick_PrunesStaleCounters(t *testing.T) {
	t.Parallel()

	p, err := newVTC("test", nil)
	require.NoError(t, err)

	state := &vtcState{
		counters: map[string]float64{
			"a":     100,
			"stale": 999,
		},
	}

	// Only flow a is active now.
	keyA := flowcontrol.FlowKey{ID: "a"}
	qA := newMockQueueWithCost(keyA, 5, 100)
	band := buildBand(state, qA)

	_, err = p.Pick(context.Background(), band)
	require.NoError(t, err)

	state.mu.Lock()
	_, hasStale := state.counters["stale"]
	state.mu.Unlock()

	assert.False(t, hasStale, "counter for inactive flow should be pruned")
}

func TestVTC_Pick_CounterNormalization(t *testing.T) {
	t.Parallel()

	p, err := newVTC("test", nil)
	require.NoError(t, err)

	// Set counters near the normalization threshold.
	state := &vtcState{
		counters: map[string]float64{
			"a": normalizationThreshold + 100,
			"b": normalizationThreshold - 50,
		},
	}

	keyA := flowcontrol.FlowKey{ID: "a"}
	keyB := flowcontrol.FlowKey{ID: "b"}
	qA := newMockQueueWithCost(keyA, 5, 100)
	qB := newMockQueueWithCost(keyB, 5, 100)
	band := buildBand(state, qA, qB)

	_, err = p.Pick(context.Background(), band)
	require.NoError(t, err)

	// After pick, b was selected (lower counter), its counter advanced by 100.
	// Then normalization should subtract the new minimum from all counters.
	state.mu.Lock()
	for _, c := range state.counters {
		assert.Less(t, c, normalizationThreshold, "counters should be normalized below threshold")
	}
	state.mu.Unlock()
}

func TestVTC_Pick_Concurrency(t *testing.T) {
	t.Parallel()

	p, err := newVTC("test", nil)
	require.NoError(t, err)
	state := p.NewState(context.Background())

	keyA := flowcontrol.FlowKey{ID: "a"}
	keyB := flowcontrol.FlowKey{ID: "b"}

	var wg sync.WaitGroup
	numGoroutines := 10
	picksPerGoroutine := 50

	for range numGoroutines {
		wg.Go(func() {
			for range picksPerGoroutine {
				qA := newMockQueueWithCost(keyA, 5, 100)
				qB := newMockQueueWithCost(keyB, 5, 100)
				band := buildBand(state, qA, qB)

				_, err := p.Pick(context.Background(), band)
				assert.NoError(t, err)
			}
		})
	}

	wg.Wait()
}
