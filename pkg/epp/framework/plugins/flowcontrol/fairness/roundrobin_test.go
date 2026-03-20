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

package fairness

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol/mocks"
)

var (
	flow1Key = flowcontrol.FlowKey{ID: "flow1", Priority: 0}
	flow2Key = flowcontrol.FlowKey{ID: "flow2", Priority: 0}
	flow3Key = flowcontrol.FlowKey{ID: "flow3", Priority: 0}
)

func TestRoundRobin_Name(t *testing.T) {
	t.Parallel()
	policy := newRoundRobin("test-rr")
	assert.Equal(t, "test-rr", policy.TypedName().Name)
	assert.Equal(t, RoundRobinFairnessPolicyType, policy.TypedName().Type)
}

func TestRoundRobin_Pick_Logic(t *testing.T) {
	t.Parallel()
	policy := newRoundRobin("")
	ctx := context.Background()
	state := policy.NewState(ctx)

	// Setup: Three non-empty queues
	queue1 := &frameworkmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow1Key}
	queue2 := &frameworkmocks.MockFlowQueueAccessor{LenV: 2, FlowKeyV: flow2Key}
	queue3 := &frameworkmocks.MockFlowQueueAccessor{LenV: 3, FlowKeyV: flow3Key}

	mockBand := &frameworkmocks.MockPriorityBandAccessor{
		PolicyStateV: state,
		FlowKeysFunc: func() []flowcontrol.FlowKey { return []flowcontrol.FlowKey{flow3Key, flow1Key, flow2Key} }, // Unsorted to test sorting
		QueueFunc: func(id string) flowcontrol.FlowQueueAccessor {
			switch id {
			case "flow1":
				return queue1
			case "flow2":
				return queue2
			case "flow3":
				return queue3
			}
			return nil
		},
	}

	// Expected order is based on sorted FlowKeys: flow1, flow2, flow3
	expectedOrder := []string{"flow1", "flow2", "flow3"}

	// First cycle
	for i := range expectedOrder {
		selected, err := policy.Pick(ctx, mockBand)
		require.NoError(t, err, "Pick should not error on a valid band")
		require.NotNil(t, selected, "Pick should have selected a queue")
		assert.Equal(t, expectedOrder[i], selected.FlowKey().ID,
			"Cycle 1, selection %d should be %s", i+1, expectedOrder[i])
	}

	// Second cycle (wraps around)
	for i := range expectedOrder {
		selected, err := policy.Pick(ctx, mockBand)
		require.NoError(t, err, "Pick should not error on a valid band")
		require.NotNil(t, selected, "Pick should have selected a queue")
		assert.Equal(t, expectedOrder[i], selected.FlowKey().ID,
			"Cycle 2, selection %d should be %s", i+1, expectedOrder[i])
	}
}

func TestRoundRobin_Pick_SkipsEmptyQueues(t *testing.T) {
	t.Parallel()
	policy := newRoundRobin("")
	ctx := context.Background()
	state := policy.NewState(ctx)

	// Setup: Two non-empty queues and one empty queue
	flowEmptyKey := flowcontrol.FlowKey{ID: "flowEmpty", Priority: 0}
	queue1 := &frameworkmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow1Key}
	queueEmpty := &frameworkmocks.MockFlowQueueAccessor{LenV: 0, FlowKeyV: flowEmptyKey}
	queue3 := &frameworkmocks.MockFlowQueueAccessor{LenV: 3, FlowKeyV: flow3Key}

	mockBand := &frameworkmocks.MockPriorityBandAccessor{
		PolicyStateV: state,
		FlowKeysFunc: func() []flowcontrol.FlowKey { return []flowcontrol.FlowKey{flow1Key, flowEmptyKey, flow3Key} },
		QueueFunc: func(id string) flowcontrol.FlowQueueAccessor {
			switch id {
			case "flow1":
				return queue1
			case "flowEmpty":
				return queueEmpty
			case "flow3":
				return queue3
			}
			return nil
		},
	}

	// Expected order: flow1, flow3, flow1, flow3, ...
	selected, err := policy.Pick(ctx, mockBand)
	require.NoError(t, err, "Pick should not error when skipping queues")
	require.NotNil(t, selected, "Pick should select the first non-empty queue")
	assert.Equal(t, "flow1", selected.FlowKey().ID, "First selection should be flow1")

	selected, err = policy.Pick(ctx, mockBand)
	require.NoError(t, err, "Pick should not error when skipping queues")
	require.NotNil(t, selected, "Pick should select the second non-empty queue")
	assert.Equal(t, "flow3", selected.FlowKey().ID, "Second selection should be flow3, skipping flowEmpty")

	selected, err = policy.Pick(ctx, mockBand)
	require.NoError(t, err, "Pick should not error when wrapping around")
	require.NotNil(t, selected, "Pick should wrap around and select a queue")
	assert.Equal(t, "flow1", selected.FlowKey().ID, "Should wrap around and select flow1 again")
}

func TestRoundRobin_Pick_HandlesDynamicFlows(t *testing.T) {
	t.Parallel()
	policy := newRoundRobin("")
	ctx := context.Background()
	state := policy.NewState(ctx)

	// Initial setup
	queue1 := &frameworkmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow1Key}
	queue2 := &frameworkmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow2Key}
	mockBand := &frameworkmocks.MockPriorityBandAccessor{
		PolicyStateV: state,
		FlowKeysFunc: func() []flowcontrol.FlowKey { return []flowcontrol.FlowKey{flow1Key, flow2Key} },
		QueueFunc: func(id string) flowcontrol.FlowQueueAccessor {
			if id == "flow1" {
				return queue1
			}
			return queue2
		},
	}

	// First selection
	selected, err := policy.Pick(ctx, mockBand)
	require.NoError(t, err, "Pick should not error on initial selection")
	require.NotNil(t, selected, "Pick should select a queue initially")
	assert.Equal(t, "flow1", selected.FlowKey().ID, "First selection should be flow1")

	// --- Simulate adding a flow ---
	queue3 := &frameworkmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow3Key}
	mockBand.FlowKeysFunc = func() []flowcontrol.FlowKey { return []flowcontrol.FlowKey{flow1Key, flow2Key, flow3Key} }
	mockBand.QueueFunc = func(id string) flowcontrol.FlowQueueAccessor {
		switch id {
		case "flow1":
			return queue1
		case "flow2":
			return queue2
		case "flow3":
			return queue3
		}
		return nil
	}

	// Next selection should be flow2 (continues from last index)
	selected, err = policy.Pick(ctx, mockBand)
	require.NoError(t, err, "Pick should not error after adding a flow")
	require.NotNil(t, selected, "Pick should select a queue after adding a flow")
	assert.Equal(t, "flow2", selected.FlowKey().ID, "Next selection should be flow2")

	// Next selection should be flow3
	selected, err = policy.Pick(ctx, mockBand)
	require.NoError(t, err, "Pick should not error on the third selection")
	require.NotNil(t, selected, "Pick should select the new flow")
	assert.Equal(t, "flow3", selected.FlowKey().ID, "Next selection should be the new flow3")

	// --- Simulate removing a flow ---
	mockBand.FlowKeysFunc = func() []flowcontrol.FlowKey { return []flowcontrol.FlowKey{flow1Key, flow3Key} } // flow2 is removed

	// Next selection should wrap around and pick flow1
	selected, err = policy.Pick(ctx, mockBand)
	require.NoError(t, err, "Pick should not error after removing a flow")
	require.NotNil(t, selected, "Pick should select a queue after removing a flow")
	assert.Equal(t, "flow1", selected.FlowKey().ID, "Next selection should wrap around to flow1 after a removal")
}

func TestRoundRobin_Pick_Concurrency(t *testing.T) {
	t.Parallel()
	// Run this test multiple times to increase the chance of catching race conditions.
	for i := range 5 {
		t.Run(fmt.Sprintf("Iteration%d", i), func(t *testing.T) {
			t.Parallel()
			policy := newRoundRobin("")
			ctx := context.Background()
			state := policy.NewState(ctx) // Shared state for all goroutines

			// Setup: Three non-empty queues
			queues := []*frameworkmocks.MockFlowQueueAccessor{
				{LenV: 1, FlowKeyV: flow1Key},
				{LenV: 2, FlowKeyV: flow2Key},
				{LenV: 3, FlowKeyV: flow3Key},
			}
			numQueues := int64(len(queues))

			mockBand := &frameworkmocks.MockPriorityBandAccessor{
				PolicyStateV: state,
				FlowKeysFunc: func() []flowcontrol.FlowKey { return []flowcontrol.FlowKey{flow1Key, flow2Key, flow3Key} },
				QueueFunc: func(id string) flowcontrol.FlowQueueAccessor {
					for _, q := range queues {
						if q.FlowKey().ID == id {
							return q
						}
					}
					return nil
				},
			}

			var wg sync.WaitGroup
			numGoroutines := 10
			selectionsPerGoroutine := 30
			totalSelections := int64(numGoroutines * selectionsPerGoroutine)

			var selectionCounts sync.Map // Used like a concurrent map[string]*atomic.Int64

			wg.Add(numGoroutines)
			for range numGoroutines {
				go func() {
					defer wg.Done()
					for range selectionsPerGoroutine {
						selected, err := policy.Pick(ctx, mockBand)
						if err == nil && selected != nil {
							val, _ := selectionCounts.LoadOrStore(selected.FlowKey().ID, new(atomic.Int64))
							val.(*atomic.Int64).Add(1)
						}
					}
				}()
			}
			wg.Wait()

			var finalCount int64
			countsStr := ""
			selectionCounts.Range(func(key, value any) bool {
				count := value.(*atomic.Int64).Load()
				finalCount += count
				countsStr += fmt.Sprintf("%s: %d, ", key, count)

				// Check for reasonable distribution.
				// In a perfect world, each queue gets totalSelections / numQueues.
				// We allow for some variance due to scheduling, but it shouldn't be wildly off.
				// A simple check is that each queue gets at least a certain fraction of its expected share.
				expectedCount := totalSelections / numQueues
				minExpectedCount := expectedCount / 2 // Expect at least half of the ideal distribution
				assert.True(t, count > minExpectedCount,
					"Queue %s was selected only %d times, expected at least %d", key, count, minExpectedCount)
				return true
			})

			assert.Equal(t, totalSelections, finalCount, "Total selections should match the expected number")
			t.Logf("Selection distribution: %s", countsStr)
		})
	}
}
