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

package roundrobin

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

func TestRoundRobin_Name(t *testing.T) {
	t.Parallel()
	policy := newRoundRobin()
	assert.Equal(t, RoundRobinPolicyName, policy.Name(), "Name should match the policy's constant")
}

func TestRoundRobin_SelectQueue_Logic(t *testing.T) {
	t.Parallel()
	policy := newRoundRobin()

	// Setup: Three non-empty queues
	queue1 := &frameworkmocks.MockFlowQueueAccessor{LenV: 1, FlowSpecV: types.FlowSpecification{ID: "flow1"}}
	queue2 := &frameworkmocks.MockFlowQueueAccessor{LenV: 2, FlowSpecV: types.FlowSpecification{ID: "flow2"}}
	queue3 := &frameworkmocks.MockFlowQueueAccessor{LenV: 3, FlowSpecV: types.FlowSpecification{ID: "flow3"}}

	mockBand := &frameworkmocks.MockPriorityBandAccessor{
		FlowIDsFunc: func() []string { return []string{"flow3", "flow1", "flow2"} }, // Unsorted to test sorting
		QueueFunc: func(id string) framework.FlowQueueAccessor {
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

	// Expected order is based on sorted FlowIDs: flow1, flow2, flow3
	expectedOrder := []string{"flow1", "flow2", "flow3"}

	// First cycle
	for i := range expectedOrder {
		selected, err := policy.SelectQueue(mockBand)
		require.NoError(t, err, "SelectQueue should not error on a valid band")
		require.NotNil(t, selected, "SelectQueue should have selected a queue")
		assert.Equal(t, expectedOrder[i], selected.FlowSpec().ID,
			"Cycle 1, selection %d should be %s", i+1, expectedOrder[i])
	}

	// Second cycle (wraps around)
	for i := range expectedOrder {
		selected, err := policy.SelectQueue(mockBand)
		require.NoError(t, err, "SelectQueue should not error on a valid band")
		require.NotNil(t, selected, "SelectQueue should have selected a queue")
		assert.Equal(t, expectedOrder[i], selected.FlowSpec().ID,
			"Cycle 2, selection %d should be %s", i+1, expectedOrder[i])
	}
}

func TestRoundRobin_SelectQueue_SkipsEmptyQueues(t *testing.T) {
	t.Parallel()
	policy := newRoundRobin()

	// Setup: Two non-empty queues and one empty queue
	queue1 := &frameworkmocks.MockFlowQueueAccessor{LenV: 1, FlowSpecV: types.FlowSpecification{ID: "flow1"}}
	queueEmpty := &frameworkmocks.MockFlowQueueAccessor{LenV: 0, FlowSpecV: types.FlowSpecification{ID: "flowEmpty"}}
	queue3 := &frameworkmocks.MockFlowQueueAccessor{LenV: 3, FlowSpecV: types.FlowSpecification{ID: "flow3"}}

	mockBand := &frameworkmocks.MockPriorityBandAccessor{
		FlowIDsFunc: func() []string { return []string{"flow1", "flowEmpty", "flow3"} },
		QueueFunc: func(id string) framework.FlowQueueAccessor {
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
	selected, err := policy.SelectQueue(mockBand)
	require.NoError(t, err, "SelectQueue should not error when skipping queues")
	require.NotNil(t, selected, "SelectQueue should select the first non-empty queue")
	assert.Equal(t, "flow1", selected.FlowSpec().ID, "First selection should be flow1")

	selected, err = policy.SelectQueue(mockBand)
	require.NoError(t, err, "SelectQueue should not error when skipping queues")
	require.NotNil(t, selected, "SelectQueue should select the second non-empty queue")
	assert.Equal(t, "flow3", selected.FlowSpec().ID, "Second selection should be flow3, skipping flowEmpty")

	selected, err = policy.SelectQueue(mockBand)
	require.NoError(t, err, "SelectQueue should not error when wrapping around")
	require.NotNil(t, selected, "SelectQueue should wrap around and select a queue")
	assert.Equal(t, "flow1", selected.FlowSpec().ID, "Should wrap around and select flow1 again")
}

func TestRoundRobin_SelectQueue_HandlesDynamicFlows(t *testing.T) {
	t.Parallel()
	policy := newRoundRobin()

	// Initial setup
	queue1 := &frameworkmocks.MockFlowQueueAccessor{LenV: 1, FlowSpecV: types.FlowSpecification{ID: "flow1"}}
	queue2 := &frameworkmocks.MockFlowQueueAccessor{LenV: 1, FlowSpecV: types.FlowSpecification{ID: "flow2"}}
	mockBand := &frameworkmocks.MockPriorityBandAccessor{
		FlowIDsFunc: func() []string { return []string{"flow1", "flow2"} },
		QueueFunc: func(id string) framework.FlowQueueAccessor {
			if id == "flow1" {
				return queue1
			}
			return queue2
		},
	}

	// First selection
	selected, err := policy.SelectQueue(mockBand)
	require.NoError(t, err, "SelectQueue should not error on initial selection")
	require.NotNil(t, selected, "SelectQueue should select a queue initially")
	assert.Equal(t, "flow1", selected.FlowSpec().ID, "First selection should be flow1")

	// --- Simulate adding a flow ---
	queue3 := &frameworkmocks.MockFlowQueueAccessor{LenV: 1, FlowSpecV: types.FlowSpecification{ID: "flow3"}}
	mockBand.FlowIDsFunc = func() []string { return []string{"flow1", "flow2", "flow3"} }
	mockBand.QueueFunc = func(id string) framework.FlowQueueAccessor {
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
	selected, err = policy.SelectQueue(mockBand)
	require.NoError(t, err, "SelectQueue should not error after adding a flow")
	require.NotNil(t, selected, "SelectQueue should select a queue after adding a flow")
	assert.Equal(t, "flow2", selected.FlowSpec().ID, "Next selection should be flow2")

	// Next selection should be flow3
	selected, err = policy.SelectQueue(mockBand)
	require.NoError(t, err, "SelectQueue should not error on the third selection")
	require.NotNil(t, selected, "SelectQueue should select the new flow")
	assert.Equal(t, "flow3", selected.FlowSpec().ID, "Next selection should be the new flow3")

	// --- Simulate removing a flow ---
	mockBand.FlowIDsFunc = func() []string { return []string{"flow1", "flow3"} } // flow2 is removed

	// Next selection should wrap around and pick flow1
	selected, err = policy.SelectQueue(mockBand)
	require.NoError(t, err, "SelectQueue should not error after removing a flow")
	require.NotNil(t, selected, "SelectQueue should select a queue after removing a flow")
	assert.Equal(t, "flow1", selected.FlowSpec().ID, "Next selection should wrap around to flow1 after a removal")
}

func TestRoundRobin_SelectQueue_Concurrency(t *testing.T) {
	t.Parallel()
	// Run this test multiple times to increase the chance of catching race conditions.
	for i := range 5 {
		t.Run(fmt.Sprintf("Iteration%d", i), func(t *testing.T) {
			t.Parallel()
			policy := newRoundRobin()

			// Setup: Three non-empty queues
			queues := []*frameworkmocks.MockFlowQueueAccessor{
				{LenV: 1, FlowSpecV: types.FlowSpecification{ID: "flow1"}},
				{LenV: 2, FlowSpecV: types.FlowSpecification{ID: "flow2"}},
				{LenV: 3, FlowSpecV: types.FlowSpecification{ID: "flow3"}},
			}
			numQueues := int64(len(queues))

			mockBand := &frameworkmocks.MockPriorityBandAccessor{
				FlowIDsFunc: func() []string { return []string{"flow1", "flow2", "flow3"} },
				QueueFunc: func(id string) framework.FlowQueueAccessor {
					for _, q := range queues {
						if q.FlowSpec().ID == id {
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
						selected, err := policy.SelectQueue(mockBand)
						if err == nil && selected != nil {
							val, _ := selectionCounts.LoadOrStore(selected.FlowSpec().ID, new(atomic.Int64))
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
