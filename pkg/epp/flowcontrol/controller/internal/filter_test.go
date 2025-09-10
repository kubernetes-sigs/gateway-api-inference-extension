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

package internal

import (
	"context"
	"sort"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

func TestNewSaturationFilter(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name                  string
		isSaturated           bool
		expectShouldPause     bool
		expectFilteredBandNil bool
	}{
		{
			name:                  "should not pause or filter when system is not saturated",
			isSaturated:           false,
			expectShouldPause:     false,
			expectFilteredBandNil: true, // nil band signals the fast path
		},
		{
			name:                  "should pause when system is saturated",
			isSaturated:           true,
			expectShouldPause:     true,
			expectFilteredBandNil: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// --- ARRANGE ---
			mockSD := &mocks.MockSaturationDetector{IsSaturatedFunc: func(ctx context.Context) bool { return tc.isSaturated }}
			filter := NewSaturationFilter(mockSD)
			require.NotNil(t, filter, "NewSaturationFilter should not return nil")

			mockBand := &frameworkmocks.MockPriorityBandAccessor{}

			// --- ACT ---
			filteredBand, shouldPause := filter(context.Background(), mockBand, logr.Discard())

			// --- ASSERT ---
			assert.Equal(t, tc.expectShouldPause, shouldPause, "The filter's pause signal should match the expected value")

			if tc.expectFilteredBandNil {
				assert.Nil(t, filteredBand, "Expected filtered band to be nil")
			} else {
				assert.NotNil(t, filteredBand, "Expected a non-nil filtered band")
			}
		})
	}
}

func TestSubsetPriorityBandAccessor(t *testing.T) {
	t.Parallel()

	// --- ARRANGE ---
	// Setup a mock original accessor that knows about three flows.
	flowAKey := types.FlowKey{ID: "flow-a", Priority: 10}
	flowBKey := types.FlowKey{ID: "flow-b", Priority: 10}
	flowCKey := types.FlowKey{ID: "flow-c", Priority: 10}

	mockQueueA := &frameworkmocks.MockFlowQueueAccessor{FlowKeyV: flowAKey}
	mockQueueB := &frameworkmocks.MockFlowQueueAccessor{FlowKeyV: flowBKey}
	mockQueueC := &frameworkmocks.MockFlowQueueAccessor{FlowKeyV: flowCKey}

	originalAccessor := &frameworkmocks.MockPriorityBandAccessor{
		PriorityV:     10,
		PriorityNameV: "High",
		FlowKeysFunc: func() []types.FlowKey {
			return []types.FlowKey{flowAKey, flowBKey, flowCKey}
		},
		QueueFunc: func(id string) framework.FlowQueueAccessor {
			switch id {
			case "flow-a":
				return mockQueueA
			case "flow-b":
				return mockQueueB
			case "flow-c":
				return mockQueueC
			}
			return nil
		},
		IterateQueuesFunc: func(callback func(queue framework.FlowQueueAccessor) bool) {
			if !callback(mockQueueA) {
				return
			}
			if !callback(mockQueueB) {
				return
			}
			callback(mockQueueC)
		},
	}

	// Create a subset view that only allows two of the flows.
	allowedFlows := []types.FlowKey{flowAKey, flowCKey}
	subsetAccessor := newSubsetPriorityBandAccessor(originalAccessor, allowedFlows)
	require.NotNil(t, subsetAccessor, "newSubsetPriorityBandAccessor should not return nil")

	t.Run("should pass through priority and name", func(t *testing.T) {
		t.Parallel()
		assert.Equal(t, 10, subsetAccessor.Priority(), "Priority() should pass through from the original accessor")
		assert.Equal(t, "High", subsetAccessor.PriorityName(),
			"PriorityName() should pass through from the original accessor")
	})

	t.Run("should only return allowed flow keys", func(t *testing.T) {
		t.Parallel()
		flowKeys := subsetAccessor.FlowKeys()
		// Sort for consistent comparison, as the pre-computed slice order is not guaranteed.
		sort.Slice(flowKeys, func(i, j int) bool {
			return flowKeys[i].ID < flowKeys[j].ID
		})
		assert.Equal(t, []types.FlowKey{flowAKey, flowCKey}, flowKeys, "FlowKeys() should only return the allowed subset")
	})

	t.Run("should only return queues for allowed flows", func(t *testing.T) {
		t.Parallel()
		assert.Same(t, mockQueueA, subsetAccessor.Queue("flow-a"), "Should return queue for allowed flow 'a'")
		assert.Nil(t, subsetAccessor.Queue("flow-b"), "Should not return queue for disallowed flow 'b'")
		assert.Same(t, mockQueueC, subsetAccessor.Queue("flow-c"), "Should return queue for allowed flow 'c'")
	})

	t.Run("should only iterate over allowed queues", func(t *testing.T) {
		t.Parallel()
		var iterated []string
		subsetAccessor.IterateQueues(func(queue framework.FlowQueueAccessor) bool {
			iterated = append(iterated, queue.FlowKey().ID)
			return true
		})
		// Sort for consistent comparison, as iteration order is not guaranteed.
		sort.Strings(iterated)
		assert.Equal(t, []string{"flow-a", "flow-c"}, iterated, "IterateQueues() should only visit allowed flows")
	})

	t.Run("should stop iteration if callback returns false", func(t *testing.T) {
		t.Parallel()
		var iterated []string
		subsetAccessor.IterateQueues(func(queue framework.FlowQueueAccessor) bool {
			iterated = append(iterated, queue.FlowKey().ID)
			return false // Exit after the first item.
		})
		assert.Len(t, iterated, 1, "Iteration should have stopped after one item")
	})
}
