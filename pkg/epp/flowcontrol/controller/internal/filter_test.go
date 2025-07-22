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
		name              string
		isSaturated       bool
		expectShouldPause bool
		expectAllowed     map[string]struct{}
	}{
		{
			name:              "should not pause or filter when system is not saturated",
			isSaturated:       false,
			expectShouldPause: false,
			expectAllowed:     nil, // nil map signals the fast path
		},
		{
			name:              "should pause when system is saturated",
			isSaturated:       true,
			expectShouldPause: true,
			expectAllowed:     nil,
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
			allowed, shouldPause := filter(context.Background(), mockBand, logr.Discard())

			// --- ASSERT ---
			assert.Equal(t, tc.expectShouldPause, shouldPause, "The filter's pause signal should match the expected value")

			if tc.expectAllowed == nil {
				assert.Nil(t, allowed, "Expected allowed map to be nil for the fast path")
			} else {
				assert.Equal(t, tc.expectAllowed, allowed, "The set of allowed flows should match the expected value")
			}
		})
	}
}

func TestSubsetPriorityBandAccessor(t *testing.T) {
	t.Parallel()

	// --- ARRANGE ---
	// Setup a mock original accessor that knows about three flows.
	mockQueueA := &frameworkmocks.MockFlowQueueAccessor{FlowSpecV: types.FlowSpecification{ID: "flow-a"}}
	mockQueueB := &frameworkmocks.MockFlowQueueAccessor{FlowSpecV: types.FlowSpecification{ID: "flow-b"}}
	mockQueueC := &frameworkmocks.MockFlowQueueAccessor{FlowSpecV: types.FlowSpecification{ID: "flow-c"}}

	originalAccessor := &frameworkmocks.MockPriorityBandAccessor{
		PriorityV:     10,
		PriorityNameV: "High",
		FlowIDsFunc: func() []string {
			return []string{"flow-a", "flow-b", "flow-c"}
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
	allowedFlows := map[string]struct{}{
		"flow-a": {},
		"flow-c": {},
	}
	subsetAccessor := newSubsetPriorityBandAccessor(originalAccessor, allowedFlows)
	require.NotNil(t, subsetAccessor, "newSubsetPriorityBandAccessor should not return nil")

	t.Run("should pass through priority and name", func(t *testing.T) {
		t.Parallel()
		assert.Equal(t, uint(10), subsetAccessor.Priority(), "Priority() should pass through from the original accessor")
		assert.Equal(t, "High", subsetAccessor.PriorityName(),
			"PriorityName() should pass through from the original accessor")
	})

	t.Run("should only return allowed flow IDs", func(t *testing.T) {
		t.Parallel()
		flowIDs := subsetAccessor.FlowIDs()
		// Sort for consistent comparison, as the pre-computed slice order is not guaranteed.
		sort.Strings(flowIDs)
		assert.Equal(t, []string{"flow-a", "flow-c"}, flowIDs, "FlowIDs() should only return the allowed subset")
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
			iterated = append(iterated, queue.FlowSpec().ID)
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
			iterated = append(iterated, queue.FlowSpec().ID)
			return false // Exit after the first item.
		})
		assert.Len(t, iterated, 1, "Iteration should have stopped after one item")
	})
}
