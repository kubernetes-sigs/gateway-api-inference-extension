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

package registry

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// fsTestHarness holds all the common components for a `flowState` test.
type fsTestHarness struct {
	t         *testing.T
	fs        *flowState
	allShards []*registryShard
}

// newFsTestHarness creates a new test harness, initializing a `flowState` with a default spec and shard layout.
func newFsTestHarness(t *testing.T) *fsTestHarness {
	t.Helper()

	spec := types.FlowSpecification{
		Key: types.FlowKey{ID: "f1", Priority: 10},
	}
	// Note: The shards are simple structs for this test, as we only need their IDs.
	allShards := []*registryShard{{id: "s1"}, {id: "s2"}, {id: "s3-draining"}}

	fs := newFlowState(spec, allShards)

	return &fsTestHarness{
		t:         t,
		fs:        fs,
		allShards: allShards,
	}
}

func TestFlowState_New(t *testing.T) {
	t.Parallel()
	h := newFsTestHarness(t)

	assert.Equal(t, uint64(0), h.fs.lastActiveGeneration, "Initial lastActiveGeneration should be 0")
	require.Len(t, h.fs.emptyOnShards, 3, "Should track emptiness status for all initial shards")

	// A new flow instance should be considered empty on all shards by default.
	assert.True(t, h.fs.emptyOnShards["s1"], "Queue on s1 should start as empty")
	assert.True(t, h.fs.emptyOnShards["s2"], "Queue on s2 should start as empty")
	assert.True(t, h.fs.emptyOnShards["s3-draining"], "Queue on s3-draining should start as empty")
}

func TestFlowState_Update(t *testing.T) {
	t.Parallel()

	t.Run("ShouldUpdateSpec_WhenKeyIsUnchanged", func(t *testing.T) {
		t.Parallel()
		h := newFsTestHarness(t)

		// Create a new spec with the same key but different (hypothetical) content.
		updatedSpec := types.FlowSpecification{
			Key: h.fs.spec.Key,
			// Imagine other fields like per-flow `framework.IntraFlowDispatchPolicy` overrides being added here in the
			// future.
		}

		h.fs.update(updatedSpec)
		assert.Equal(t, updatedSpec, h.fs.spec, "Spec should be updated with new content")
	})

	t.Run("ShouldPanic_WhenKeyIsChanged", func(t *testing.T) {
		t.Parallel()
		h := newFsTestHarness(t)

		invalidSpec := types.FlowSpecification{
			Key: types.FlowKey{ID: "f1", Priority: 99}, // Different priority
		}

		assert.PanicsWithValue(t,
			fmt.Sprintf("invariant violation: attempted to update flowState for key %s with a new key %s",
				h.fs.spec.Key, invalidSpec.Key),
			func() { h.fs.update(invalidSpec) },
			"Should panic when attempting to change the immutable FlowKey",
		)
	})
}

func TestFlowState_HandleQueueSignal(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name          string
		signal        queueStateSignal
		initialState  bool
		expectedState bool
		message       string
	}{
		{
			name:          "BecameNonEmpty_WhenPreviouslyEmpty",
			signal:        queueStateSignalBecameNonEmpty,
			initialState:  true,
			expectedState: false,
			message:       "Should mark queue as non-empty (false)",
		},
		{
			name:          "BecameEmpty_WhenPreviouslyNonEmpty",
			signal:        queueStateSignalBecameEmpty,
			initialState:  false,
			expectedState: true,
			message:       "Should mark queue as empty (true)",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			h := newFsTestHarness(t)
			shardID := "s1"

			h.fs.emptyOnShards[shardID] = tc.initialState
			h.fs.handleQueueSignal(shardID, tc.signal)
			assert.Equal(t, tc.expectedState, h.fs.emptyOnShards[shardID], tc.message)
		})
	}
}

func TestFlowState_IsIdle(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name       string
		setup      func(h *fsTestHarness)
		shards     []*registryShard // The set of shards to check for idleness.
		expectIdle bool
		message    string
	}{
		{
			name:       "WhenAllShardsAreEmpty",
			setup:      nil, // Default state is all empty.
			expectIdle: true,
			message:    "Should be idle when all tracked queues are empty",
		},
		{
			name: "WhenOneShardIsNonEmpty",
			setup: func(h *fsTestHarness) {
				h.fs.handleQueueSignal("s2", queueStateSignalBecameNonEmpty)
			},
			expectIdle: false,
			message:    "Should not be idle if any queue is non-empty",
		},
		{
			name: "WhenADrainingShardIsNonEmpty",
			setup: func(h *fsTestHarness) {
				h.fs.handleQueueSignal("s3-draining", queueStateSignalBecameNonEmpty)
			},
			expectIdle: false,
			message:    "Should not be idle if a draining shard's queue is non-empty",
		},
		{
			name: "WhenNonEmptyShardIsNotInCheckedSet",
			setup: func(h *fsTestHarness) {
				// The flow is active on a shard that is Draining.
				h.fs.handleQueueSignal("s3-draining", queueStateSignalBecameNonEmpty)
			},
			// But we are only checking the set of Active shards.
			shards:     []*registryShard{{id: "s1"}, {id: "s2"}},
			expectIdle: true,
			message:    "Should be considered idle if the only non-empty queue is on a shard that is not being checked",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			h := newFsTestHarness(t)
			if tc.setup != nil {
				tc.setup(h)
			}

			shardsToCheck := tc.shards
			if shardsToCheck == nil {
				shardsToCheck = h.allShards
			}

			assert.Equal(t, tc.expectIdle, h.fs.isIdle(shardsToCheck), tc.message)
		})
	}
}

func TestFlowState_PurgeShard(t *testing.T) {
	t.Parallel()
	h := newFsTestHarness(t)
	shardToPurge := "s2"

	require.Contains(t, h.fs.emptyOnShards, shardToPurge, "Test setup: shard must exist before purging")
	h.fs.purgeShard(shardToPurge)

	assert.NotContains(t, h.fs.emptyOnShards, shardToPurge, "Shard should be purged from the tracking map")
	assert.Contains(t, h.fs.emptyOnShards, "s1", "Other shards should remain in the tracking map")
	assert.Len(t, h.fs.emptyOnShards, 2, "Map length should be reduced by one")
}
