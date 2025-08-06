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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// fsTestHarness holds all the common components for a `flowState` test.
type fsTestHarness struct {
	t            *testing.T
	fs           *flowState
	allShards    []*registryShard
	activeShards []*registryShard
	spec         types.FlowSpecification
}

// newFsTestHarness creates a new test harness, initializing a `flowState` with a default spec and shard layout.
func newFsTestHarness(t *testing.T) *fsTestHarness {
	t.Helper()

	spec := types.FlowSpecification{ID: "f1", Priority: pHigh}
	activeShards := []*registryShard{{id: "s1"}, {id: "s2"}}
	allShards := append(activeShards, &registryShard{id: "s3-draining"}) // `s3` is conceptually draining

	fs := newFlowState(spec, allShards)

	return &fsTestHarness{
		t:            t,
		fs:           fs,
		allShards:    allShards,
		activeShards: activeShards,
		spec:         spec,
	}
}

func TestFlowState(t *testing.T) {
	t.Parallel()

	t.Run("New", func(t *testing.T) {
		t.Parallel()
		h := newFsTestHarness(t)

		assert.Equal(t, h.spec, h.fs.spec, "Spec should be correctly initialized")
		assert.Equal(t, uint64(1), h.fs.generation, "Initial generation should be 1")
		require.Len(t, h.fs.activeQueueEmptyOnShards, 3, "Should track all initial shards")

		// New queues should start empty on all shards.
		assert.True(t, h.fs.activeQueueEmptyOnShards["s1"], "Queue on s1 should start empty")
		assert.True(t, h.fs.activeQueueEmptyOnShards["s2"], "Queue on s2 should start empty")
		assert.True(t, h.fs.activeQueueEmptyOnShards["s3-draining"], "Queue on s3-draining should start empty")
	})

	t.Run("Update", func(t *testing.T) {
		t.Parallel()

		specHigh := types.FlowSpecification{ID: "f1", Priority: pHigh}
		specLow := types.FlowSpecification{ID: "f1", Priority: pLow}

		testCases := []struct {
			name        string
			setup       func(h *fsTestHarness)
			updatedSpec types.FlowSpecification
			assertions  func(h *fsTestHarness)
		}{
			{
				name: "WhenPriorityIsUnchanged_ShouldPreserveState",
				setup: func(h *fsTestHarness) {
					h.fs.handleQueueSignal("s1", pHigh, queueStateSignalBecameNonEmpty)
				},
				updatedSpec: specHigh,
				assertions: func(h *fsTestHarness) {
					assert.Equal(t, uint64(2), h.fs.generation, "Generation should increment on any update")
					assert.Equal(t, specHigh, h.fs.spec, "Spec should be updated")
					assert.Empty(t, h.fs.drainingQueuesEmptyOnShards, "Draining map should be empty when priority does not change")
					assert.False(t, h.fs.activeQueueEmptyOnShards["s1"], "State of non-emptiness should be preserved")
				},
			},
			{
				name: "WhenPriorityChanges_ShouldMoveOldActiveToDraining",
				setup: func(h *fsTestHarness) {
					h.fs.handleQueueSignal("s1", pHigh, queueStateSignalBecameNonEmpty)
					require.False(t, h.fs.activeQueueEmptyOnShards["s1"], "Test setup: s1 should be non-empty")
				},
				updatedSpec: specLow,
				assertions: func(h *fsTestHarness) {
					assert.Equal(t, uint64(2), h.fs.generation, "Generation should increment after update")
					assert.Equal(t, specLow, h.fs.spec, "Spec should be updated to new priority")

					require.Contains(t, h.fs.drainingQueuesEmptyOnShards, uint(pHigh), "pHigh should now be in the draining map")
					drainingState := h.fs.drainingQueuesEmptyOnShards[pHigh]
					assert.False(t, drainingState["s1"], "s1 should still be non-empty for pHigh (draining)")
					assert.True(t, drainingState["s2"], "s2 should still be empty for pHigh (draining)")

					assert.True(t, h.fs.activeQueueEmptyOnShards["s1"], "s1 should start empty for new pLow (active)")
					assert.True(t, h.fs.activeQueueEmptyOnShards["s2"], "s2 should start empty for new pLow (active)")
				},
			},
			{
				name: "WhenPriorityRollsBack_ShouldReactivateDrainingQueue",
				setup: func(h *fsTestHarness) {
					// Step 1: `pHigh` -> `pLow`. This makes `pHigh` draining.
					h.fs.update(specLow, h.allShards)
					require.Contains(t, h.fs.drainingQueuesEmptyOnShards, uint(pHigh), "Test setup: pHigh should be draining")
					require.Equal(t, uint64(2), h.fs.generation, "Test setup: generation should be 2")
				},
				updatedSpec: specHigh, // This is the rollback update.
				assertions: func(h *fsTestHarness) {
					assert.Equal(t, uint64(3), h.fs.generation, "Generation should increment upon reactivation")
					assert.NotContains(t, h.fs.drainingQueuesEmptyOnShards, uint(pHigh),
						"pHigh should be removed from draining map on reactivation")
					assert.Contains(t, h.fs.drainingQueuesEmptyOnShards, uint(pLow), "pLow should now be in the draining map")
					assert.Equal(t, specHigh, h.fs.spec, "Spec should be back to pHigh")
				},
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				h := newFsTestHarness(t)
				if tc.setup != nil {
					tc.setup(h)
				}

				h.fs.update(tc.updatedSpec, h.allShards)

				tc.assertions(h)
			})
		}
	})

	t.Run("HandleQueueSignal", func(t *testing.T) {
		t.Parallel()
		testCases := []struct {
			name        string
			signal      queueStateSignal
			shardID     string
			priority    uint
			setup       func(h *fsTestHarness)
			assertState func(h *fsTestHarness)
		}{
			{
				name:        "BecameNonEmpty_ShouldMarkActiveQueueAsNonEmpty",
				signal:      queueStateSignalBecameNonEmpty,
				shardID:     "s1",
				priority:    pHigh,
				assertState: func(h *fsTestHarness) { assert.False(t, h.fs.activeQueueEmptyOnShards["s1"]) },
			},
			{
				name:        "BecameEmpty_ShouldMarkActiveQueueAsEmpty",
				signal:      queueStateSignalBecameEmpty,
				shardID:     "s1",
				priority:    pHigh,
				setup:       func(h *fsTestHarness) { h.fs.activeQueueEmptyOnShards["s1"] = false },
				assertState: func(h *fsTestHarness) { assert.True(t, h.fs.activeQueueEmptyOnShards["s1"]) },
			},
			{
				name:     "BecameDrained_ShouldMarkDrainingQueueAsEmpty",
				signal:   queueStateSignalBecameDrained,
				shardID:  "s2",
				priority: pHigh, // The original priority, which is now draining
				setup: func(h *fsTestHarness) {
					h.fs.update(types.FlowSpecification{ID: "f1", Priority: pLow}, h.allShards)
					h.fs.drainingQueuesEmptyOnShards[pHigh]["s2"] = false
				},
				assertState: func(h *fsTestHarness) { assert.True(t, h.fs.drainingQueuesEmptyOnShards[pHigh]["s2"]) },
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				h := newFsTestHarness(t)
				if tc.setup != nil {
					tc.setup(h)
				}
				h.fs.handleQueueSignal(tc.shardID, tc.priority, tc.signal)
				tc.assertState(h)
			})
		}
	})

	t.Run("IsIdle", func(t *testing.T) {
		t.Parallel()
		testCases := []struct {
			name       string
			setup      func(h *fsTestHarness)
			expectIdle bool
		}{
			{"WhenAllActiveShardsAreEmpty_ShouldReturnTrue", nil, true},
			{
				"WhenOneActiveShardIsNonEmpty_ShouldReturnFalse",
				func(h *fsTestHarness) { h.fs.handleQueueSignal("s1", pHigh, queueStateSignalBecameNonEmpty) },
				false,
			},
			{
				"WhenOnlyDrainingShardsAreNonEmpty_ShouldReturnTrue",
				func(h *fsTestHarness) { h.fs.handleQueueSignal("s3-draining", pHigh, queueStateSignalBecameNonEmpty) },
				true,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				h := newFsTestHarness(t)
				if tc.setup != nil {
					tc.setup(h)
				}
				assert.Equal(t, tc.expectIdle, h.fs.isIdle(h.activeShards))
			})
		}
	})

	t.Run("IsDrained", func(t *testing.T) {
		t.Parallel()
		drainingPriority := uint(pHigh)
		activePriority := uint(pLow)
		testCases := []struct {
			name           string
			priorityToTest uint
			setup          func(h *fsTestHarness)
			expectDrained  bool
		}{
			{"WhenQueueIsEmptyOnAllShards_ShouldReturnTrue", drainingPriority, nil, true},
			{
				"WhenQueueIsNonEmptyOnActiveShard_ShouldReturnFalse",
				drainingPriority,
				func(h *fsTestHarness) { h.fs.drainingQueuesEmptyOnShards[drainingPriority]["s1"] = false },
				false,
			},
			{
				"WhenQueueIsNonEmptyOnDrainingShard_ShouldReturnFalse",
				drainingPriority,
				func(h *fsTestHarness) { h.fs.drainingQueuesEmptyOnShards[drainingPriority]["s3-draining"] = false },
				false,
			},
			{"ForActivePriority_ShouldReturnFalse", activePriority, nil, false},
			{"ForUnknownPriority_ShouldReturnFalse", 99, nil, false},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				h := newFsTestHarness(t)
				h.fs.update(types.FlowSpecification{ID: "f1", Priority: pLow}, h.allShards)
				if tc.setup != nil {
					tc.setup(h)
				}
				assert.Equal(t, tc.expectDrained, h.fs.isDrained(tc.priorityToTest, h.allShards))
			})
		}
	})

	t.Run("PurgeShard", func(t *testing.T) {
		t.Parallel()
		h := newFsTestHarness(t)
		// Simulate a priority change to populate both active and draining maps.
		h.fs.update(types.FlowSpecification{ID: "f1", Priority: pLow}, h.allShards)
		shardToPurge := "s2"
		require.Contains(t, h.fs.activeQueueEmptyOnShards, shardToPurge, "Test setup: s2 must be in active map")
		require.Contains(t, h.fs.drainingQueuesEmptyOnShards[pHigh], shardToPurge, "Test setup: s2 must be in draining map")

		h.fs.purgeShard(shardToPurge)

		assert.NotContains(t, h.fs.activeQueueEmptyOnShards, shardToPurge, "s2 should be purged from active map")
		assert.Contains(t, h.fs.activeQueueEmptyOnShards, "s1", "s1 should remain in active map")
		assert.NotContains(t, h.fs.drainingQueuesEmptyOnShards[pHigh], shardToPurge, "s2 should be purged from draining map")
		assert.Contains(t, h.fs.drainingQueuesEmptyOnShards[pHigh], "s1", "s1 should remain in draining map")
	})
}
