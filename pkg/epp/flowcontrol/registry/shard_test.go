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
	"errors"
	"fmt"
	"sort"
	"sync"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	inter "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/interflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/interflow/dispatch/besthead"
	intra "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch/fcfs"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue/listqueue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

// mockShardSignalRecorder is a thread-safe helper for recording shard state signals.
type mockShardSignalRecorder struct {
	mu      sync.Mutex
	signals []shardStateSignal
}

func (r *mockShardSignalRecorder) signal(shard *registryShard, signal shardStateSignal) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.signals = append(r.signals, signal)
}

func (r *mockShardSignalRecorder) getSignals() []shardStateSignal {
	r.mu.Lock()
	defer r.mu.Unlock()
	// Return a copy to prevent data races if the caller iterates over the slice while new signals are concurrently added
	// by the system under test.
	signalsCopy := make([]shardStateSignal, len(r.signals))
	copy(signalsCopy, r.signals)
	return signalsCopy
}

// shardTestHarness holds the components needed for a `registryShard` test.
type shardTestHarness struct {
	t               *testing.T
	config          *Config
	shard           *registryShard
	shardSignaler   *mockShardSignalRecorder
	statsPropagator *mockStatsPropagator
}

// newShardTestHarness creates a new test harness for testing the `registryShard`.
func newShardTestHarness(t *testing.T) *shardTestHarness {
	t.Helper()

	config := &Config{
		PriorityBands: []PriorityBandConfig{
			{Priority: pHigh, PriorityName: "High"},
			{Priority: pLow, PriorityName: "Low"},
		},
	}
	// Apply defaults to the master config first, as the parent registry would.
	err := config.validateAndApplyDefaults()
	require.NoError(t, err, "Test setup: validating and defaulting config should not fail")

	shardSignaler := &mockShardSignalRecorder{}
	statsPropagator := &mockStatsPropagator{}
	// The parent registry would partition the config. For a single shard test, we can use the defaulted one directly.
	callbacks := shardCallbacks{
		propagateStatsDelta: statsPropagator.propagate,
		signalQueueState:    func(string, types.FlowSpecification, queueStateSignal) {},
		signalShardState:    shardSignaler.signal,
	}
	shard, err := newShard("test-shard-1", config, logr.Discard(), callbacks)
	require.NoError(t, err, "Test setup: newShard should not return an error")
	require.NotNil(t, shard, "Test setup: newShard should return a non-nil shard")

	return &shardTestHarness{
		t:               t,
		config:          config,
		shard:           shard,
		shardSignaler:   shardSignaler,
		statsPropagator: statsPropagator,
	}
}

// synchronizeFlow is a test helper that simulates the parent registry's logic for instantiating plugins and reconciling
// a flow.
func (h *shardTestHarness) synchronizeFlow(spec types.FlowSpecification) {
	h.t.Helper()

	// Look up the configuration from the master config, as the `FlowRegistry` would.
	// We use the internal optimized lookup method populated during harness creation.
	bandConfig, err := h.config.getBandConfig(spec.Priority)
	// If `getBandConfig` fails, it means the priority band doesn't exist in the config.
	require.NoError(h.t, err, "Test setup: priority band %d should exist in master config", spec.Priority)

	policy, err := intra.NewPolicyFromName(bandConfig.IntraFlowDispatchPolicy)
	require.NoError(h.t, err, "Test setup: failed to create intra-flow policy")

	q, err := queue.NewQueueFromName(bandConfig.Queue, policy.Comparator())
	require.NoError(h.t, err, "Test setup: failed to create queue")

	h.shard.synchronizeFlow(spec, policy, q)
}

// addItem is a test helper to add an item to a specific flow on the shard.
func (h *shardTestHarness) addItem(flowID string, priority uint, size uint64) types.QueueItemAccessor {
	h.t.Helper()
	mq, err := h.shard.ManagedQueue(flowID, priority)
	require.NoError(h.t, err, "Helper addItem failed to get queue for flow %q at priority %d", flowID, priority)
	item := mocks.NewMockQueueItemAccessor(size, "req", flowID)
	require.NoError(h.t, mq.Add(item), "Helper addItem failed to add item to queue")
	return item
}

// removeItem is a test helper to remove an item from a specific flow's queue on the shard.
func (h *shardTestHarness) removeItem(flowID string, priority uint, item types.QueueItemAccessor) {
	h.t.Helper()
	mq, err := h.shard.ManagedQueue(flowID, priority)
	require.NoError(h.t, err, "Helper removeItem failed to get queue for flow %q at priority %d", flowID, priority)
	_, err = mq.Remove(item.Handle())
	require.NoError(h.t, err, "Helper removeItem failed to remove item from queue")
}

func TestShard(t *testing.T) {
	t.Parallel()

	t.Run("New_InitialState", func(t *testing.T) {
		t.Parallel()
		h := newShardTestHarness(t)

		assert.Equal(t, "test-shard-1", h.shard.ID(), "ID should be set correctly")
		assert.True(t, h.shard.IsActive(), "A new shard should be active")
		require.Len(t, h.shard.priorityBands, 2, "Should have 2 priority bands")

		// Check that priority levels are sorted correctly.
		assert.Equal(t, []uint{pHigh, pLow}, h.shard.AllOrderedPriorityLevels(), "Priority levels should be ordered")

		// Check band `pHigh`.
		bandHigh, ok := h.shard.priorityBands[pHigh]
		require.True(t, ok, "High priority band should exist")
		assert.Equal(t, pHigh, bandHigh.config.Priority, "High priority band should have correct priority")
		assert.Equal(t, "High", bandHigh.config.PriorityName, "High priority band should have correct name")
		assert.NotNil(t, bandHigh.interFlowDispatchPolicy,
			"Inter-flow policy for high priority band should be instantiated")
		assert.Equal(t, besthead.BestHeadPolicyName, bandHigh.interFlowDispatchPolicy.Name(),
			"Correct default inter-flow policy should be used")
		assert.Equal(t, string(fcfs.FCFSPolicyName), string(bandHigh.config.IntraFlowDispatchPolicy),
			"Correct default intra-flow policy should be used")

		// Check band `pLow`.
		bandLow, ok := h.shard.priorityBands[pLow]
		require.True(t, ok, "Low priority band should exist")
		assert.Equal(t, pLow, bandLow.config.Priority, "Low priority band should have correct priority")
		assert.Equal(t, "Low", bandLow.config.PriorityName, "Low priority band should have correct name")
		assert.NotNil(t, bandLow.interFlowDispatchPolicy, "Inter-flow policy for low priority band should be instantiated")
	})

	t.Run("Stats_Aggregation", func(t *testing.T) {
		t.Parallel()
		h := newShardTestHarness(t)

		// Add a queue and some items to test stats aggregation
		h.synchronizeFlow(types.FlowSpecification{ID: "flow1", Priority: pHigh})
		h.addItem("flow1", pHigh, 100)
		h.addItem("flow1", pHigh, 50)

		stats := h.shard.Stats()

		// Check shard-level stats
		assert.Equal(t, uint64(2), stats.TotalLen, "Total length should be 2")
		assert.Equal(t, uint64(150), stats.TotalByteSize, "Total byte size should be 150")

		// Check per-band stats
		require.Len(t, stats.PerPriorityBandStats, 2, "Should have stats for 2 bands")
		bandHighStats := stats.PerPriorityBandStats[pHigh]
		assert.Equal(t, pHigh, bandHighStats.Priority, "High priority band stats should have correct priority")
		assert.Equal(t, uint64(2), bandHighStats.Len, "High priority band length should be 2")
		assert.Equal(t, uint64(150), bandHighStats.ByteSize, "High priority band byte size should be 150")

		bandLowStats := stats.PerPriorityBandStats[pLow]
		assert.Equal(t, pLow, bandLowStats.Priority, "Low priority band stats should have correct priority")
		assert.Zero(t, bandLowStats.Len, "Low priority band length should be 0")
		assert.Zero(t, bandLowStats.ByteSize, "Low priority band byte size should be 0")
	})

	t.Run("Accessors", func(t *testing.T) {
		t.Parallel()

		t.Run("Scenarios", func(t *testing.T) {
			t.Parallel()
			h := newShardTestHarness(t)

			flowID := "test-flow"

			// Setup state with one active and one draining queue for the same flow.
			h.synchronizeFlow(types.FlowSpecification{ID: flowID, Priority: pHigh})
			h.addItem(flowID, pHigh, 1) // Add item so it doesn't immediately become Drained.
			h.synchronizeFlow(types.FlowSpecification{ID: flowID, Priority: pLow})

			// The second reconcile call makes the `pHigh` queue draining.
			activeQueue, err := h.shard.ActiveManagedQueue(flowID)
			require.NoError(t, err)
			drainingQueue, err := h.shard.ManagedQueue(flowID, pHigh)
			require.NoError(t, err)

			t.Run("ActiveManagedQueue_ReturnsCorrectQueue", func(t *testing.T) {
				t.Parallel()
				retrievedActiveQueue, err := h.shard.ActiveManagedQueue(flowID)
				require.NoError(t, err, "ActiveManagedQueue should not error for an existing flow")
				assert.Same(t, activeQueue, retrievedActiveQueue, "Should return the correct active queue")
				assert.Equal(t, pLow, retrievedActiveQueue.FlowQueueAccessor().FlowSpec().Priority,
					"Active queue should have the correct priority")

				_, err = h.shard.ActiveManagedQueue("non-existent-flow")
				require.Error(t, err, "ActiveManagedQueue should error for a non-existent flow")
				assert.ErrorIs(t, err, contracts.ErrFlowInstanceNotFound, "Error should be ErrFlowInstanceNotFound")
			})

			t.Run("ManagedQueue_ReturnsDrainingQueue", func(t *testing.T) {
				t.Parallel()
				retrievedDrainingQueue, err := h.shard.ManagedQueue(flowID, pHigh)
				require.NoError(t, err, "ManagedQueue should not error for a draining queue")
				assert.Same(t, drainingQueue, retrievedDrainingQueue, "Should return the correct draining queue")

				// Verify the retrieved queue is in a draining state.
				mq := retrievedDrainingQueue.(*managedQueue)
				status := componentStatus(mq.status.Load())
				assert.Equal(t, componentStatusDraining, status, "Retrieved queue should be in draining status")
			})

			t.Run("IntraFlowDispatchPolicy_ReturnsCorrectPolicy", func(t *testing.T) {
				t.Parallel()
				retrievedActivePolicy, err := h.shard.IntraFlowDispatchPolicy(flowID, pLow)
				require.NoError(t, err, "IntraFlowDispatchPolicy should not error for an active instance")
				assert.Same(t, activeQueue.(*managedQueue).dispatchPolicy, retrievedActivePolicy,
					"Should return the policy from the active instance")
			})

			t.Run("InterFlowDispatchPolicy_ReturnsCorrectPolicy", func(t *testing.T) {
				t.Parallel()
				retrievedInterPolicy, err := h.shard.InterFlowDispatchPolicy(pHigh)
				require.NoError(t, err, "InterFlowDispatchPolicy should not error for an existing priority")
				assert.Same(t, h.shard.priorityBands[pHigh].interFlowDispatchPolicy, retrievedInterPolicy,
					"Should return the correct inter-flow policy")
			})
		})

		t.Run("ErrorPaths", func(t *testing.T) {
			t.Parallel()
			h := newShardTestHarness(t)
			h.synchronizeFlow(types.FlowSpecification{ID: "flow-a", Priority: pHigh})

			testCases := []struct {
				name       string
				action     func() error
				expectErr  error
				errMessage string
			}{
				{
					name: "ManagedQueue_WhenPriorityNotFound_ShouldFail",
					action: func() error {
						_, err := h.shard.ManagedQueue("flow-a", 99)
						return err
					},
					expectErr:  contracts.ErrPriorityBandNotFound,
					errMessage: "ManagedQueue should error for a non-existent priority",
				},
				{
					name: "ManagedQueue_WhenFlowNotFound_ShouldFail",
					action: func() error {
						_, err := h.shard.ManagedQueue("non-existent-flow", pHigh)
						return err
					},
					expectErr:  contracts.ErrFlowInstanceNotFound,
					errMessage: "ManagedQueue should error for flow not in band",
				},
				{
					name: "IntraFlowDispatchPolicy_WhenPriorityNotFound_ShouldFail",
					action: func() error {
						_, err := h.shard.IntraFlowDispatchPolicy("flow-a", 99)
						return err
					},
					expectErr:  contracts.ErrPriorityBandNotFound,
					errMessage: "IntraFlowDispatchPolicy should error for non-existent priority",
				},
				{
					name: "IntraFlowDispatchPolicy_WhenFlowNotFound_ShouldFail",
					action: func() error {
						// flow-a exists at priority `pHigh`, but flow-b does not.
						_, err := h.shard.IntraFlowDispatchPolicy("flow-b", pHigh)
						return err
					},
					expectErr:  contracts.ErrFlowInstanceNotFound,
					errMessage: "IntraFlowDispatchPolicy should error for a flow not in the band",
				},
				{
					name: "InterFlowDispatchPolicy_WhenPriorityNotFound_ShouldFail",
					action: func() error {
						_, err := h.shard.InterFlowDispatchPolicy(99)
						return err
					},
					expectErr:  contracts.ErrPriorityBandNotFound,
					errMessage: "InterFlowDispatchPolicy should error for non-existent priority",
				},
			}

			for _, tc := range testCases {
				t.Run(tc.name, func(t *testing.T) {
					t.Parallel()
					err := tc.action()
					require.Error(t, err, tc.errMessage)
					assert.ErrorIs(t, err, tc.expectErr)
				})
			}
		})
	})

	t.Run("PriorityBandAccessor_Scenarios", func(t *testing.T) {
		t.Parallel()
		h := newShardTestHarness(t)

		// Setup shard state for the tests
		h.synchronizeFlow(types.FlowSpecification{ID: "flow1", Priority: pHigh})
		h.synchronizeFlow(types.FlowSpecification{ID: "flow1", Priority: pLow}) // `pHigh` is now draining
		h.synchronizeFlow(types.FlowSpecification{ID: "flow2", Priority: pHigh})

		t.Run("WhenPriorityExists_ShouldSucceed", func(t *testing.T) {
			t.Parallel()
			accessor, err := h.shard.PriorityBandAccessor(pHigh)
			require.NoError(t, err, "PriorityBandAccessor should not fail for existing priority")
			require.NotNil(t, accessor, "Accessor should not be nil")

			t.Run("Properties_ShouldReturnCorrectValues", func(t *testing.T) {
				t.Parallel()
				assert.Equal(t, pHigh, accessor.Priority(), "Accessor should have correct priority")
				assert.Equal(t, "High", accessor.PriorityName(), "Accessor should have correct priority name")
			})

			t.Run("FlowIDs_ShouldReturnAllFlowsInBand", func(t *testing.T) {
				t.Parallel()
				flowIDs := accessor.FlowIDs()
				sort.Strings(flowIDs)
				assert.Equal(t, []string{"flow1", "flow2"}, flowIDs,
					"Accessor should return correct flow IDs for the priority band")
			})

			t.Run("Queue_ShouldReturnCorrectAccessor", func(t *testing.T) {
				t.Parallel()
				q := accessor.Queue("flow1")
				require.NotNil(t, q, "Accessor should return queue for flow1")
				assert.Equal(t, pHigh, q.FlowSpec().Priority, "Queue should have the correct priority")
				assert.Nil(t, accessor.Queue("non-existent"), "Accessor should return nil for non-existent flow")
			})

			t.Run("IterateQueues_ShouldVisitAllQueues", func(t *testing.T) {
				t.Parallel()
				var iteratedFlows []string
				accessor.IterateQueues(func(queue framework.FlowQueueAccessor) bool {
					iteratedFlows = append(iteratedFlows, queue.FlowSpec().ID)
					return true
				})
				sort.Strings(iteratedFlows)
				assert.Equal(t, []string{"flow1", "flow2"}, iteratedFlows, "IterateQueues should visit all flows in the band")
			})

			t.Run("IterateQueues_ShouldExitEarly", func(t *testing.T) {
				t.Parallel()
				var iteratedFlows []string
				accessor.IterateQueues(func(queue framework.FlowQueueAccessor) bool {
					iteratedFlows = append(iteratedFlows, queue.FlowSpec().ID)
					return false // Exit after first item
				})
				assert.Len(t, iteratedFlows, 1, "IterateQueues should exit early if callback returns false")
			})
		})

		t.Run("WhenPriorityDoesNotExist_ShouldFail", func(t *testing.T) {
			t.Parallel()
			_, err := h.shard.PriorityBandAccessor(99)
			require.Error(t, err, "PriorityBandAccessor should fail for non-existent priority")
			assert.ErrorIs(t, err, contracts.ErrPriorityBandNotFound, "Error should be ErrPriorityBandNotFound")
		})
	})

	t.Run("Lifecycle", func(t *testing.T) {
		t.Parallel()

		t.Run("SynchronizeFlow", func(t *testing.T) {
			t.Parallel()
			const flowID = "test-flow"
			specHigh := types.FlowSpecification{ID: flowID, Priority: pHigh}
			specLow := types.FlowSpecification{ID: flowID, Priority: pLow}

			t.Run("ForNewFlow_ShouldCreateActiveQueue", func(t *testing.T) {
				t.Parallel()
				h := newShardTestHarness(t)
				h.synchronizeFlow(specHigh)

				// Assert state.
				assert.Contains(t, h.shard.activeFlows, flowID, "Flow should be in the active map")
				activeQueue := h.shard.activeFlows[flowID]
				assert.Equal(t, pHigh, activeQueue.flowSpec.Priority, "Active flow should have the correct priority")
				assert.Equal(t, componentStatusActive, componentStatus(activeQueue.status.Load()), "Queue should be active")

				band, ok := h.shard.priorityBands[pHigh]
				require.True(t, ok, "High priority band should exist")
				assert.Contains(t, band.queues, flowID, "Queue should be in the correct priority band map")
			})

			t.Run("ForExistingFlowAtSamePriority_ShouldBeNoOp", func(t *testing.T) {
				t.Parallel()
				h := newShardTestHarness(t)
				// Initial synchronization.
				h.synchronizeFlow(specHigh)
				initialQueue := h.shard.activeFlows[flowID]

				// Second synchronization with the same spec,
				h.synchronizeFlow(specHigh)

				// Assert state is unchanged.
				assert.Len(t, h.shard.activeFlows, 1, "There should still be only one active flow")
				assert.Same(t, initialQueue, h.shard.activeFlows[flowID], "The queue instance should not have been replaced")
			})

			t.Run("ForPriorityChange_ShouldDrainOldQueue", func(t *testing.T) {
				t.Parallel()
				h := newShardTestHarness(t)
				h.synchronizeFlow(specHigh)
				h.addItem(flowID, pHigh, 1) // Add item so the queue doesn't immediately become Drained.
				oldQueue := h.shard.activeFlows[flowID]

				// Synchronize with the new, lower priority.
				h.synchronizeFlow(specLow)

				// Assert state of the new active queue.
				assert.Contains(t, h.shard.activeFlows, flowID, "Flow should still be active")
				newActiveQueue := h.shard.activeFlows[flowID]
				assert.Equal(t, pLow, newActiveQueue.flowSpec.Priority, "Active flow should now have the low priority")
				assert.Equal(t, componentStatusActive, componentStatus(newActiveQueue.status.Load()),
					"New queue should be active")

				// Assert state of the old, now-draining queue.
				assert.Contains(t, h.shard.priorityBands[pHigh].queues, flowID,
					"Old queue should still exist in the high priority band")
				assert.Equal(t, componentStatusDraining, componentStatus(oldQueue.status.Load()),
					"Old queue should be marked as draining")
			})

			t.Run("ForPriorityRollback_ShouldReactivateDrainingQueue", func(t *testing.T) {
				t.Parallel()
				h := newShardTestHarness(t)
				// Step 1: Create active queue at `pHigh`.
				h.synchronizeFlow(specHigh)
				drainingQueueCandidate := h.shard.activeFlows[flowID]
				h.addItem(flowID, pHigh, 1)

				// Step 2: Change priority to `pLow`, making the `pHigh` queue drain.
				h.synchronizeFlow(specLow)
				assert.Equal(t, componentStatusDraining, componentStatus(drainingQueueCandidate.status.Load()),
					"Queue at pHigh should be draining")

				// Step 3: Roll back to `pHigh`.
				h.synchronizeFlow(specHigh)

				// Assert that the original queue was reactivated
				assert.Len(t, h.shard.activeFlows, 1, "There should be one active flow")
				assert.Same(t, drainingQueueCandidate, h.shard.activeFlows[flowID],
					"The original queue should have been reactivated")
				assert.Equal(t, componentStatusActive, componentStatus(h.shard.activeFlows[flowID].status.Load()),
					"The reactivated queue should be marked as active")

				// Assert that the pLow queue is now draining.
				lowPriorityBand := h.shard.priorityBands[pLow]
				assert.Contains(t, lowPriorityBand.queues, flowID, "The pLow queue should now be in the low priority band")
				lowQueue := lowPriorityBand.queues[flowID]
				// Since the low-priority queue was empty when the rollback happened, it should be immediately Drained.
				assert.Equal(t, componentStatusDrained, componentStatus(lowQueue.status.Load()),
					"The pLow queue should now be drained")
			})

			t.Run("GarbageCollect_ShouldRemoveDrainingQueue", func(t *testing.T) {
				t.Parallel()
				h := newShardTestHarness(t)
				h.synchronizeFlow(specHigh)
				h.synchronizeFlow(specLow) // `specHigh` is now draining

				h.shard.garbageCollect(flowID, pHigh)

				assert.NotContains(t, h.shard.priorityBands[pHigh].queues, flowID,
					"Queue should have been removed from the priority band")
			})

			t.Run("GarbageCollect_ShouldRemoveActiveQueue", func(t *testing.T) {
				t.Parallel()
				h := newShardTestHarness(t)
				h.synchronizeFlow(specHigh)

				// Verify the queue is active before GC.
				require.Contains(t, h.shard.activeFlows, flowID, "Test setup: queue must be in active map before GC")

				h.shard.garbageCollect(flowID, pHigh)

				assert.NotContains(t, h.shard.priorityBands[pHigh].queues, flowID,
					"Queue should have been removed from the priority band map")
				assert.NotContains(t, h.shard.activeFlows, flowID, "Queue should have been removed from the active flows map")
			})
		})

		t.Run("DrainingTransitions", func(t *testing.T) {
			t.Parallel()

			t.Run("MarkAsDraining_OnNonEmptyShard_ShouldTransitionToDraining", func(t *testing.T) {
				t.Parallel()
				h := newShardTestHarness(t)
				h.synchronizeFlow(types.FlowSpecification{ID: "flow1", Priority: pHigh})
				h.synchronizeFlow(types.FlowSpecification{ID: "flow2", Priority: pLow})

				// Add items to make queues non-empty.
				h.addItem("flow1", pHigh, 1)
				h.addItem("flow2", pLow, 1)

				// Mark the shard as draining.
				h.shard.markAsDraining()

				// Assert shard status.
				assert.False(t, h.shard.IsActive(), "Shard should no longer be active")
				assert.Equal(t, componentStatusDraining, componentStatus(h.shard.status.Load()),
					"Shard status should be Draining")

				// Assert status of constituent queues
				mq1, _ := h.shard.ManagedQueue("flow1", pHigh)
				mq2, _ := h.shard.ManagedQueue("flow2", pLow)
				assert.Equal(t, componentStatusDraining, componentStatus(mq1.(*managedQueue).status.Load()),
					"Queue for flow1 should be draining")
				assert.Equal(t, componentStatusDraining, componentStatus(mq2.(*managedQueue).status.Load()),
					"Queue for flow2 should be draining")
			})

			t.Run("MarkAsDraining_OnEmptyShard_ShouldTransitionToDrainedAndSignal", func(t *testing.T) {
				t.Parallel()
				h := newShardTestHarness(t)

				// Mark the empty shard as draining.
				h.shard.markAsDraining()

				// Assert status and signal.
				assert.Equal(t, componentStatusDrained, componentStatus(h.shard.status.Load()),
					"Shard status should be Drained")
				require.Len(t, h.shardSignaler.getSignals(), 1, "A signal should have been sent")
				assert.Equal(t, shardStateSignalBecameDrained, h.shardSignaler.signals[0],
					"The correct signal should have been sent")
			})

			t.Run("WhenDraining_ShouldTransitionToDrainedWhenLastItemIsRemoved", func(t *testing.T) {
				t.Parallel()
				h := newShardTestHarness(t)
				h.synchronizeFlow(types.FlowSpecification{ID: "flow1", Priority: pHigh})
				item1 := h.addItem("flow1", pHigh, 100)
				item2 := h.addItem("flow1", pHigh, 200)

				// Mark the shard as draining; it should be Draining, not Drained, since it's not empty.
				h.shard.markAsDraining()
				assert.Equal(t, componentStatusDraining, componentStatus(h.shard.status.Load()),
					"Shard should be Draining while it contains items")
				assert.Empty(t, h.shardSignaler.getSignals(),
					"No signal should be sent while the shard is still draining with items")

				// Remove one item; it should still be Draining.
				h.removeItem("flow1", pHigh, item1)
				assert.Equal(t, componentStatusDraining, componentStatus(h.shard.status.Load()),
					"Shard should remain Draining after one item is removed")
				assert.Empty(t, h.shardSignaler.getSignals(), "No signal should be sent yet")

				// Remove the final item; it should now transition to Drained and signal.
				h.removeItem("flow1", pHigh, item2)
				assert.Equal(t, componentStatusDrained, componentStatus(h.shard.status.Load()),
					"Shard should become Drained after the last item is removed")
				require.Len(t, h.shardSignaler.getSignals(), 1, "A signal should have been sent upon becoming empty")
				assert.Equal(t, shardStateSignalBecameDrained, h.shardSignaler.signals[0],
					"The correct drained signal should have been sent")
			})
		})

		// DrainingRaceWithConcurrentRemovals targets the race where multiple goroutines remove the last few items from a
		// draining shard. It ensures the Draining -> Drained transition happens exactly once.
		t.Run("DrainingRaceWithConcurrentRemovals", func(t *testing.T) {
			t.Parallel()
			h := newShardTestHarness(t)

			numItems := 50
			var items []types.QueueItemAccessor
			for i := range numItems {
				// Add items to two different flows to make it more realistic.
				flowID := fmt.Sprintf("flow-%d", i%2)
				h.synchronizeFlow(types.FlowSpecification{ID: flowID, Priority: pHigh})
				item := h.addItem(flowID, pHigh, 10)
				items = append(items, item)
			}

			// Mark the shard as draining. It has items, so it will enter the Draining state.
			h.shard.markAsDraining()
			require.Equal(t, componentStatusDraining, componentStatus(h.shard.status.Load()),
				"Test setup: shard must be in Draining state")

			var wg sync.WaitGroup
			wg.Add(numItems)
			for _, item := range items {
				go func(it types.QueueItemAccessor) {
					defer wg.Done()
					h.removeItem(it.OriginalRequest().FlowID(), pHigh, it)
				}(item)
			}
			wg.Wait()

			// Verification:
			// No matter which goroutine removed the "last" item, the final state must be Drained, and the signal must have
			// been sent exactly once.
			assert.Equal(t, componentStatusDrained, componentStatus(h.shard.status.Load()), "Final state must be Drained")
			assert.Len(t, h.shardSignaler.getSignals(), 1, "BecameDrained signal must be sent exactly once")
			assert.Equal(t, shardStateSignalBecameDrained, h.shardSignaler.signals[0], "The correct signal must be sent")
		})

		// DrainingRaceWithMarkAndEmpty targets the race between markAsDraining() and the shard becoming empty via a
		// concurrent item removal. It proves the atomic CAS correctly arbitrates the race.
		t.Run("DrainingRaceWithMarkAndEmpty", func(t *testing.T) {
			t.Parallel()
			h := newShardTestHarness(t)

			// Test setup: Start with a shard containing a single item.
			h.synchronizeFlow(types.FlowSpecification{ID: "flow1", Priority: pHigh})
			item := h.addItem("flow1", pHigh, 1)

			var wg sync.WaitGroup
			wg.Add(2)

			// Goroutine 1: Attempts to mark the shard as draining.
			go func() {
				defer wg.Done()
				h.shard.markAsDraining()
			}()

			// Goroutine 2: Concurrently removes the single item.
			go func() {
				defer wg.Done()
				h.removeItem("flow1", pHigh, item)
			}()

			wg.Wait()

			// Verification:
			// Either markAsDraining found an empty queue, or propagateStatsDelta found a draining queue.
			// In either case, the final state must be Drained and the signal must have been sent exactly once.
			assert.Equal(t, componentStatusDrained, componentStatus(h.shard.status.Load()),
				"Final state must be Drained regardless of race outcome")
			assert.Len(t, h.shardSignaler.getSignals(), 1, "BecameDrained signal must be sent exactly once")
		})
	})

	t.Run("UpdateConfig_UpdatesInternalState", func(t *testing.T) {
		t.Parallel()
		h := newShardTestHarness(t)
		h.synchronizeFlow(types.FlowSpecification{ID: "flow1", Priority: pHigh})

		// Create a new config with different values.
		newConfig := h.config.deepCopy()
		newConfig.MaxBytes = 9999
		newConfig.PriorityBands[0].MaxBytes = 8888 // for priority `pHigh`

		// Update the shard's config.
		h.shard.updateConfig(newConfig)

		// Assert that the shard's internal config pointer was updated.
		assert.Same(t, newConfig, h.shard.config, "Shard's internal config pointer should be updated")

		// Assert that the stats reflect the new capacity.
		stats := h.shard.Stats()
		assert.Equal(t, uint64(9999), stats.TotalCapacityBytes, "Shard's total capacity should be updated in stats")
		bandHighStats, ok := stats.PerPriorityBandStats[pHigh]
		require.True(t, ok, "Stats for high priority band should exist")
		assert.Equal(t, uint64(8888), bandHighStats.CapacityBytes, "Priority band's capacity should be updated in stats")

		// Assert that the config within the internal `priorityBand` struct was also updated.
		bandHigh, ok := h.shard.priorityBands[pHigh]
		require.True(t, ok, "Internal high priority band struct should exist")
		assert.Equal(t, uint64(8888), bandHigh.config.MaxBytes, "Internal priority band's config should be updated")
	})

	// TestShard_Invariants_PanicOnCorruption tests conditions that should cause a panic, as they represent a corrupted or
	// inconsistent state that cannot be recovered from.
	t.Run("Invariants_PanicOnCorruption", func(t *testing.T) {
		t.Parallel()

		t.Run("PropagateStatsDelta_WithUnknownPriority_ShouldPanic", func(t *testing.T) {
			t.Parallel()
			h := newShardTestHarness(t)
			// Manually create a managedQueue with a priority that does not exist in the shard's config.
			// This simulates a state corruption where an invalid queue instance is somehow created.
			invalidPriority := uint(99)
			spec := types.FlowSpecification{ID: "bad-flow", Priority: invalidPriority}
			q, _ := queue.NewQueueFromName(listqueue.ListQueueName, nil)
			policy, _ := intra.NewPolicyFromName(fcfs.FCFSPolicyName)
			callbacks := managedQueueCallbacks{
				propagateStatsDelta: h.shard.propagateStatsDelta, // Use the real shard's propagator.
				signalQueueState:    func(spec types.FlowSpecification, signal queueStateSignal) {},
			}
			mq := newManagedQueue(q, policy, spec, logr.Discard(), callbacks)

			// The call to propagateStatsDelta is what should panic.
			// This is triggered by calling `Add()` on the manually created queue.
			expectedPanicMsg := fmt.Sprintf("invariant violation: received stats propagation for unknown priority band (%d)",
				invalidPriority)
			assert.PanicsWithValue(t,
				expectedPanicMsg,
				func() {
					// This call will trigger the panic inside the shard's callback.
					_ = mq.Add(mocks.NewMockQueueItemAccessor(1, "req", "bad-flow"))
				},
				"propagateStatsDelta must panic when called with a priority that doesn't exist on the shard",
			)
		})

		t.Run("UpdateConfig_WithMissingPriorityBand_ShouldPanic", func(t *testing.T) {
			t.Parallel()
			h := newShardTestHarness(t) // This harness has priorities `pHigh` and `pLow`.

			// Create a new config that is missing one of the shard's existing priority bands (priority `pLow` is missing).
			newConfig := &Config{
				PriorityBands: []PriorityBandConfig{
					{Priority: pHigh, PriorityName: "High-Updated"},
				},
			}
			err := newConfig.validateAndApplyDefaults()
			require.NoError(t, err, "Test setup: creating updated config should not fail")

			// This call should panic because the shard has a band for `pLow`, but the new config does not.
			// The full error string is complex due to error wrapping.
			expectedErrStr := "invariant violation: priority band (30) missing in new configuration during update: config for priority 30 not found: priority band not found"
			assert.PanicsWithError(t,
				expectedErrStr,
				func() {
					h.shard.updateConfig(newConfig)
				},
				"updateConfig must panic when an existing priority band is missing from the new config",
			)
		})
	})
}

// TestShard_New_ErrorPaths modifies a global plugin registry, so it cannot be run in parallel with other tests that
// might also manipulate the same global state.
func TestShard_New_ErrorPaths(t *testing.T) {
	baseConfig := &Config{
		PriorityBands: []PriorityBandConfig{{
			Priority:                pHigh,
			PriorityName:            "High",
			IntraFlowDispatchPolicy: fcfs.FCFSPolicyName,
			InterFlowDispatchPolicy: besthead.BestHeadPolicyName,
			Queue:                   listqueue.ListQueueName,
		}},
	}
	require.NoError(t, baseConfig.validateAndApplyDefaults(), "Test setup: base config should be valid")

	t.Run("WhenInterFlowPolicyIsInvalid_ShouldFail", func(t *testing.T) {
		failingPolicyName := inter.RegisteredPolicyName(fmt.Sprintf("failing-inter-policy-%s", t.Name()))
		inter.MustRegisterPolicy(failingPolicyName, func() (framework.InterFlowDispatchPolicy, error) {
			return nil, errors.New("inter-flow instantiation failed")
		})

		badConfig := baseConfig.deepCopy()
		badConfig.PriorityBands[0].InterFlowDispatchPolicy = failingPolicyName

		_, err := newShard("test", badConfig, logr.Discard(), shardCallbacks{})
		require.Error(t, err, "newShard should fail with an invalid inter-flow policy")
	})
}
