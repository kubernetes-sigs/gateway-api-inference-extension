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
	"sync"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	inter "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/interflow/dispatch"
	intra "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

// --- Test Harness and Mocks ---

// shardTestHarness holds the components needed for a `registryShard` test.
type shardTestHarness struct {
	t               *testing.T
	globalConfig    *Config
	shard           *registryShard
	shardSignaler   *mockShardSignalRecorder
	statsPropagator *mockStatsPropagator
	callbacks       shardCallbacks // Callbacks to be passed to `newShard`
}

// newShardTestHarness creates a new test harness for testing the `registryShard`.
// It correctly simulates the parent registry's behavior by creating a global `Config`, partitioning it, and then
// creating a shard from the partitioned `ShardConfig`.
func newShardTestHarness(t *testing.T) *shardTestHarness {
	t.Helper()

	globalConfig, err := NewConfig(Config{
		PriorityBands: []PriorityBandConfig{
			{Priority: 10, PriorityName: "High"},
			{Priority: 20, PriorityName: "Low"},
		},
	})
	require.NoError(t, err, "Test setup: validating and defaulting config should not fail")

	shardSignaler := &mockShardSignalRecorder{}
	statsPropagator := &mockStatsPropagator{}
	callbacks := shardCallbacks{
		propagateStatsDelta: statsPropagator.propagate,
		// For most tests, we don't care about queue signals, so this is a no-op.
		signalQueueState: func(string, types.FlowKey, queueStateSignal) {},
		signalShardState: shardSignaler.signal,
	}

	// Partition the global config to create a shard-specific config.
	shardConfig := globalConfig.partition(0, 1)

	shard, err := newShard("test-shard-1", shardConfig, logr.Discard(), callbacks, inter.NewPolicyFromName)
	require.NoError(t, err, "Test setup: newShard should not return an error")

	return &shardTestHarness{
		t:               t,
		globalConfig:    globalConfig,
		shard:           shard,
		shardSignaler:   shardSignaler,
		statsPropagator: statsPropagator,
		callbacks:       callbacks,
	}
}

// synchronizeFlowWithMocks is a test helper that simulates the parent registry's logic for calling `synchronizeFlow` on
// the shard, but with mock plugins to ensure test isolation.
func (h *shardTestHarness) synchronizeFlowWithMocks(key types.FlowKey, q framework.SafeQueue) {
	h.t.Helper()
	spec := types.FlowSpecification{Key: key}
	mockPolicy := &frameworkmocks.MockIntraFlowDispatchPolicy{}
	h.shard.synchronizeFlow(spec, mockPolicy, q)
}

// synchronizeFlowWithRealQueue is a test helper that uses a real queue implementation.
// This is essential for integration and concurrency tests where the interaction between the shard and a real queue is
// being validated.
func (h *shardTestHarness) synchronizeFlowWithRealQueue(key types.FlowKey) {
	h.t.Helper()
	spec := types.FlowSpecification{Key: key}

	// This logic correctly mimics the parent registry's instantiation process.
	policy, err := intra.NewPolicyFromName(defaultIntraFlowDispatchPolicy)
	require.NoError(h.t, err, "Test setup: failed to create real intra-flow policy")
	q, err := queue.NewQueueFromName(defaultQueue, policy.Comparator())
	require.NoError(h.t, err, "Test setup: failed to create real queue")
	h.shard.synchronizeFlow(spec, policy, q)
}

// addItem adds an item to a specific flow on the shard, failing the test on any error.
func (h *shardTestHarness) addItem(key types.FlowKey, size uint64) types.QueueItemAccessor {
	h.t.Helper()
	mq, err := h.shard.ManagedQueue(key)
	require.NoError(h.t, err, "Helper addItem: failed to get queue for flow %v", key)
	item := mocks.NewMockQueueItemAccessor(size, "req", key)
	require.NoError(h.t, mq.Add(item), "Helper addItem: failed to add item to queue")
	return item
}

// removeItem removes an item from a specific flow, failing the test on any error.
func (h *shardTestHarness) removeItem(key types.FlowKey, item types.QueueItemAccessor) {
	h.t.Helper()
	mq, err := h.shard.ManagedQueue(key)
	require.NoError(h.t, err, "Helper removeItem: failed to get queue for flow %v", key)
	_, err = mq.Remove(item.Handle())
	require.NoError(h.t, err, "Helper removeItem: failed to remove item from queue")
}

// mockShardSignalRecorder is a thread-safe helper for recording shard state signals.
type mockShardSignalRecorder struct {
	mu      sync.Mutex
	signals []shardStateSignal
}

func (r *mockShardSignalRecorder) signal(_ string, signal shardStateSignal) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.signals = append(r.signals, signal)
}

func (r *mockShardSignalRecorder) getSignals() []shardStateSignal {
	r.mu.Lock()
	defer r.mu.Unlock()
	signalsCopy := make([]shardStateSignal, len(r.signals))
	copy(signalsCopy, r.signals)
	return signalsCopy
}

// --- Basic Tests ---

func TestShard_New(t *testing.T) {
	t.Parallel()
	h := newShardTestHarness(t)

	assert.Equal(t, "test-shard-1", h.shard.ID(), "ID should be set correctly")
	assert.True(t, h.shard.IsActive(), "A new shard should be active by default")
	assert.Equal(t, []uint{10, 20}, h.shard.AllOrderedPriorityLevels(), "Priority levels should be correctly ordered")

	bandHigh, ok := h.shard.priorityBands[10]
	require.True(t, ok, "High priority band should exist")
	assert.Equal(t, "High", bandHigh.config.PriorityName, "High priority band should have correct name")
	assert.NotNil(t, bandHigh.interFlowDispatchPolicy, "Inter-flow policy should be instantiated")
	assert.Equal(t, string(defaultInterFlowDispatchPolicy), bandHigh.interFlowDispatchPolicy.Name(),
		"Correct default inter-flow policy should be used")
}

func TestShard_New_ErrorPaths(t *testing.T) {
	t.Parallel()

	t.Run("ShouldFail_WhenInterFlowPolicyIsMissing", func(t *testing.T) {
		t.Parallel()
		shardConfig := &ShardConfig{
			PriorityBands: []ShardPriorityBandConfig{{Priority: 10, PriorityName: "High"}},
		}
		_, err := newShard("test-shard-1", shardConfig, logr.Discard(), shardCallbacks{}, inter.NewPolicyFromName)
		assert.Error(t, err, "newShard should fail when missing a configured inter-flow policy")
	})
}

func TestShard_Stats(t *testing.T) {
	t.Parallel()
	h := newShardTestHarness(t)
	flowKeyHigh := types.FlowKey{ID: "flow1", Priority: 10}
	h.synchronizeFlowWithMocks(flowKeyHigh, &frameworkmocks.MockSafeQueue{})
	h.addItem(flowKeyHigh, 100)
	h.addItem(flowKeyHigh, 50)

	stats := h.shard.Stats()

	assert.Equal(t, uint64(2), stats.TotalLen, "Total length should be 2")
	assert.Equal(t, uint64(150), stats.TotalByteSize, "Total byte size should be 150")

	bandHighStats := stats.PerPriorityBandStats[10]
	assert.Equal(t, uint64(2), bandHighStats.Len, "High priority band length should be 2")
	assert.Equal(t, uint64(150), bandHighStats.ByteSize, "High priority band byte size should be 150")

	bandLowStats := stats.PerPriorityBandStats[20]
	assert.Zero(t, bandLowStats.Len, "Low priority band length should be 0")
}

func TestShard_Accessors_SuccessPaths(t *testing.T) {
	t.Parallel()
	h := newShardTestHarness(t)
	flowKey := types.FlowKey{ID: "test-flow", Priority: 10}
	h.synchronizeFlowWithRealQueue(flowKey)

	t.Run("ManagedQueue_ShouldReturnCorrectInstance", func(t *testing.T) {
		t.Parallel()
		mq, err := h.shard.ManagedQueue(flowKey)

		require.NoError(t, err, "ManagedQueue should not error for an existing flow")
		require.NotNil(t, mq, "Returned ManagedQueue should not be nil")
		accessor := mq.FlowQueueAccessor()
		assert.Equal(t, flowKey, accessor.FlowKey(), "Returned queue should have the correct flow key")
	})

	t.Run("IntraFlowDispatchPolicy_ShouldReturnCorrectInstance", func(t *testing.T) {
		t.Parallel()
		policy, err := h.shard.IntraFlowDispatchPolicy(flowKey)

		require.NoError(t, err, "IntraFlowDispatchPolicy should not error for an existing flow")
		require.NotNil(t, policy, "Returned policy should not be nil")
		assert.Equal(t, string(defaultIntraFlowDispatchPolicy), policy.Name(),
			"Should return the correct default policy for the band")
	})

	t.Run("InterFlowDispatchPolicy_ShouldReturnCorrectInstance", func(t *testing.T) {
		t.Parallel()
		policy, err := h.shard.InterFlowDispatchPolicy(uint(10))

		require.NoError(t, err, "InterFlowDispatchPolicy should not error for an existing priority")
		require.NotNil(t, policy, "Returned policy should not be nil")
		assert.Equal(t, string(defaultInterFlowDispatchPolicy), policy.Name(),
			"Should return the correct default policy for the band")
	})
}

func TestShard_Accessors_ErrorPaths(t *testing.T) {
	t.Parallel()
	h := newShardTestHarness(t)
	flowKey := types.FlowKey{ID: "flow-a", Priority: 10}
	h.synchronizeFlowWithMocks(flowKey, &frameworkmocks.MockSafeQueue{})

	testCases := []struct {
		name      string
		action    func() error
		expectErr error
	}{
		{
			name:      "ManagedQueue_PriorityNotFound",
			action:    func() error { _, err := h.shard.ManagedQueue(types.FlowKey{ID: "flow-a", Priority: 99}); return err },
			expectErr: contracts.ErrPriorityBandNotFound,
		},
		{
			name:      "ManagedQueue_FlowNotFound",
			action:    func() error { _, err := h.shard.ManagedQueue(types.FlowKey{ID: "missing", Priority: 10}); return err },
			expectErr: contracts.ErrFlowInstanceNotFound,
		},
		{
			name:      "InterFlowDispatchPolicy_PriorityNotFound",
			action:    func() error { _, err := h.shard.InterFlowDispatchPolicy(99); return err },
			expectErr: contracts.ErrPriorityBandNotFound,
		},
		{
			name: "IntraFlowDispatchPolicy_PriorityNotFound",
			action: func() error {
				_, err := h.shard.IntraFlowDispatchPolicy(types.FlowKey{ID: "flow-a", Priority: 99})
				return err
			},
			expectErr: contracts.ErrPriorityBandNotFound,
		},
		{
			name: "IntraFlowDispatchPolicy_FlowNotFound",
			action: func() error {
				_, err := h.shard.IntraFlowDispatchPolicy(types.FlowKey{ID: "missing", Priority: 10})
				return err
			},
			expectErr: contracts.ErrFlowInstanceNotFound,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			err := tc.action()
			require.Error(t, err, "Action should have returned an error")
			assert.ErrorIs(t, err, tc.expectErr, "Error should wrap the expected sentinel error")
		})
	}
}

func TestShard_PriorityBandAccessor(t *testing.T) {
	t.Parallel()

	// The shard will have two flows in the "High" priority band and one in the "Low" band.
	h := newShardTestHarness(t)
	highPriorityKey1 := types.FlowKey{ID: "flow1", Priority: 10}
	highPriorityKey2 := types.FlowKey{ID: "flow2", Priority: 10}
	lowPriorityKey := types.FlowKey{ID: "flow3", Priority: 20}

	h.synchronizeFlowWithMocks(highPriorityKey1, &frameworkmocks.MockSafeQueue{})
	h.synchronizeFlowWithMocks(highPriorityKey2, &frameworkmocks.MockSafeQueue{})
	h.synchronizeFlowWithMocks(lowPriorityKey, &frameworkmocks.MockSafeQueue{})

	t.Run("ShouldFail_WhenPriorityDoesNotExist", func(t *testing.T) {
		t.Parallel()
		_, err := h.shard.PriorityBandAccessor(99)
		require.Error(t, err, "PriorityBandAccessor should fail for a non-existent priority")
		assert.ErrorIs(t, err, contracts.ErrPriorityBandNotFound, "Error should be ErrPriorityBandNotFound")
	})

	t.Run("ShouldSucceed_WhenPriorityExists", func(t *testing.T) {
		t.Parallel()
		accessor, err := h.shard.PriorityBandAccessor(10)
		require.NoError(t, err, "PriorityBandAccessor should not fail for an existing priority")
		require.NotNil(t, accessor, "Accessor should not be nil")

		t.Run("Properties_ShouldReturnCorrectValues", func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, uint(10), accessor.Priority(), "Accessor should have the correct priority")
			assert.Equal(t, "High", accessor.PriorityName(), "Accessor should have the correct priority name")
		})

		t.Run("FlowKeys_ShouldReturnAllKeysInBand", func(t *testing.T) {
			t.Parallel()
			keys := accessor.FlowKeys()
			expectedKeys := []types.FlowKey{highPriorityKey1, highPriorityKey2}
			assert.ElementsMatch(t, expectedKeys, keys,
				"Accessor should return all flow keys for the priority band")
		})

		t.Run("Queue_ShouldReturnCorrectAccessor", func(t *testing.T) {
			t.Parallel()
			q := accessor.Queue("flow1")
			require.NotNil(t, q, "Accessor should return a non-nil accessor for an existing flow")
			assert.Equal(t, highPriorityKey1, q.FlowKey(), "Queue accessor should have the correct flow key")
			assert.Nil(t, accessor.Queue("non-existent"), "Accessor should return nil for a non-existent flow")
		})

		t.Run("IterateQueues_ShouldVisitAllQueuesInBand", func(t *testing.T) {
			t.Parallel()
			var iteratedKeys []types.FlowKey
			accessor.IterateQueues(func(queue framework.FlowQueueAccessor) bool {
				iteratedKeys = append(iteratedKeys, queue.FlowKey())
				return true
			})
			expectedKeys := []types.FlowKey{highPriorityKey1, highPriorityKey2}
			assert.ElementsMatch(t, expectedKeys, iteratedKeys, "IterateQueues should visit all flows in the band")
		})

		t.Run("IterateQueues_ShouldExitEarly_WhenCallbackReturnsFalse", func(t *testing.T) {
			t.Parallel()
			var iterationCount int
			accessor.IterateQueues(func(queue framework.FlowQueueAccessor) bool {
				iterationCount++
				return false // Signal to stop iteration immediately.
			})
			assert.Equal(t, 1, iterationCount, "IterateQueues should exit after the first item")
		})
	})
}

// --- Lifecycle and State Management Tests ---

func TestShard_SynchronizeFlow(t *testing.T) {
	t.Parallel()
	h := newShardTestHarness(t)
	flowKey := types.FlowKey{ID: "flow1", Priority: 10}

	// Synchronization should create the queue.
	h.synchronizeFlowWithMocks(flowKey, &frameworkmocks.MockSafeQueue{})
	mq1, err := h.shard.ManagedQueue(flowKey)
	require.NoError(t, err, "Queue should exist after first synchronization")
	assert.Contains(t, h.shard.priorityBands[10].queues, flowKey.ID,
		"Queue should be in the correct priority band map")

	// Synchronization should be idempotent.
	h.synchronizeFlowWithMocks(flowKey, &frameworkmocks.MockSafeQueue{})
	mq2, err := h.shard.ManagedQueue(flowKey)
	require.NoError(t, err, "Queue should still exist after second synchronization")
	assert.Same(t, mq1, mq2, "Queue instance should not have been replaced on second sync")
}

func TestShard_GarbageCollect(t *testing.T) {
	t.Parallel()
	h := newShardTestHarness(t)
	flowKey := types.FlowKey{ID: "flow1", Priority: 10}
	h.synchronizeFlowWithMocks(flowKey, &frameworkmocks.MockSafeQueue{})
	require.Contains(t, h.shard.priorityBands[10].queues, flowKey.ID, "Test setup: queue must exist before GC")

	h.shard.garbageCollectLocked(flowKey)
	assert.NotContains(t, h.shard.priorityBands[10].queues, flowKey.ID,
		"Queue should have been removed from the priority band")
}

// --- Draining Lifecycle and Concurrency Tests ---

func TestShard_Lifecycle_Draining(t *testing.T) {
	t.Parallel()

	t.Run("ShouldTransitionToDraining_WhenMarkedWhileNonEmpty", func(t *testing.T) {
		t.Parallel()
		h := newShardTestHarness(t)
		flowKey := types.FlowKey{ID: "flow1", Priority: 10}
		h.synchronizeFlowWithMocks(flowKey, &frameworkmocks.MockSafeQueue{})
		h.addItem(flowKey, 100)

		h.shard.markAsDraining()
		assert.False(t, h.shard.IsActive(), "Shard should no longer be active")
		assert.Equal(t, componentStatusDraining, componentStatus(h.shard.status.Load()), "Shard status should be Draining")
	})

	t.Run("ShouldTransitionToDrainedAndSignal_WhenMarkedWhileEmpty", func(t *testing.T) {
		t.Parallel()
		h := newShardTestHarness(t)
		h.shard.markAsDraining()
		assert.Equal(t, componentStatusDrained, componentStatus(h.shard.status.Load()), "Shard status should be Drained")
		assert.Equal(t, []shardStateSignal{shardStateSignalBecameDrained}, h.shardSignaler.getSignals(),
			"Should have sent BecameDrained signal")
	})

	t.Run("ShouldTransitionToDrainedAndSignal_WhenLastItemIsRemoved", func(t *testing.T) {
		t.Parallel()
		h := newShardTestHarness(t)
		flowKey := types.FlowKey{ID: "flow1", Priority: 10}
		h.synchronizeFlowWithRealQueue(flowKey)
		item := h.addItem(flowKey, 100)

		h.shard.markAsDraining()
		require.Equal(t, componentStatusDraining, componentStatus(h.shard.status.Load()),
			"Shard should be Draining while it contains items")

		h.removeItem(flowKey, item)
		assert.Equal(t, componentStatusDrained, componentStatus(h.shard.status.Load()),
			"Shard should become Drained after last item is removed")
		assert.Len(t, h.shardSignaler.getSignals(), 1, "A signal should have been sent")
	})
}

// TestShard_Concurrency_DrainingRace targets the race between `markAsDraining()` and the shard becoming empty via
// concurrent item removals. It proves the atomic CAS correctly arbitrates the race and signals exactly once.
func TestShard_Concurrency_DrainingRace(t *testing.T) {
	t.Parallel()
	h := newShardTestHarness(t)
	flowKey := types.FlowKey{ID: "flow1", Priority: 10}
	h.synchronizeFlowWithRealQueue(flowKey)
	item := h.addItem(flowKey, 1)

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
		h.removeItem(flowKey, item)
	}()

	wg.Wait()

	// Verification: No matter which operation "won" the race, the final state must be Drained, and the signal must have
	// been sent exactly once.
	assert.Equal(t, componentStatusDrained, componentStatus(h.shard.status.Load()), "Final state must be Drained")
	assert.Len(t, h.shardSignaler.getSignals(), 1, "BecameDrained signal must be sent exactly once")
}

// --- Invariant Panic Tests ---

func TestShard_PanicOnCorruption(t *testing.T) {
	t.Parallel()

	t.Run("ShouldPanic_WhenStatsPropagatedForUnknownPriority", func(t *testing.T) {
		t.Parallel()
		h := newShardTestHarness(t)
		invalidPriority := uint(99)
		assert.Panics(t, func() {
			// This simulates a corrupted callback from a rogue `managedQueue`.
			h.shard.propagateStatsDelta(invalidPriority, 1, 1)
		}, "propagateStatsDelta must panic for an unknown priority")
	})

	t.Run("ShouldPanic_WhenUpdatingConfigWithMissingPriority", func(t *testing.T) {
		t.Parallel()
		h := newShardTestHarness(t)

		// Create a new config that is missing one of the shard's existing bands (priority 20).
		newGlobalConfig, _ := NewConfig(Config{
			PriorityBands: []PriorityBandConfig{{Priority: 10, PriorityName: "High-Updated"}},
		})
		newShardConfig := newGlobalConfig.partition(0, 1)

		assert.Panics(t, func() {
			h.shard.updateConfig(newShardConfig)
		}, "updateConfig must panic when an existing priority is missing from the new config")
	})

	t.Run("ShouldPanic_WhenSynchronizingFlowWithMissingPriority", func(t *testing.T) {
		t.Parallel()
		h := newShardTestHarness(t)
		assert.Panics(t, func() {
			h.shard.synchronizeFlow(
				types.FlowSpecification{Key: types.FlowKey{ID: "flow", Priority: 99}},
				&frameworkmocks.MockIntraFlowDispatchPolicy{},
				&frameworkmocks.MockSafeQueue{})
		}, "synchronizeFlow must panic for an unknown priority")
	})
}
