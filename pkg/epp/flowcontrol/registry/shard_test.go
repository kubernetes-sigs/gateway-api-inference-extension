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
	"sync"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/interflow"
	intra "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

const (
	// highPriority is the priority level for the "High" priority band in the test harness config.
	highPriority int = 20
	// lowPriority is the priority level for the "Low" priority band in the test harness config.
	lowPriority int = 10
	// nonExistentPriority is a priority that is known not to exist in the test harness config.
	nonExistentPriority int = 99
)

// --- Test Harness and Mocks ---

// shardTestHarness holds all components for a `registryShard` test.
type shardTestHarness struct {
	t                *testing.T
	shard            *registryShard
	statsPropagator  *mockStatsPropagator
	highPriorityKey1 types.FlowKey
	highPriorityKey2 types.FlowKey
	lowPriorityKey   types.FlowKey
}

// newShardTestHarness initializes a `shardTestHarness` with a default configuration.
func newShardTestHarness(t *testing.T) *shardTestHarness {
	t.Helper()
	globalConfig, err := newConfig(Config{
		PriorityBands: []PriorityBandConfig{
			{Priority: highPriority, PriorityName: "High"},
			{Priority: lowPriority, PriorityName: "Low"},
		},
	})
	require.NoError(t, err, "Test setup: validating and defaulting config should not fail")

	statsPropagator := &mockStatsPropagator{}
	shardConfig := globalConfig.partition(0, 1)
	shard, err := newShard(
		"test-shard-1",
		shardConfig, logr.Discard(),
		statsPropagator.propagate,
		interflow.NewPolicyFromName,
	)
	require.NoError(t, err, "Test setup: newShard should not return an error with valid configuration")

	h := &shardTestHarness{
		t:                t,
		shard:            shard,
		statsPropagator:  statsPropagator,
		highPriorityKey1: types.FlowKey{ID: "hp-flow-1", Priority: highPriority},
		highPriorityKey2: types.FlowKey{ID: "hp-flow-2", Priority: highPriority},
		lowPriorityKey:   types.FlowKey{ID: "lp-flow-1", Priority: lowPriority},
	}
	// Automatically sync some default flows for convenience.
	h.synchronizeFlow(h.highPriorityKey1)
	h.synchronizeFlow(h.highPriorityKey2)
	h.synchronizeFlow(h.lowPriorityKey)
	return h
}

// synchronizeFlow simulates the registry synchronizing a flow with a real queue.
func (h *shardTestHarness) synchronizeFlow(key types.FlowKey) {
	h.t.Helper()
	spec := types.FlowSpecification{Key: key}
	policy, err := intra.NewPolicyFromName(defaultIntraFlowDispatchPolicy)
	require.NoError(h.t, err, "Helper synchronizeFlow: failed to create real intra-flow policy for synchronization")
	q, err := queue.NewQueueFromName(defaultQueue, policy.Comparator())
	require.NoError(h.t, err, "Helper synchronizeFlow: failed to create real queue for synchronization")
	h.shard.synchronizeFlow(spec, policy, q)
}

// addItem adds an item to a specific flow's queue on the shard.
func (h *shardTestHarness) addItem(key types.FlowKey, size uint64) types.QueueItemAccessor {
	h.t.Helper()
	mq, err := h.shard.ManagedQueue(key)
	require.NoError(h.t, err, "Helper addItem: failed to get queue for flow %s; ensure flow is synchronized", key)
	item := mocks.NewMockQueueItemAccessor(size, "req", key)
	require.NoError(h.t, mq.Add(item), "Helper addItem: failed to add item to queue for flow %s", key)
	return item
}

// removeItem removes an item from a specific flow's queue.
func (h *shardTestHarness) removeItem(key types.FlowKey, item types.QueueItemAccessor) {
	h.t.Helper()
	mq, err := h.shard.ManagedQueue(key)
	require.NoError(h.t, err, "Helper removeItem: failed to get queue for flow %s; ensure flow is synchronized", key)
	_, err = mq.Remove(item.Handle())
	require.NoError(h.t, err, "Helper removeItem: failed to remove item from queue for flow %s", key)
}

// --- Basic Tests ---

func TestShard_New(t *testing.T) {
	t.Parallel()

	t.Run("ShouldInitializeCorrectly_WithDefaultConfig", func(t *testing.T) {
		t.Parallel()
		h := newShardTestHarness(t)

		assert.Equal(t, "test-shard-1", h.shard.ID(), "Shard ID must match the value provided during construction")
		assert.True(t, h.shard.IsActive(), "A newly created shard must be initialized in the Active state")
		assert.Equal(t, []int{highPriority, lowPriority}, h.shard.AllOrderedPriorityLevels(),
			"Shard must report configured priority levels sorted numerically (highest priority first)")

		bandHigh, ok := h.shard.priorityBands[highPriority]
		require.True(t, ok, "Priority band %d (High) must be initialized", highPriority)
		assert.Equal(t, "High", bandHigh.config.PriorityName, "Priority band name must match the configuration")
		require.NotNil(t, bandHigh.interFlowDispatchPolicy, "Inter-flow policy must be instantiated during construction")
		assert.Equal(t, string(defaultInterFlowDispatchPolicy), bandHigh.interFlowDispatchPolicy.Name(),
			"The default inter-flow policy implementation must be used when not overridden")
	})

	t.Run("ShouldFail_WhenInterFlowPolicyFactoryFails", func(t *testing.T) {
		t.Parallel()
		shardConfig, _ := newConfig(Config{PriorityBands: []PriorityBandConfig{
			{Priority: highPriority, PriorityName: "High"},
		}})
		failingFactory := func(interflow.RegisteredPolicyName) (framework.InterFlowDispatchPolicy, error) {
			return nil, errors.New("policy not found")
		}
		_, err := newShard("test-shard-1", shardConfig.partition(0, 1), logr.Discard(), nil, failingFactory)
		require.Error(t, err, "newShard must fail if the inter-flow policy cannot be instantiated during initialization")
	})
}

func TestShard_Stats(t *testing.T) {
	t.Parallel()
	h := newShardTestHarness(t)
	h.addItem(h.highPriorityKey1, 100)
	h.addItem(h.highPriorityKey1, 50)

	stats := h.shard.Stats()

	assert.Equal(t, h.shard.ID(), stats.ID, "Stats ID must match the shard ID")
	assert.True(t, stats.IsActive, "Shard must report itself as active in the stats snapshot")
	assert.Equal(t, uint64(2), stats.TotalLen, "Total shard length must aggregate counts from all bands")
	assert.Equal(t, uint64(150), stats.TotalByteSize, "Total shard byte size must aggregate sizes from all bands")

	bandHighStats, ok := stats.PerPriorityBandStats[highPriority]
	require.True(t, ok, "Stats snapshot must include entries for all configured priority bands (e.g., %d)", highPriority)
	assert.Equal(t, uint64(2), bandHighStats.Len, "Priority band length must reflect the items queued at that level")
	assert.Equal(t, uint64(150), bandHighStats.ByteSize,
		"Priority band byte size must reflect the items queued at that level")
}

func TestShard_Accessors(t *testing.T) {
	t.Parallel()

	t.Run("SuccessPaths", func(t *testing.T) {
		t.Parallel()
		h := newShardTestHarness(t)

		t.Run("ManagedQueue", func(t *testing.T) {
			t.Parallel()
			mq, err := h.shard.ManagedQueue(h.highPriorityKey1)
			require.NoError(t, err, "ManagedQueue accessor must succeed for a synchronized flow")
			require.NotNil(t, mq, "Returned ManagedQueue must not be nil")
			assert.Equal(t, h.highPriorityKey1, mq.FlowQueueAccessor().FlowKey(),
				"The returned queue instance must correspond to the requested FlowKey")
		})

		t.Run("IntraFlowDispatchPolicy", func(t *testing.T) {
			t.Parallel()
			policy, err := h.shard.IntraFlowDispatchPolicy(h.highPriorityKey1)
			require.NoError(t, err, "IntraFlowDispatchPolicy accessor must succeed for a synchronized flow")
			require.NotNil(t, policy, "Returned policy must not be nil (guaranteed by contract)")
			assert.Equal(t, string(defaultIntraFlowDispatchPolicy), policy.Name(),
				"Must return the default intra-flow policy implementation")
		})

		t.Run("InterFlowDispatchPolicy", func(t *testing.T) {
			t.Parallel()
			policy, err := h.shard.InterFlowDispatchPolicy(highPriority)
			require.NoError(t, err, "InterFlowDispatchPolicy accessor must succeed for a configured priority band")
			require.NotNil(t, policy, "Returned policy must not be nil (guaranteed by contract)")
			assert.Equal(t, string(defaultInterFlowDispatchPolicy), policy.Name(),
				"Must return the default inter-flow policy implementation")
		})
	})

	t.Run("ErrorPaths", func(t *testing.T) {
		t.Parallel()
		testCases := []struct {
			name      string
			action    func(s *registryShard) error
			expectErr error
		}{
			{
				name: "ManagedQueue_PriorityNotFound",
				action: func(s *registryShard) error {
					_, err := s.ManagedQueue(types.FlowKey{Priority: nonExistentPriority})
					return err
				},
				expectErr: contracts.ErrPriorityBandNotFound,
			},
			{
				name: "ManagedQueue_FlowNotFound",
				action: func(s *registryShard) error {
					_, err := s.ManagedQueue(types.FlowKey{ID: "missing", Priority: highPriority})
					return err
				},
				expectErr: contracts.ErrFlowInstanceNotFound,
			},
			{
				name: "InterFlowDispatchPolicy_PriorityNotFound",
				action: func(s *registryShard) error {
					_, err := s.InterFlowDispatchPolicy(nonExistentPriority)
					return err
				},
				expectErr: contracts.ErrPriorityBandNotFound,
			},
			{
				name: "IntraFlowDispatchPolicy_PriorityNotFound",
				action: func(s *registryShard) error {
					_, err := s.IntraFlowDispatchPolicy(types.FlowKey{Priority: nonExistentPriority})
					return err
				},
				expectErr: contracts.ErrPriorityBandNotFound,
			},
			{
				name: "IntraFlowDispatchPolicy_FlowNotFound",
				action: func(s *registryShard) error {
					_, err := s.IntraFlowDispatchPolicy(types.FlowKey{ID: "missing", Priority: highPriority})
					return err
				},
				expectErr: contracts.ErrFlowInstanceNotFound,
			},
		}
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				h := newShardTestHarness(t)
				err := tc.action(h.shard)
				require.Error(t, err, "The accessor method must return an error for this scenario")
				assert.ErrorIs(t, err, tc.expectErr,
					"The error must wrap the specific sentinel error defined in the contracts package")
			})
		}
	})
}

func TestShard_PriorityBandAccessor(t *testing.T) {
	t.Parallel()

	t.Run("ShouldFail_WhenPriorityDoesNotExist", func(t *testing.T) {
		t.Parallel()
		h := newShardTestHarness(t)
		_, err := h.shard.PriorityBandAccessor(nonExistentPriority)
		assert.ErrorIs(t, err, contracts.ErrPriorityBandNotFound,
			"Requesting an accessor for an unconfigured priority must fail with ErrPriorityBandNotFound")
	})

	t.Run("ShouldSucceed_WhenPriorityExists", func(t *testing.T) {
		t.Parallel()
		h := newShardTestHarness(t)
		accessor, err := h.shard.PriorityBandAccessor(h.highPriorityKey1.Priority)
		require.NoError(t, err, "Requesting an accessor for a configured priority must succeed")
		require.NotNil(t, accessor, "The returned accessor instance must not be nil")

		t.Run("Properties_ShouldReturnCorrectValues", func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, h.highPriorityKey1.Priority, accessor.Priority(),
				"Accessor Priority() must match the configured numerical priority")
			assert.Equal(t, "High", accessor.PriorityName(), "Accessor PriorityName() must match the configured name")
		})

		t.Run("FlowKeys_ShouldReturnAllKeysInBand", func(t *testing.T) {
			t.Parallel()
			keys := accessor.FlowKeys()
			expectedKeys := []types.FlowKey{h.highPriorityKey1, h.highPriorityKey2}
			assert.ElementsMatch(t, expectedKeys, keys,
				"FlowKeys() must return a complete snapshot of all flows registered in this band")
		})

		t.Run("Queue_ShouldReturnCorrectAccessor", func(t *testing.T) {
			t.Parallel()
			q := accessor.Queue(h.highPriorityKey1.ID)
			require.NotNil(t, q, "Queue() must return a non-nil accessor for a registered flow ID")
			assert.Equal(t, h.highPriorityKey1, q.FlowKey(), "The returned queue accessor must have the correct FlowKey")
			assert.Nil(t, accessor.Queue("non-existent"), "Queue() must return nil if the flow ID is not found in this band")
		})

		t.Run("IterateQueues", func(t *testing.T) {
			t.Parallel()

			t.Run("ShouldVisitAllQueuesInBand", func(t *testing.T) {
				t.Parallel()
				var iteratedKeys []types.FlowKey
				accessor.IterateQueues(func(queue framework.FlowQueueAccessor) bool {
					iteratedKeys = append(iteratedKeys, queue.FlowKey())
					return true
				})
				expectedKeys := []types.FlowKey{h.highPriorityKey1, h.highPriorityKey2}
				assert.ElementsMatch(t, expectedKeys, iteratedKeys,
					"IterateQueues must visit every registered flow in the band exactly once")
			})

			t.Run("ShouldExitEarly_WhenCallbackReturnsFalse", func(t *testing.T) {
				t.Parallel()
				var iterationCount int
				accessor.IterateQueues(func(queue framework.FlowQueueAccessor) bool {
					iterationCount++
					return false
				})
				assert.Equal(t, 1, iterationCount, "IterateQueues must terminate immediately when the callback returns false")
			})

			t.Run("ShouldBeSafe_DuringConcurrentMapModification", func(t *testing.T) {
				t.Parallel()
				h := newShardTestHarness(t) // Isolated harness to avoid corrupting the state for other parallel tests
				accessor, err := h.shard.PriorityBandAccessor(highPriority)
				require.NoError(t, err)

				var wg sync.WaitGroup
				wg.Add(2)

				// Goroutine A: The Iterator (constantly reading)
				go func() {
					defer wg.Done()
					for i := 0; i < 100; i++ {
						accessor.IterateQueues(func(queue framework.FlowQueueAccessor) bool {
							// Accessing data should not panic or race.
							_ = queue.FlowKey()
							return true
						})
					}
				}()

				// Goroutine B: The Modifier (constantly writing)
				go func() {
					defer wg.Done()
					for i := 0; i < 100; i++ {
						key := types.FlowKey{ID: fmt.Sprintf("new-flow-%d", i), Priority: highPriority}
						h.synchronizeFlow(key)
						h.shard.deleteFlow(key)
					}
				}()

				// The primary assertion is that this test completes without the race detector firing, which proves the
				// `RLock/WLock` separation is correct.
				wg.Wait()
			})
		})

		t.Run("OnEmptyBand", func(t *testing.T) {
			t.Parallel()
			h := newShardTestHarness(t)
			h.shard.deleteFlow(h.lowPriorityKey)
			accessor, err := h.shard.PriorityBandAccessor(lowPriority)
			require.NoError(t, err, "Setup: getting an accessor for an empty band must succeed")

			keys := accessor.FlowKeys()
			assert.NotNil(t, keys, "FlowKeys() on an empty band must return a non-nil slice")
			assert.Empty(t, keys, "FlowKeys() on an empty band must return an empty slice")

			var callbackExecuted bool
			accessor.IterateQueues(func(queue framework.FlowQueueAccessor) bool {
				callbackExecuted = true
				return true
			})
			assert.False(t, callbackExecuted, "IterateQueues must not execute the callback for an empty band")
		})
	})
}

// --- Lifecycle and State Management Tests ---

func TestShard_SynchronizeFlow(t *testing.T) {
	t.Parallel()
	h := newShardTestHarness(t)
	flowKey := types.FlowKey{ID: "flow1", Priority: highPriority}

	h.synchronizeFlow(flowKey)
	mq1, err := h.shard.ManagedQueue(flowKey)
	require.NoError(t, err, "Flow instance should be accessible after synchronization")

	h.synchronizeFlow(flowKey)
	mq2, err := h.shard.ManagedQueue(flowKey)
	require.NoError(t, err, "Flow instance should remain accessible after idempotent re-synchronization")
	assert.Same(t, mq1, mq2, "Idempotent synchronization must not replace the existing queue instance")
}

func TestShard_DeleteFlow(t *testing.T) {
	t.Parallel()
	h := newShardTestHarness(t)
	_, err := h.shard.ManagedQueue(h.highPriorityKey1)
	require.NoError(t, err, "Test setup: flow instance must exist before deletion")

	h.shard.deleteFlow(h.highPriorityKey1)

	_, err = h.shard.ManagedQueue(h.highPriorityKey1)
	require.Error(t, err, "Flow instance should not be accessible after deletion")
	assert.ErrorIs(t, err, contracts.ErrFlowInstanceNotFound,
		"Accessing a deleted flow must return ErrFlowInstanceNotFound")
}

func TestShard_MarkAsDraining(t *testing.T) {
	t.Parallel()
	h := newShardTestHarness(t)
	assert.True(t, h.shard.IsActive(), "Shard should be active initially")

	h.shard.markAsDraining()
	assert.False(t, h.shard.IsActive(), "Shard must report IsActive as false after being marked for draining")

	h.shard.markAsDraining()
	assert.False(t, h.shard.IsActive(), "Marking as draining should be idempotent")
}

// --- Concurrency Test ---

// TestShard_Concurrency_MixedWorkload is a general stability test that simulates a realistic workload by having
// concurrent readers (e.g., dispatchers) and writers operating on the same shard.
// It provides high confidence that the fine-grained locking strategy is free of deadlocks and data races under
// sustained, mixed contention.
func TestShard_Concurrency_MixedWorkload(t *testing.T) {
	t.Parallel()
	const (
		numReaders   = 5
		numWriters   = 2
		opsPerWriter = 100
	)

	h := newShardTestHarness(t)
	stopCh := make(chan struct{})
	var readersWg, writersWg sync.WaitGroup

	readersWg.Add(numReaders)
	for range numReaders {
		go func() {
			defer readersWg.Done()
			for {
				select {
				case <-stopCh:
					return
				default:
					for _, priority := range h.shard.AllOrderedPriorityLevels() {
						accessor, err := h.shard.PriorityBandAccessor(priority)
						if err == nil {
							accessor.IterateQueues(func(q framework.FlowQueueAccessor) bool { return true })
						}
					}
				}
			}
		}()
	}

	writersWg.Add(numWriters)
	for range numWriters {
		go func() {
			defer writersWg.Done()
			for j := range opsPerWriter {
				// Alternate writing to different flows and priorities to increase contention.
				if j%2 == 0 {
					item := h.addItem(h.highPriorityKey1, 10)
					h.removeItem(h.highPriorityKey1, item)
				} else {
					item := h.addItem(h.lowPriorityKey, 5)
					h.removeItem(h.lowPriorityKey, item)
				}
			}
		}()
	}

	// Wait for all writers to complete first.
	writersWg.Wait()

	// Now stop the readers and wait for them to exit.
	close(stopCh)
	readersWg.Wait()

	// The primary assertion is that this test completes without the race detector firing; however, we can make some final
	// assertions on state consistency.
	finalStats := h.shard.Stats()
	assert.Zero(t, finalStats.TotalLen, "After all paired add/remove operations, the total length should be zero")
	assert.Zero(t, finalStats.TotalByteSize, "After all paired add/remove operations, the total byte size should be zero")
}

// --- Invariant Tests ---

func TestShard_InvariantPanics(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name   string
		action func(h *shardTestHarness)
	}{
		{
			name:   "OnStatsPropagatedForUnknownPriority",
			action: func(h *shardTestHarness) { h.shard.propagateStatsDelta(nonExistentPriority, 1, 1) },
		},
		{
			name:   "OnSynchronizingFlowWithMissingPriority",
			action: func(h *shardTestHarness) { h.synchronizeFlow(types.FlowKey{ID: "flow", Priority: nonExistentPriority}) },
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			h := newShardTestHarness(t)
			assert.Panics(t, func() { tc.action(h) },
				"The action must trigger a panic when a critical system invariant is violated")
		})
	}
}
