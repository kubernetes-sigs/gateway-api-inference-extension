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
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	testclock "k8s.io/utils/clock/testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/interflow"
	intra "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

// --- Test Harness ---

// registryTestHarness provides a fully initialized test harness for the `FlowRegistry`.
type registryTestHarness struct {
	t         *testing.T
	fr        *FlowRegistry
	config    Config
	fakeClock *testclock.FakeClock
}

// harnessOptions configures the test harness.
type harnessOptions struct {
	config            *Config
	initialShardCount int
}

// newRegistryTestHarness creates and starts a new `FlowRegistry` for testing.
func newRegistryTestHarness(t *testing.T, opts harnessOptions) *registryTestHarness {
	t.Helper()

	var config Config
	if opts.config != nil {
		config = *opts.config
	} else {
		config = Config{
			FlowGCTimeout: 5 * time.Minute, // Use a realistic but controllable GC time.
			PriorityBands: []PriorityBandConfig{
				{Priority: highPriority, PriorityName: "High"},
				{Priority: lowPriority, PriorityName: "Low"},
			},
		}
	}
	if opts.initialShardCount > 0 {
		config.InitialShardCount = opts.initialShardCount
	}

	validatedCfg, err := config.ValidateAndApplyDefaults()
	require.NoError(t, err, "Test setup: validating config should not fail")

	fakeClock := testclock.NewFakeClock(time.Now())
	registryOpts := []RegistryOption{withClock(fakeClock)}
	fr, err := NewFlowRegistry(*validatedCfg, logr.Discard(), registryOpts...)
	require.NoError(t, err, "Test setup: NewFlowRegistry should not fail")

	// Start the GC loop in the background.
	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		fr.Run(ctx)
	}()
	t.Cleanup(func() {
		cancel()
		wg.Wait()
	})

	return &registryTestHarness{
		t:         t,
		fr:        fr,
		config:    *fr.config,
		fakeClock: fakeClock,
	}
}

// assertFlowExists synchronously checks if a flow's queue exists on the first shard.
func (h *registryTestHarness) assertFlowExists(key types.FlowKey, msgAndArgs ...any) {
	h.t.Helper()
	require.NotEmpty(h.t, h.fr.allShards, "Cannot check for flow existence when no shards are present")
	_, err := h.fr.allShards[0].ManagedQueue(key)
	assert.NoError(h.t, err, msgAndArgs...)
}

// assertFlowDoesNotExist synchronously checks if a flow's queue does not exist.
func (h *registryTestHarness) assertFlowDoesNotExist(key types.FlowKey, msgAndArgs ...any) {
	h.t.Helper()
	if len(h.fr.allShards) == 0 {
		assert.True(h.t, true, "Flow correctly does not exist because no shards exist")
		return
	}
	_, err := h.fr.allShards[0].ManagedQueue(key)
	require.Error(h.t, err, "Expected an error when getting a non-existent flow, but got none")
	assert.ErrorIs(h.t, err, contracts.ErrFlowInstanceNotFound, msgAndArgs...)
}

// openConnectionOnFlow ensures a flow is registered for the provided `key`.
func (h *registryTestHarness) openConnectionOnFlow(key types.FlowKey) {
	h.t.Helper()
	err := h.fr.WithConnection(key, func(conn contracts.ActiveFlowConnection) error { return nil })
	require.NoError(h.t, err, "Registering flow %s should not fail", key)
	h.assertFlowExists(key, "Flow %s should exist after registration", key)
}

// --- Constructor and Lifecycle Tests ---

func TestFlowRegistry_New(t *testing.T) {
	t.Parallel()

	t.Run("ShouldFail_WhenInitialShardCreationFails", func(t *testing.T) {
		t.Parallel()
		config, err := newConfig(
			Config{PriorityBands: []PriorityBandConfig{{Priority: highPriority, PriorityName: "A"}}},
			withInterFlowDispatchPolicyFactory(func(interflow.RegisteredPolicyName) (framework.InterFlowDispatchPolicy, error) {
				return nil, errors.New("injected factory failure")
			}),
		)
		require.NoError(t, err, "Test setup: creating the config object itself should not fail")
		_, err = NewFlowRegistry(*config, logr.Discard())
		require.Error(t, err, "NewFlowRegistry should fail when initial shard setup fails")
		assert.Contains(t, err.Error(), "injected factory failure",
			"Error message should reflect the root cause from the failing plugin factory")
	})
}

// --- `FlowRegistryClient` API Tests ---

func TestFlowRegistry_WithConnection_AndHandle(t *testing.T) {
	t.Parallel()

	t.Run("ShouldJITRegisterFlow_OnFirstConnection", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{})
		key := types.FlowKey{ID: "jit-flow", Priority: highPriority}

		h.assertFlowDoesNotExist(key, "Flow should not exist before the first connection")

		err := h.fr.WithConnection(key, func(conn contracts.ActiveFlowConnection) error {
			h.assertFlowExists(key, "Flow should exist immediately after JIT registration within the connection")
			require.NotNil(t, conn, "Connection handle provided to callback must not be nil")
			return nil
		})

		require.NoError(t, err, "WithConnection should succeed for a new flow")
		h.assertFlowExists(key, "Flow should remain in the registry after the connection is closed")
	})

	t.Run("ShouldFail_WhenFlowIDIsEmpty", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{})
		key := types.FlowKey{ID: "", Priority: highPriority} // Invalid key

		err := h.fr.WithConnection(key, func(conn contracts.ActiveFlowConnection) error {
			t.Fatal("Callback must not be executed when the provided flow key is invalid")
			return nil
		})

		require.Error(t, err, "WithConnection must return an error for an empty flow ID")
		assert.ErrorIs(t, err, contracts.ErrFlowIDEmpty, "The returned error must be of the correct type")
	})

	t.Run("ShouldFail_WhenJITFails", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{})
		h.fr.config.intraFlowDispatchPolicyFactory = func(intra.RegisteredPolicyName) (framework.IntraFlowDispatchPolicy, error) {
			return nil, errors.New("injected factory failure")
		}
		key := types.FlowKey{ID: "test-flow", Priority: highPriority}

		err := h.fr.WithConnection(key, func(conn contracts.ActiveFlowConnection) error {
			t.Fatal("Callback must not be executed when the flow failes to register JIT")
			return nil
		})

		require.Error(t, err, "WithConnection must return an error for a failed flow JIT registration")
		assert.ErrorContains(t, err, "injected factory failure", "The returned error must propagate the reason")
	})

	t.Run("Handle_Shards_ShouldReturnAllActiveShardsAndBeACopy", func(t *testing.T) {
		t.Parallel()
		// Create a registry with a known mixed topology of Active and Draining shards.
		h := newRegistryTestHarness(t, harnessOptions{initialShardCount: 3})
		err := h.fr.updateShardCount(2) // This leaves one shard in the Draining state.
		require.NoError(t, err, "Test setup: scaling down to create a draining shard should not fail")
		require.Len(t, h.fr.allShards, 3, "Test setup: should have 2 active and 1 draining shard")

		key := types.FlowKey{ID: "test-flow", Priority: highPriority}

		err = h.fr.WithConnection(key, func(conn contracts.ActiveFlowConnection) error {
			shards := conn.ActiveShards()

			assert.Len(t, shards, 2, "ActiveShards() must only return the Active shards")

			// Assert it's a copy by maliciously modifying it.
			require.NotEmpty(t, shards, "Test setup assumes shards are present")
			shards[0] = nil // Modify the local copy.

			return nil
		})
		require.NoError(t, err)

		// Prove the registry's internal state was not mutated by the modification.
		assert.NotNil(t, h.fr.activeShards[0],
			"Modifying the slice returned by ActiveShards() must not affect the registry's internal state")
	})
}

// --- `FlowRegistryAdmin` API Tests ---

func TestFlowRegistry_Stats(t *testing.T) {
	t.Parallel()

	h := newRegistryTestHarness(t, harnessOptions{initialShardCount: 2})
	keyHigh := types.FlowKey{ID: "high-pri-flow", Priority: highPriority}
	keyLow := types.FlowKey{ID: "low-pri-flow", Priority: lowPriority}
	h.openConnectionOnFlow(keyHigh)
	h.openConnectionOnFlow(keyLow)

	shards := h.fr.allShards
	require.Len(t, shards, 2, "Test setup assumes 2 shards")
	mqHigh0, _ := shards[0].ManagedQueue(keyHigh)
	mqHigh1, _ := shards[1].ManagedQueue(keyHigh)
	mqLow1, _ := shards[1].ManagedQueue(keyLow)
	require.NoError(t, mqHigh0.Add(mocks.NewMockQueueItemAccessor(10, "req1", keyHigh)),
		"Adding item to queue should not fail")
	require.NoError(t, mqHigh1.Add(mocks.NewMockQueueItemAccessor(20, "req2", keyHigh)),
		"Adding item to queue should not fail")
	require.NoError(t, mqLow1.Add(mocks.NewMockQueueItemAccessor(30, "req3", keyLow)),
		"Adding item to queue should not fail")

	// Although the production `Stats()` method provides a 'fuzzy snapshot' under high contention, our test validates it
	// in a quiescent state, so these assertions can and must be exact.
	globalStats := h.fr.Stats()
	assert.Equal(t, uint64(3), globalStats.TotalLen, "Global TotalLen should be the sum of all items")
	assert.Equal(t, uint64(60), globalStats.TotalByteSize, "Global TotalByteSize should be the sum of all item sizes")

	shardStats := h.fr.ShardStats()
	require.Len(t, shardStats, 2, "Should return stats for 2 shards")
	var totalShardLen, totalShardBytes uint64
	for _, ss := range shardStats {
		assert.True(t, ss.IsActive, "All shards should be active in this test")
		assert.NotEmpty(t, ss.PerPriorityBandStats, "Each shard should have stats for its priority bands")
		assert.NotEmpty(t, ss.ID, "Each shard should have a non-empty ID")
		totalShardLen += ss.TotalLen
		totalShardBytes += ss.TotalByteSize
	}
	assert.Equal(t, globalStats.TotalLen, totalShardLen, "Sum of shard lengths must equal global length")
	assert.Equal(t, globalStats.TotalByteSize, totalShardBytes, "Sum of shard byte sizes must equal global byte size")
}

// --- Garbage Collection Tests ---

func TestFlowRegistry_GarbageCollection(t *testing.T) {
	t.Run("ShouldCollectIdleFlow", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{})
		key := types.FlowKey{ID: "idle-flow", Priority: highPriority}

		h.openConnectionOnFlow(key)                            // Create a flow, which is born Idle.
		h.fakeClock.Step(h.config.FlowGCTimeout + time.Second) // Advance the clock just past the GC timeout.
		h.fr.executeGCCycle()                                  // Manually and deterministically trigger a GC cycle.

		h.assertFlowDoesNotExist(key, "Idle flow should be collected by the GC")
	})

	t.Run("ShouldNotCollectActiveFlow", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{})
		key := types.FlowKey{ID: "active-flow", Priority: highPriority}

		var wg sync.WaitGroup
		leaseAcquired := make(chan struct{})
		releaseLease := make(chan struct{})
		wg.Add(1)

		go func() {
			// This goroutine holds the lease. It will not exit until the main test goroutine calls `wg.Done()`.
			defer wg.Done()
			err := h.fr.WithConnection(key, func(contracts.ActiveFlowConnection) error {
				close(leaseAcquired) // Signal to the main test that the lease is now active.
				<-releaseLease       // Block here, holding the lease, until signaled.

				return nil
			})
			require.NoError(t, err, "WithConnection in the background goroutine should not fail")
		}()
		t.Cleanup(func() {
			close(releaseLease) // Unblock the goroutine.
			wg.Wait()           // Wait for the goroutine to fully exit.
		})

		<-leaseAcquired                              // Wait until the goroutine confirms that it has acquired the lease.
		h.fakeClock.Step(h.config.FlowGCTimeout * 2) // Advance the clock well past the GC timeout.
		h.fr.executeGCCycle()                        // Manually and deterministically trigger a GC cycle.

		h.assertFlowExists(key, "An active flow must not be garbage collected, even after a forced GC cycle")
	})

	t.Run("ShouldResetGCTimer_WhenFlowBecomesActive", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{})
		key := types.FlowKey{ID: "reactivated-flow", Priority: highPriority}
		h.openConnectionOnFlow(key)                            // Create an flow with a new idleness timer.
		h.fakeClock.Step(h.config.FlowGCTimeout - time.Second) // Advance the clock to just before the GC timeout.
		h.openConnectionOnFlow(key)                            // Open a new connection, resetting its idleness timer.
		h.fakeClock.Step(2 * time.Second)                      // Advance the clock again.
		h.fr.executeGCCycle()                                  // Manually and deterministically trigger a GC cycle.

		h.assertFlowExists(key, "Flow should survive GC because its idleness timer was reset")
	})

	t.Run("ShouldAbortSweep_WhenFlowBecomesActiveAfterScan", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{})
		key := types.FlowKey{ID: "re-activated-flow", Priority: highPriority}
		h.openConnectionOnFlow(key)

		// Make the flow a candidate for GC.
		h.fakeClock.Step(h.config.FlowGCTimeout + time.Second)
		candidates := []types.FlowKey{key}

		// Get the flow's state so we can manipulate its lock.
		val, ok := h.fr.flowStates.Load(key)
		require.True(t, ok, "Test setup: flow state must exist")
		state := val.(*flowState)

		// Lock the flow's `gcLock` to pause the GC.
		state.gcLock.Lock()
		defer state.gcLock.Unlock()

		// Start the sweep in the background; it will block.
		sweepDone := make(chan struct{})
		go func() {
			defer close(sweepDone)
			h.fr.verifyAndSweepFlows(candidates)
		}()

		// While the sweep is blocked, make the flow Active again.
		// This simulates a new connection arriving just in time.
		state.leaseCount.Add(1)

		// Unblock the sweep.
		state.gcLock.Unlock()

		// Wait for the sweep to complete.
		select {
		case <-sweepDone:
			// Continue to assertion.
		case <-time.After(time.Second):
			t.Fatal("verifyAndSweepFlows deadlocked or timed out")
		}

		h.assertFlowExists(key, "Flow should not be collected because it became active before the sweep")
		state.gcLock.Lock() // Re-lock for the deferred unlock.
	})

	t.Run("ShouldCollectDrainingShard_OnlyWhenEmpty", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{initialShardCount: 2})
		key := types.FlowKey{ID: "flow-on-draining-shard", Priority: highPriority}
		h.openConnectionOnFlow(key)

		// Add an item to a queue on the soon-to-be Draining shard to keep it busy.
		drainingShard := h.fr.activeShards[1] // The shard that will become draining.
		mq, err := drainingShard.ManagedQueue(key)
		require.NoError(t, err, "Test setup: getting queue on draining shard failed")
		item := mocks.NewMockQueueItemAccessor(100, "req1", key)
		require.NoError(t, mq.Add(item), "Adding item to non-active shard should be allowed for in-flight requests")

		// Scale down to mark one shard as Draining.
		require.NoError(t, h.fr.updateShardCount(1), "Test setup: scale down should succeed")
		require.Len(t, h.fr.drainingShards, 1, "Test setup: one shard should be draining")

		// Trigger a GC cycle while the shard is not empty.
		h.fr.sweepDrainingShards()
		require.Len(t, h.fr.drainingShards, 1, "Draining shard should not be collected while it is not empty")

		// Empty the shard and trigger GC again.
		_, err = mq.Remove(item.Handle())
		require.NoError(t, err, "Test setup: removing item from draining shard failed")
		h.fr.sweepDrainingShards()
		assert.Empty(t, h.fr.drainingShards, "Draining shard should be collected after it becomes empty")
	})

	t.Run("ShouldHandleBenignRace_WhenSweepingAlreadyDeletedFlow", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{})
		key := types.FlowKey{ID: "benign-race-flow", Priority: highPriority}
		h.openConnectionOnFlow(key)

		// Get the flow state so we can lock it.
		val, ok := h.fr.flowStates.Load(key)
		require.True(t, ok, "Test setup: flow state must exist")
		state := val.(*flowState)

		// Make the flow a candidate for GC.
		h.fakeClock.Step(h.config.FlowGCTimeout + time.Second)
		candidates := []types.FlowKey{key}

		// Manually lock the flow's `gcLock`. This simulates the GC being stuck just before its "Verify" phase.
		state.gcLock.Lock()
		defer state.gcLock.Unlock()

		// In a background goroutine, run the sweep. It will block on the lock.
		sweepDone := make(chan struct{})
		go func() {
			defer close(sweepDone)
			h.fr.verifyAndSweepFlows(candidates)
		}()

		// While the sweep is blocked, delete the flow from underneath it.
		// This creates the benign race condition.
		h.fr.flowStates.Delete(key)

		// Unblock the sweep logic.
		state.gcLock.Unlock() // Temporarily unlock to let the sweep proceed.

		// The sweep must complete without panicking.
		select {
		case <-sweepDone:
			// Success! The test completed gracefully.
		case <-time.After(time.Second):
			t.Fatal("verifyAndSweepFlows deadlocked or timed out")
		}
		state.gcLock.Lock() // Re-lock for the deferred unlock.
	})
}

// --- Shard Management Tests ---

func TestFlowRegistry_UpdateShardCount(t *testing.T) {
	t.Parallel()
	const (
		globalCapacity = 100
		bandCapacity   = 50
	)
	testCases := []struct {
		name                                string
		initialShardCount                   int
		targetShardCount                    int
		expectedActiveCount                 int
		expectedPartitionedGlobalCapacities map[uint64]int
		expectedPartitionedBandCapacities   map[uint64]int
		expectErrIs                         error // Optional
	}{
		{
			name:                                "NoOp_ScaleToSameCount",
			initialShardCount:                   2,
			targetShardCount:                    2,
			expectedActiveCount:                 2,
			expectedPartitionedGlobalCapacities: map[uint64]int{50: 2},
			expectedPartitionedBandCapacities:   map[uint64]int{25: 2},
		},
		{
			name:                                "Succeeds_ScaleUp_FromOne",
			initialShardCount:                   1,
			targetShardCount:                    4,
			expectedActiveCount:                 4,
			expectedPartitionedGlobalCapacities: map[uint64]int{25: 4},
			expectedPartitionedBandCapacities:   map[uint64]int{12: 2, 13: 2},
		},
		{
			name:                                "Succeeds_ScaleDown_ToOne",
			initialShardCount:                   3,
			targetShardCount:                    1,
			expectedActiveCount:                 1,
			expectedPartitionedGlobalCapacities: map[uint64]int{100: 1},
			expectedPartitionedBandCapacities:   map[uint64]int{50: 1},
		},
		{
			name:                                "Error_ScaleDown_ToZero",
			initialShardCount:                   2,
			targetShardCount:                    0,
			expectedActiveCount:                 2,
			expectErrIs:                         contracts.ErrInvalidShardCount,
			expectedPartitionedGlobalCapacities: map[uint64]int{50: 2},
			expectedPartitionedBandCapacities:   map[uint64]int{25: 2},
		},
		{
			name:                                "Error_ScaleDown_ToNegative",
			initialShardCount:                   1,
			targetShardCount:                    -1,
			expectedActiveCount:                 1,
			expectErrIs:                         contracts.ErrInvalidShardCount,
			expectedPartitionedGlobalCapacities: map[uint64]int{100: 1},
			expectedPartitionedBandCapacities:   map[uint64]int{50: 1},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			config := Config{
				MaxBytes: globalCapacity,
				PriorityBands: []PriorityBandConfig{
					{Priority: highPriority, PriorityName: "A", MaxBytes: bandCapacity},
				},
				InitialShardCount: tc.initialShardCount,
			}

			h := newRegistryTestHarness(t, harnessOptions{config: &config})
			key := types.FlowKey{ID: "flow", Priority: highPriority}
			h.openConnectionOnFlow(key)

			err := h.fr.updateShardCount(tc.targetShardCount)
			if tc.expectErrIs != nil {
				require.Error(t, err, "UpdateShardCount should have returned an error")
				assert.ErrorIs(t, err, tc.expectErrIs, "Error should be the expected type")
			} else {
				require.NoError(t, err, "UpdateShardCount should not have returned an error")
			}

			globalCapacities := make(map[uint64]int)
			bandCapacities := make(map[uint64]int)

			h.fr.mu.RLock()
			finalActiveCount := len(h.fr.activeShards)
			finalDrainingCount := len(h.fr.drainingShards)
			for _, shard := range h.fr.activeShards {
				stats := shard.Stats()
				globalCapacities[stats.TotalCapacityBytes]++
				bandCapacities[stats.PerPriorityBandStats[highPriority].CapacityBytes]++
				h.assertFlowExists(key, "Shard %s should contain the existing flow", shard.ID())
			}
			h.fr.mu.RUnlock()

			expectedDrainingCount := 0
			if tc.initialShardCount > tc.expectedActiveCount {
				expectedDrainingCount = tc.initialShardCount - tc.expectedActiveCount
			}
			assert.Equal(t, tc.expectedActiveCount, finalActiveCount, "Final active shard count is incorrect")
			assert.Equal(t, expectedDrainingCount, finalDrainingCount, "Final draining shard count in registry is incorrect")
			assert.Equal(t, tc.expectedPartitionedGlobalCapacities, globalCapacities,
				"Global capacity re-partitioning incorrect")
			assert.Equal(t, tc.expectedPartitionedBandCapacities, bandCapacities, "Band capacity re-partitioning incorrect")
		})
	}
}

// --- Concurrency Tests ---

func TestFlowRegistry_Concurrency(t *testing.T) {
	t.Parallel()

	t.Run("ConcurrentJITRegistrations_ShouldBeSafe", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{})
		key := types.FlowKey{ID: "concurrent-flow", Priority: highPriority}
		numGoroutines := 50
		var wg sync.WaitGroup
		wg.Add(numGoroutines)

		// Hammer the `WithConnection` method for the same key from many goroutines.
		for range numGoroutines {
			go func() {
				defer wg.Done()
				err := h.fr.WithConnection(key, func(contracts.ActiveFlowConnection) error {
					// Do a small amount of work inside the connection.
					time.Sleep(1 * time.Millisecond)
					return nil
				})
				require.NoError(t, err, "Concurrent WithConnection calls must not fail")
			}()
		}
		wg.Wait()

		// The primary assertion is that this completes without the race detector firing.
		// We can also check that the flow state is consistent.
		h.assertFlowExists(key, "Flow must exist after concurrent JIT registration")
	})

	t.Run("MixedAdminAndDataPlaneWorkload", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{initialShardCount: 1})
		const (
			numWorkers    = 10
			opsPerWorker  = 50
			maxShardCount = 4
		)

		var wg sync.WaitGroup
		wg.Add(numWorkers + 1) // +1 for the scaling goroutine

		// Data Plane Workers: Constantly creating new flows.
		for i := range numWorkers {
			workerID := i
			go func() {
				defer wg.Done()
				for j := range opsPerWorker {
					key := types.FlowKey{ID: fmt.Sprintf("flow-%d-%d", workerID, j), Priority: highPriority}
					_ = h.fr.WithConnection(key, func(contracts.ActiveFlowConnection) error { return nil })
				}
			}()
		}

		// Admin Worker: Constantly scaling the number of shards up and down.
		go func() {
			defer wg.Done()
			for i := 1; i < maxShardCount; i++ {
				_ = h.fr.updateShardCount(i + 1)
				_ = h.fr.updateShardCount(i)
			}
		}()

		wg.Wait()

		// The test completing without a race condition is the primary assertion.
		// We can also assert a consistent final state.
		assert.Len(t, h.fr.activeShards, maxShardCount-1, "Final active shard count should be consistent")
		flowCount := 0
		h.fr.flowStates.Range(func(_, _ any) bool {
			flowCount++
			return true
		})
		assert.Equal(t, numWorkers*opsPerWorker, flowCount, "All concurrently registered flows must be present")
	})
}
