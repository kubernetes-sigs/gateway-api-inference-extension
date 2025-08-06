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
	"k8s.io/utils/clock"
	testclock "k8s.io/utils/clock/testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	inter "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/interflow/dispatch"
	intra "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

const syncTimeout = 2 * time.Second

// --- Test Harness and Mocks ---

// registryTestHarness provides a fully initialized test harness for the `FlowRegistry`.
type registryTestHarness struct {
	t           *testing.T
	ctx         context.Context
	cancel      context.CancelFunc
	fr          *FlowRegistry
	config      Config
	fakeClock   *testclock.FakeClock
	activeItems map[types.FlowKey]types.QueueItemAccessor
}

type harnessOptions struct {
	config            *Config
	useFakeClock      bool
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
			FlowGCTimeout: 1 * time.Minute,
			PriorityBands: []PriorityBandConfig{
				{Priority: 10, PriorityName: "High"},
				{Priority: 20, PriorityName: "Low"},
			},
		}
	}
	if opts.initialShardCount > 0 {
		config.InitialShardCount = opts.initialShardCount
	}

	var clk clock.WithTickerAndDelayedExecution
	var fakeClock *testclock.FakeClock
	if opts.useFakeClock {
		fakeClock = testclock.NewFakeClock(time.Now())
		clk = fakeClock
	}

	fr, err := NewFlowRegistry(config, logr.Discard(), WithClock(clk))
	require.NoError(t, err, "Test setup: NewFlowRegistry should not fail")

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
		t:           t,
		ctx:         ctx,
		cancel:      cancel,
		fr:          fr,
		config:      config,
		fakeClock:   fakeClock,
		activeItems: make(map[types.FlowKey]types.QueueItemAccessor),
	}
}

// synchronize blocks until the `FlowRegistry`'s event loop has processed all preceding events.
func (h *registryTestHarness) synchronize() {
	h.t.Helper()
	doneCh := make(chan struct{})
	select {
	case h.fr.events <- &syncEvent{doneCh: doneCh}:
		select {
		case <-doneCh:
		case <-time.After(syncTimeout):
			h.t.Fatalf("Timeout waiting for FlowRegistry synchronization ack")
		}
	case <-time.After(syncTimeout):
		h.t.Fatalf("Timeout sending sync event to FlowRegistry (channel may be full)")
	}
}

// waitForGCTick advances the fake clock to trigger a GC cycle and then blocks until that cycle is complete.
// This works by sandwiching the clock step between two synchronization barriers.
func (h *registryTestHarness) waitForGCTick() {
	h.t.Helper()
	require.NotNil(h.t, h.fakeClock, "waitForGCTick requires the fake clock to be enabled")
	h.synchronize()
	h.fakeClock.Step(h.config.FlowGCTimeout + time.Millisecond)
	h.synchronize()
}

// setFlowActive makes a flow Active (by adding an item) or Idle (by removing it) and synchronizes the state change.
func (h *registryTestHarness) setFlowActive(key types.FlowKey, active bool) {
	h.t.Helper()
	shard := h.getFirstActiveShard()
	mq, err := shard.ManagedQueue(key)
	require.NoError(h.t, err, "Failed to get managed queue to set flow activity for flow %s", key)

	if active {
		require.NotContains(h.t, h.activeItems, key, "Flow %s is already marked as Active in the test harness", key)
		item := mocks.NewMockQueueItemAccessor(100, "req", key)
		require.NoError(h.t, mq.Add(item), "Failed to add item to make flow %s Active", key)
		h.activeItems[key] = item
	} else {
		item, ok := h.activeItems[key]
		require.True(h.t, ok, "Flow %s was not Active in the test harness, cannot make it Idle", key)
		_, err := mq.Remove(item.Handle())
		require.NoError(h.t, err, "Failed to remove item to make flow %s Idle", key)
		delete(h.activeItems, key)
	}
	h.synchronize()
}

func (h *registryTestHarness) getFirstActiveShard() contracts.RegistryShard {
	h.t.Helper()
	allShards := h.fr.Shards()
	for _, s := range allShards {
		if s.IsActive() {
			return s
		}
	}
	h.t.Fatalf("Failed to find any active shard in list of %d shards", len(allShards))
	return nil
}

func (h *registryTestHarness) assertFlowExists(key types.FlowKey, msgAndArgs ...any) {
	h.t.Helper()
	allShards := h.fr.Shards()
	require.NotEmpty(h.t, allShards, "Cannot assert flow existence without shards")

	// It's sufficient to check one shard, as the registry guarantees consistency.
	// A more robust check could iterate all, but this is a good balance.
	_, err := allShards[0].ManagedQueue(key)
	assert.NoError(h.t, err, msgAndArgs...)
}

func (h *registryTestHarness) assertFlowDoesNotExist(key types.FlowKey, msgAndArgs ...any) {
	h.t.Helper()
	allShards := h.fr.Shards()
	// If there are no shards, the flow cannot exist.
	if len(allShards) == 0 {
		return
	}
	_, err := allShards[0].ManagedQueue(key)
	assert.ErrorIs(h.t, err, contracts.ErrFlowInstanceNotFound, msgAndArgs...)
}

// --- Basic Tests ---

func TestFlowRegistry_New_ErrorPaths(t *testing.T) {
	t.Parallel()

	t.Run("ShouldFail_WhenConfigIsInvalid", func(t *testing.T) {
		t.Parallel()
		_, err := NewFlowRegistry(Config{}, logr.Discard()) // No priority bands is invalid.
		require.Error(t, err, "NewFlowRegistry should fail with an invalid config")
	})

	t.Run("ShouldFail_WhenInitialShardCreationFails", func(t *testing.T) {
		t.Parallel()
		failingInterFlowFactory := func(_ inter.RegisteredPolicyName) (framework.InterFlowDispatchPolicy, error) {
			return nil, errors.New("injected shard creation failure")
		}
		configWithFailingFactory := Config{
			PriorityBands: []PriorityBandConfig{{Priority: 10, PriorityName: "High"}},
		}
		cfg, err := NewConfig(configWithFailingFactory, withInterFlowDispatchPolicyFactory(failingInterFlowFactory))
		require.NoError(t, err, "Test setup: creating the config object itself should not fail")

		_, err = NewFlowRegistry(*cfg, logr.Discard())

		require.Error(t, err, "NewFlowRegistry should fail when initial shard setup fails")
		assert.Contains(t, err.Error(), "injected shard creation failure", "Error message should reflect the root cause")
	})
}

// --- Administrative Method Tests ---

func TestFlowRegistry_RegisterOrUpdateFlow(t *testing.T) {
	t.Parallel()
	h := newRegistryTestHarness(t, harnessOptions{})
	key := types.FlowKey{ID: "test-flow", Priority: 10}
	spec := types.FlowSpecification{Key: key}

	err := h.fr.RegisterOrUpdateFlow(spec)
	require.NoError(t, err, "Registering a new flow should succeed")
	h.assertFlowExists(key, "Flow should exist in registry after initial registration")

	err = h.fr.RegisterOrUpdateFlow(spec)
	require.NoError(t, err, "Re-registering the same flow should be idempotent")
	h.assertFlowExists(key, "Flow should still exist after idempotent re-registration")
}

func TestFlowRegistry_RegisterOrUpdateFlow_ErrorPaths(t *testing.T) {
	t.Parallel()

	// Define mock factories that can be injected to force specific failures.
	failingPolicyFactory := func(name intra.RegisteredPolicyName) (framework.IntraFlowDispatchPolicy, error) {
		return nil, errors.New("injected policy factory failure")
	}
	failingQueueFactory := func(name queue.RegisteredQueueName, c framework.ItemComparator) (framework.SafeQueue, error) {
		return nil, errors.New("injected queue factory failure")
	}

	baseConfig := Config{
		PriorityBands: []PriorityBandConfig{
			{Priority: 10, PriorityName: "High"},
		},
	}

	testCases := []struct {
		name   string
		spec   types.FlowSpecification
		setup  func(h *registryTestHarness) // Surgical setup to induce failure.
		errStr string                       // Expected substring in the error message.
		errIs  error                        // Optional: Expected wrapped error.
	}{
		{
			name:   "ShouldFail_WhenSpecIDIsEmpty",
			spec:   types.FlowSpecification{Key: types.FlowKey{ID: "", Priority: 10}},
			errStr: "flow ID cannot be empty",
			errIs:  contracts.ErrFlowIDEmpty,
		},
		{
			name:   "ShouldFail_WhenPriorityBandIsNotFound",
			spec:   types.FlowSpecification{Key: types.FlowKey{ID: "flow", Priority: 99}},
			errStr: "failed to get configuration for priority 99",
			errIs:  contracts.ErrPriorityBandNotFound,
		},
		{
			name: "ShouldFail_WhenPolicyFactoryFails",
			spec: types.FlowSpecification{Key: types.FlowKey{ID: "flow", Priority: 10}},
			setup: func(h *registryTestHarness) {
				// After the registry is created with a valid config, we surgically replace the policy factory with one that is
				// guaranteed to fail.
				h.fr.config.intraFlowDispatchPolicyFactory = failingPolicyFactory
			},
			errStr: "failed to instantiate intra-flow policy",
		},
		{
			name: "ShouldFail_WhenQueueFactoryFails",
			spec: types.FlowSpecification{Key: types.FlowKey{ID: "flow", Priority: 10}},
			setup: func(h *registryTestHarness) {
				// Similarly, we inject a failing queue factory.
				h.fr.config.queueFactory = failingQueueFactory
			},
			errStr: "failed to instantiate queue",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			h := newRegistryTestHarness(t, harnessOptions{config: &baseConfig})
			if tc.setup != nil {
				tc.setup(h)
			}

			err := h.fr.RegisterOrUpdateFlow(tc.spec)

			require.Error(t, err, "RegisterOrUpdateFlow should have returned an error")
			assert.Contains(t, err.Error(), tc.errStr, "Error message should contain expected text")
			if tc.errIs != nil {
				assert.ErrorIs(t, err, tc.errIs, "Error should match expected wrapped error")
			}
		})
	}
}

func TestFlowRegistry_UpdateShardCount(t *testing.T) {
	t.Parallel()

	// stateForDrainingTest holds the specific item that will be placed on a shard that is about to be Drained.
	// We need to pass it from the setup phase to the assertion phase.
	type stateForDrainingTest struct {
		drainingShardID string
		item            types.QueueItemAccessor
		flowKey         types.FlowKey
	}

	testCases := []struct {
		name              string
		initialShardCount int
		targetShardCount  int
		setup             func(h *registryTestHarness) any // Returns optional state for assertions
		expectErrIs       error
		expectErrStr      string
		assertions        func(t *testing.T, h *registryTestHarness, state any)
	}{
		{
			name:              "ShouldSucceed_OnScaleUp",
			initialShardCount: 2,
			targetShardCount:  4,
			assertions: func(t *testing.T, h *registryTestHarness, _ any) {
				assert.Len(t, h.fr.Shards(), 4, "Registry should have 4 shards after scale-up")
				assert.Len(t, h.fr.activeShards, 4, "Registry should have 4 active shards")
				assert.Empty(t, h.fr.drainingShards, "Registry should have 0 draining shards")
			},
		},
		{
			name:              "ShouldBeNoOp_WhenCountIsUnchanged",
			initialShardCount: 2,
			targetShardCount:  2,
			assertions: func(t *testing.T, h *registryTestHarness, _ any) {
				assert.Len(t, h.fr.Shards(), 2, "Shard count should remain unchanged")
			},
		},
		{
			name:              "ShouldSucceed_OnScaleUp_WithExistingFlows",
			initialShardCount: 1,
			targetShardCount:  2,
			setup: func(h *registryTestHarness) any {
				key := types.FlowKey{ID: "flow", Priority: 10}
				require.NoError(t, h.fr.RegisterOrUpdateFlow(types.FlowSpecification{Key: key}),
					"Test setup: failed to register flow")
				return key // Pass the key to assertions
			},
			assertions: func(t *testing.T, h *registryTestHarness, state any) {
				key := state.(types.FlowKey)
				assert.Len(t, h.fr.Shards(), 2, "Registry should now have 2 shards")
				h.assertFlowExists(key, "Flow must exist on all shards after scaling up")
			},
		},
		{
			name:              "ShouldFail_WhenShardCountIsInvalid_Zero",
			initialShardCount: 1,
			targetShardCount:  0,
			expectErrIs:       contracts.ErrInvalidShardCount,
		},
		{
			name:              "ShouldFail_WhenShardCountIsInvalid_Negative",
			initialShardCount: 1,
			targetShardCount:  -1,
			expectErrIs:       contracts.ErrInvalidShardCount,
		},
		{
			name:              "ShouldFailAndRollback_WhenFlowSyncFailsDuringScaleUp",
			initialShardCount: 1,
			targetShardCount:  2,
			setup: func(h *registryTestHarness) any {
				// Create a valid, existing flow in the registry.
				key := types.FlowKey{ID: "flow", Priority: 10}
				require.NoError(t, h.fr.RegisterOrUpdateFlow(types.FlowSpecification{Key: key}),
					"Test setup: failed to register existing flow")

				// Sabotage the system: Inject a policy factory that is guaranteed to fail.
				// This will be called when the registry tries to create components for the existing flow on the *new* shard.
				failingPolicyFactory := func(_ intra.RegisteredPolicyName) (framework.IntraFlowDispatchPolicy, error) {
					return nil, errors.New("injected flow sync failure")
				}
				h.fr.mu.Lock()
				h.fr.config.intraFlowDispatchPolicyFactory = failingPolicyFactory
				h.fr.mu.Unlock()

				return nil
			},
			expectErrStr: "injected flow sync failure",
			assertions: func(t *testing.T, h *registryTestHarness, _ any) {
				// The scale-up must have been aborted, leaving the registry in its original state.
				assert.Len(t, h.fr.Shards(), 1, "Shard count should not have changed after a failed scale-up")
			},
		},
		{
			name:              "ShouldGracefullyDrain_OnScaleDown",
			initialShardCount: 3,
			targetShardCount:  2,
			setup: func(h *registryTestHarness) any {
				key := types.FlowKey{ID: "test-flow", Priority: 10}
				require.NoError(t, h.fr.RegisterOrUpdateFlow(types.FlowSpecification{Key: key}),
					"Test setup: failed to register flow")

				allShards := h.fr.Shards()
				require.Len(t, allShards, 3, "Test setup: Should start with 3 shards")
				drainingShard := allShards[2] // The last shard will be chosen for draining.

				mq, err := drainingShard.ManagedQueue(key)
				require.NoError(t, err, "Test setup: failed to get queue on Draining shard")
				item := mocks.NewMockQueueItemAccessor(100, "req", key)
				require.NoError(t, mq.Add(item), "Test setup: failed to add item")
				h.synchronize()

				return &stateForDrainingTest{
					drainingShardID: drainingShard.ID(),
					item:            item,
					flowKey:         key,
				}
			},
			assertions: func(t *testing.T, h *registryTestHarness, state any) {
				testState := state.(*stateForDrainingTest)
				h.synchronize() // Ensure the Draining state has been processed.

				// Assert the intermediate Draining state is correct.
				shards := h.fr.Shards()
				require.Len(t, shards, 3, "Registry should still track the Draining shard until it is empty")
				var drainingShard contracts.RegistryShard
				for _, s := range shards {
					if s.ID() == testState.drainingShardID {
						drainingShard = s
					}
				}
				require.NotNil(t, drainingShard, "Draining shard should still be present in the list")
				assert.False(t, drainingShard.IsActive(), "Target shard should be marked as not active")

				// Assert that the item on the draining shard is still accessible.
				mq, err := drainingShard.ManagedQueue(testState.flowKey)
				require.NoError(t, err, "Should still be able to get queue from draining shard")
				removedItem, err := mq.Remove(testState.item.Handle())
				require.NoError(t, err, "Should be able to remove the item from the draining shard")
				assert.Equal(t, testState.item.OriginalRequest().ID(), removedItem.OriginalRequest().ID(),
					"Correct item was removed")

				// After the item is removed, the shard becomes empty and should be fully garbage collected.
				h.synchronize() // Process the `BecameDrained` event.
				assert.Len(t, h.fr.Shards(), 2,
					"Drained shard was not garbage collected from the registry after becoming empty")
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			h := newRegistryTestHarness(t, harnessOptions{initialShardCount: tc.initialShardCount})
			var state any
			if tc.setup != nil {
				state = tc.setup(h)
			}

			err := h.fr.UpdateShardCount(tc.targetShardCount)

			if tc.expectErrIs != nil || tc.expectErrStr != "" {
				require.Error(t, err, "UpdateShardCount should have returned an error")
				if tc.expectErrIs != nil {
					assert.ErrorIs(t, err, tc.expectErrIs, "Error should be the expected type")
				}
				if tc.expectErrStr != "" {
					assert.Contains(t, err.Error(), tc.expectErrStr, "Error message should contain expected substring")
				}
			} else {
				require.NoError(t, err, "UpdateShardCount should not have returned an error")
			}

			if tc.assertions != nil {
				tc.assertions(t, h, state)
			}
		})
	}
}

// --- Stats and Observability Tests ---

func TestFlowRegistry_Stats_And_ShardStats(t *testing.T) {
	t.Parallel()
	h := newRegistryTestHarness(t, harnessOptions{initialShardCount: 2})
	keyHigh := types.FlowKey{ID: "flow-high", Priority: 10}
	keyLow := types.FlowKey{ID: "flow-low", Priority: 20}
	require.NoError(t, h.fr.RegisterOrUpdateFlow(types.FlowSpecification{Key: keyHigh}))
	require.NoError(t, h.fr.RegisterOrUpdateFlow(types.FlowSpecification{Key: keyLow}))

	// Add items to the queues. Note that these are distributed across 2 shards.
	h.setFlowActive(keyHigh, true) // 100 bytes
	h.setFlowActive(keyLow, true)  // 100 bytes
	h.synchronize()

	// Test global stats.
	stats := h.fr.Stats()
	assert.Equal(t, uint64(2), stats.TotalLen, "Global TotalLen should be correct")
	assert.Equal(t, uint64(200), stats.TotalByteSize, "Global TotalByteSize should be correct")
	assert.Equal(t, uint64(1), stats.PerPriorityBandStats[10].Len, "Global stats for priority 10 Len is incorrect")
	assert.Equal(t, uint64(100), stats.PerPriorityBandStats[10].ByteSize,
		"Global stats for priority 10 ByteSize is incorrect")

	// Test shard stats.
	shardStats := h.fr.ShardStats()
	require.Len(t, shardStats, 2, "Should be stats for 2 shards")

	var totalShardLen, totalShardBytes uint64
	for _, ss := range shardStats {
		totalShardLen += ss.TotalLen
		totalShardBytes += ss.TotalByteSize
	}
	assert.Equal(t, stats.TotalLen, totalShardLen, "Sum of shard lengths should equal global length")
	assert.Equal(t, stats.TotalByteSize, totalShardBytes, "Sum of shard byte sizes should equal global byte size")
}

// --- Garbage Collection Tests ---

func TestFlowRegistry_GarbageCollection_IdleFlows(t *testing.T) {
	t.Parallel()
	key := types.FlowKey{ID: "gc-flow", Priority: 10}
	spec := types.FlowSpecification{Key: key}

	t.Run("ShouldCollectIdleFlow_AfterGracePeriod", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{useFakeClock: true})
		require.NoError(t, h.fr.RegisterOrUpdateFlow(spec), "Test setup: failed to register flow")
		h.synchronize()

		// A new flow is granted a one-cycle grace period. The Mark phase of the first GC cycle will see
		// 1lastActiveGeneration==gcGenerationNewFlow` and "timestamp" it for survival.
		h.waitForGCTick()
		h.assertFlowExists(key, "A new Idle flow must survive its first GC cycle")

		// In the second cycle, the flow is Idle and was not marked in the preceding cycle, so it is collected.
		h.waitForGCTick()
		h.assertFlowDoesNotExist(key, "An Idle flow must be collected after its grace period expires")
	})

	t.Run("ShouldNotCollectFlow_ThatWasRecentlyActive", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{useFakeClock: true})
		require.NoError(t, h.fr.RegisterOrUpdateFlow(spec), "Test setup: failed to register flow")
		h.synchronize()

		h.setFlowActive(key, true)
		h.waitForGCTick()
		h.assertFlowExists(key, "An Active flow should not be collected")
		h.setFlowActive(key, false)

		// The flow is now Idle, but it was marked as Active in the immediately preceding cycle, so it must survive this
		// sweep.
		h.waitForGCTick()
		h.assertFlowExists(key, "A flow must not be collected in the cycle immediately after it becomes Idle")

		// Having been Idle for a full cycle, it will now be collected.
		h.waitForGCTick()
		h.assertFlowDoesNotExist(key, "A flow must be collected after being Idle for a full cycle")
	})

	t.Run("ShouldNotCollectFlow_WhenActivityRacesWithGC", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{useFakeClock: true})
		require.NoError(t, h.fr.RegisterOrUpdateFlow(spec), "Test setup: failed to register flow")
		h.synchronize()

		// Establish a state where the flow is a candidate for GC on the *next* tick.
		// This requires only getting it past its initial grace period.
		h.waitForGCTick()
		h.assertFlowExists(key, "Test setup failed: flow should exist after its grace period")

		// Trigger the race: send a GC tick event but DO NOT wait for it to be processed.
		h.fakeClock.Step(h.config.FlowGCTimeout + time.Millisecond)

		// Immediately send a competing activity signal.
		shard := h.getFirstActiveShard()
		mq, err := shard.ManagedQueue(key)
		require.NoError(t, err, "Test setup: failed to get managed queue for race")
		require.NoError(t, mq.Add(mocks.NewMockQueueItemAccessor(100, "req", key)), "Test setup: failed to add item")

		// Synchronize. The `onGCTick` runs first, but its "Verify" step will see the live item and abort.
		h.synchronize()

		// The flow must survive due to the "Trust but Verify" live check.
		h.assertFlowExists(key, "Flow should survive GC due to the 'Trust but Verify' live check")
	})

	t.Run("ShouldBeBenign_WhenGCingAlreadyDeletedFlow", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{useFakeClock: true})
		require.NoError(t, h.fr.RegisterOrUpdateFlow(spec), "Test setup: failed to register flow")

		h.waitForGCTick()
		h.waitForGCTick() // The second tick collects the flow.
		h.assertFlowDoesNotExist(key, "Test setup failed: flow was not collected")

		// Manually (white-box) call the GC function again for the same key.
		// It should return false and not panic.
		var collected bool
		assert.NotPanics(t, func() {
			h.fr.mu.Lock()
			collected = h.fr.garbageCollectFlowLocked(key)
			h.fr.mu.Unlock()
		}, "GCing a deleted flow should not panic")
		assert.False(t, collected, "GCing a deleted flow should return false")
	})

	t.Run("ShouldAbortGC_WhenCacheIsActive", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{useFakeClock: true})
		require.NoError(t, h.fr.RegisterOrUpdateFlow(spec), "Test setup: failed to register flow")
		h.setFlowActive(key, true) // Make the flow Active.

		// Manually call the GC function. It should check the cache (`isIdle`) first and abort immediately without
		// performing the "Verify" step.
		var collected bool
		assert.NotPanics(t, func() {
			h.fr.mu.Lock()
			collected = h.fr.garbageCollectFlowLocked(key)
			h.fr.mu.Unlock()
		}, "GCing an active flow should not panic")

		assert.False(t, collected, "GCing an active flow should return false")
		h.assertFlowExists(key, "Flow should still exist after aborted GC attempt")
	})
}

func TestFlowRegistry_GarbageCollection_DrainedShards(t *testing.T) {
	t.Parallel()
	h := newRegistryTestHarness(t, harnessOptions{initialShardCount: 2})
	key := types.FlowKey{ID: "test-flow", Priority: 10}
	require.NoError(t, h.fr.RegisterOrUpdateFlow(types.FlowSpecification{Key: key}),
		"Test setup: failed to register flow")

	allShards := h.fr.Shards()
	require.Len(t, allShards, 2, "Test setup: Should start with 2 shards")
	drainingShard := allShards[1]

	// Add an item to the shard that will be drained to prevent its immediate GC upon scale-down.
	mq, err := drainingShard.ManagedQueue(key)
	require.NoError(t, err, "Test setup: failed to get queue on Draining shard")
	item := mocks.NewMockQueueItemAccessor(100, "req", key)
	require.NoError(t, mq.Add(item), "Test setup: failed to add item")
	h.synchronize()

	// Scale down, which marks the shard as Draining.
	require.NoError(t, h.fr.UpdateShardCount(1), "Scaling down should succeed")
	h.synchronize()

	require.Len(t, h.fr.Shards(), 2, "Registry should still track the Draining shard")
	assert.False(t, drainingShard.IsActive(), "Target shard should be marked as Draining")

	// Remove the last item, which triggers the `ShardBecameDrained` signal.
	_, err = mq.Remove(item.Handle())
	require.NoError(t, err, "Failed to remove last item from Draining shard")
	h.synchronize() // Process the `BecameDrained` signal.

	assert.Len(t, h.fr.Shards(), 1, "Drained shard was not garbage collected from the registry")

	// Crucial memory leak check: ensure the shard's ID was purged from all flow states.
	h.fr.mu.Lock()
	defer h.fr.mu.Unlock()
	flowState, exists := h.fr.flowStates[key]
	require.True(t, exists, "Flow state should still exist for the Active flow")
	assert.NotContains(t, flowState.emptyOnShards, drainingShard.ID(),
		"Drained shard ID must be purged from flow state map to prevent memory leaks")
}

// --- Event Handling and State Machine Edge Cases ---

func TestFlowRegistry_EventHandling_StaleSignals(t *testing.T) {
	t.Parallel()

	t.Run("ShouldIgnoreQueueSignal_ForGarbageCollectedFlow", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{useFakeClock: true})
		key := types.FlowKey{ID: "flow", Priority: 10}
		require.NoError(t, h.fr.RegisterOrUpdateFlow(types.FlowSpecification{Key: key}),
			"Test setup: failed to register flow")

		h.setFlowActive(key, true)
		h.synchronize()

		// Manually (white-box) delete the flow from the registry's state map.
		// This simulates a race where a signal is in-flight while the flow is being GC'd.
		h.fr.mu.Lock()
		delete(h.fr.flowStates, key)
		h.fr.mu.Unlock()

		// Now, trigger a `BecameEmpty` signal by making the flow Idle.
		// The `onQueueStateChanged` handler should receive this, find no `flowState`, and return gracefully.
		assert.NotPanics(t, func() {
			h.setFlowActive(key, false)
		}, "Registry should not panic when receiving a signal for a deleted flow")
	})
}

// --- Invariant Panic Tests ---

func TestFlowRegistry_InvariantPanics(t *testing.T) {
	t.Parallel()

	t.Run("ShouldPanic_WhenQueueIsMissingDuringGC", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{})
		key := types.FlowKey{ID: "flow", Priority: 10}
		require.NoError(t, h.fr.RegisterOrUpdateFlow(types.FlowSpecification{Key: key}),
			"Test setup: failed to register flow")

		// Manually corrupt the state by removing the queue from a shard, creating an invariant violation.
		// This is white-box testing to validate a critical failure path.
		h.fr.mu.Lock()
		shard := h.fr.activeShards[0]
		shard.mu.Lock()
		delete(shard.priorityBands[key.Priority].queues, key.ID)
		shard.mu.Unlock()
		h.fr.mu.Unlock()

		// Manually prepare the flow for GC by setting its generation. This bypasses the need for the async `Run` loop,
		// allowing us to catch the panic in this goroutine.
		h.fr.mu.Lock()
		h.fr.flowStates[key].lastActiveGeneration = h.fr.gcGeneration - 1 // Make it a candidate
		h.fr.mu.Unlock()

		assert.Panics(t, func() { // Value assertion is too brittle as it wraps an error.
			h.fr.mu.Lock()
			defer h.fr.mu.Unlock()
			_ = h.fr.garbageCollectFlowLocked(key)
		}, "GC must panic when a queue is missing for a tracked flow state")
	})

	t.Run("ShouldPanic_OnSignalFromUntrackedDrainingShard", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{})
		fakeEvent := &shardStateChangedEvent{
			shardID: "shard-does-not-exist",
			signal:  shardStateSignalBecameDrained,
		}

		assert.PanicsWithValue(t,
			"invariant violation: shard shard-does-not-exist not found in draining map during GC",
			func() {
				h.fr.mu.Lock()
				defer h.fr.mu.Unlock()
				h.fr.onShardStateChanged(fakeEvent)
			}, "Panic should occur when a non-existent shard signals it has drained")
	})

	t.Run("ShouldPanic_OnStatsForUnknownPriority", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{})
		assert.PanicsWithValue(t,
			"invariant violation: priority band (999) stats missing during propagation",
			func() {
				h.fr.propagateStatsDelta(999, 1, 1) // 999 is not a configured priority.
			},
			"Panic should occur when stats are propagated for an unknown priority")
	})

	t.Run("ShouldPanic_OnComponentMismatchDuringSync", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{})
		spec := types.FlowSpecification{Key: types.FlowKey{ID: "flow", Priority: 10}}

		assert.PanicsWithValue(t,
			fmt.Sprintf("invariant violation: shard/queue/policy count mismatch during commit for flow %s", spec.Key),
			func() {
				h.fr.mu.Lock()
				defer h.fr.mu.Unlock()
				// Call with a mismatched number of components (0) vs. shards (1).
				h.fr.applyFlowSynchronizationLocked(spec, []flowComponents{})
			},
			"Panic should occur when component count mismatches shard count")
	})

	t.Run("ShouldPanic_OnStatsAggregation_WithStateMismatch", func(t *testing.T) {
		t.Parallel()
		h := newRegistryTestHarness(t, harnessOptions{})

		// Manually corrupt the state by adding a stats entry for a priority that does not exist in the configuration.
		h.fr.mu.Lock()
		h.fr.perPriorityBandStats[999] = &bandStats{}
		h.fr.mu.Unlock()

		assert.Panics(t, // Value assertion is too brittle as it wraps an error.
			func() { h.fr.Stats() },
			"Stats() must panic when perPriorityBandStats contains a key not in the config")
	})
}
