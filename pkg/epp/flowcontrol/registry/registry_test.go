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
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	intra "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

const (
	// Define standard priorities for testing.
	pHigh = uint(10)
	pMed  = uint(20)
	pLow  = uint(30)

	// Define timeout for synchronization operations to prevent tests from hanging indefinitely.
	syncTimeout = 2 * time.Second
)

// -- Test Fixture ---

// registryTestFixture provides a harness for testing the `FlowRegistry`.
type registryTestFixture struct {
	t      *testing.T
	fr     *FlowRegistry
	ctx    context.Context
	cancel context.CancelFunc
	config *Config
	// fakeClock is non-nil if `useFakeClock` is true, allowing deterministic time control.
	fakeClock *testclock.FakeClock
}

// fixtureOptions allows customization of the test fixture.
type fixtureOptions struct {
	initialShardCount uint
	customConfig      *Config
	// useFakeClock determines if a `FakeClock` is used (for GC tests) or `RealClock` (for stress/blocking tests).
	useFakeClock bool
}

// newRegistryTestFixture creates and starts a new `FlowRegistry` for testing.
func newRegistryTestFixture(t *testing.T, opts fixtureOptions) *registryTestFixture {
	t.Helper()

	if opts.initialShardCount == 0 {
		opts.initialShardCount = 1 // Default to 1 shard if not specified.
	}

	config := opts.customConfig.deepCopy()
	if config == nil {
		// Default configuration if none provided.
		config = &Config{
			// Use a specific timeout; we control the clock or use real time depending on the test.
			FlowGCTimeout: 1 * time.Minute,
			// Ensure a reasonable buffer size for most tests, unless overridden.
			EventChannelBufferSize: 100,
			PriorityBands: []PriorityBandConfig{
				{Priority: pHigh, PriorityName: "High"},
				{Priority: pMed, PriorityName: "Medium"},
				{Priority: pLow, PriorityName: "Low"},
			},
		}
	}

	var clk clock.WithTickerAndDelayedExecution
	var fakeClock *testclock.FakeClock

	if opts.useFakeClock {
		startTime := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
		fakeClock = testclock.NewFakeClock(startTime)
		clk = fakeClock
	} else {
		clk = &clock.RealClock{}
	}

	// Initialize the registry.
	fr, err := NewFlowRegistry(config, opts.initialShardCount, logr.Discard(), WithClock(clk))
	require.NoError(t, err, "NewFlowRegistry should not fail")

	ctx, cancel := context.WithCancel(context.Background())
	// Use a `WaitGroup` to ensure the registry's `Run` loop stops cleanly.
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

	return &registryTestFixture{
		t:         t,
		fr:        fr,
		ctx:       ctx,
		cancel:    cancel,
		config:    config,
		fakeClock: fakeClock,
	}
}

// synchronizeControlPlane blocks until the `FlowRegistry`'s event loop has processed all events currently in the queue.
// This provides a deterministic synchronization point for tests.
func (f *registryTestFixture) synchronizeControlPlane() {
	f.t.Helper()
	// Use a short timeout context for the synchronization itself.
	ctx, cancel := context.WithTimeout(f.ctx, syncTimeout)
	defer cancel()

	doneCh := make(chan struct{})
	evt := &syncEvent{doneCh: doneCh} // syncEvent is defined in `lifecycle.go`.

	// Send the sync event. This might block if the event channel is full.
	select {
	case f.fr.events <- evt:
		// Event sent, now wait for the acknowledgment from the event loop.
		select {
		case <-doneCh:
			// Synchronized successfully.
			return
		case <-ctx.Done():
			f.t.Fatalf("Timeout or cancellation waiting for FlowRegistry synchronization: %v", ctx.Err())
		}
	case <-ctx.Done():
		// This likely means the event channel is full and the context timed out before we could send the event.
		f.t.Fatalf("Timeout or cancellation sending sync event (channel likely full): %v", ctx.Err())
	}
}

// advanceTime moves the fake clock forward. It does NOT synchronize.
func (f *registryTestFixture) advanceTime(d time.Duration) {
	f.t.Helper()
	require.NotNil(f.t, f.fakeClock, "advanceTime requires the fixture to be initialized with useFakeClock=true")
	// Step triggers the timer callbacks (e.g., `AfterFunc` in `gcTracker`), which send events to the channel.
	f.fakeClock.Step(d)
}

// advanceClockToGCTimeout is a convenience helper to step the clock past the configured GC timeout.
func (f *registryTestFixture) advanceClockToGCTimeout() {
	f.t.Helper()
	// Step slightly past the timeout to ensure the timer fires.
	f.advanceTime(f.config.FlowGCTimeout + time.Millisecond)
}

// --- Fixture Helpers (Shard Access and Manipulation) ---

// getShardByID retrieves a specific shard by its ID.
func (f *registryTestFixture) getShardByID(id string) contracts.RegistryShard {
	f.t.Helper()
	shards := f.fr.Shards()
	for _, shard := range shards {
		if shard.ID() == id {
			return shard
		}
	}
	f.t.Fatalf("Shard with ID %s not found", id)
	return nil
}

// getShardByIndex retrieves a specific shard by index.
// NOTE: This relies on the deterministic ordering of `Shards()`, which is guaranteed by the implementation and
// necessary for testing partitioning logic, but should be avoided otherwise.
func (f *registryTestFixture) getShardByIndex(index int) contracts.RegistryShard {
	f.t.Helper()
	shards := f.fr.Shards()
	require.Less(f.t, index, len(shards), "Shard index %d out of bounds (Total Shards: %d)", index, len(shards))
	return shards[index]
}

// addItem adds an item to a specific flow on a specific shard ID.
func (f *registryTestFixture) addItem(
	flowID string,
	priority uint,
	shardID string,
	size uint64,
) types.QueueItemAccessor {
	f.t.Helper()
	shard := f.getShardByID(shardID)
	mq, err := shard.ManagedQueue(flowID, priority)
	require.NoError(f.t, err, "Failed to get queue for flow %s on shard %s", flowID, shardID)

	item := mocks.NewMockQueueItemAccessor(size, fmt.Sprintf("req-%s-%s", flowID, shardID), flowID)
	// Note: Add might fail if the specific queue is draining, which is expected behavior in some tests.
	err = mq.Add(item)
	// We check the shard status in the error message for easier debugging if the Add fails unexpectedly.
	require.NoError(f.t, err, "Failed to add item to queue on shard %s.", shardID)
	return item
}

// removeItem removes an item from a specific flow and priority on a specific shard ID.
func (f *registryTestFixture) removeItem(flowID string, priority uint, shardID string, item types.QueueItemAccessor) {
	f.t.Helper()
	internalShard := f.getShardByID(shardID)

	mq, err := internalShard.ManagedQueue(flowID, priority)
	require.NoError(f.t, err, "Failed to get queue for flow %s at priority %d on shard %s", flowID, priority, shardID)

	_, err = mq.Remove(item.Handle())
	require.NoError(f.t, err, "Failed to remove item from queue on shard %s", shardID)
}

// getActiveShardIDs returns the IDs of all currently active shards in deterministic order.
func (f *registryTestFixture) getActiveShardIDs() []string {
	f.t.Helper()
	var ids []string
	// `Shards()` returns shards in a deterministic order (active first, then draining).
	for _, shard := range f.fr.Shards() {
		if shard.IsActive() {
			ids = append(ids, shard.ID())
		}
	}
	return ids
}

// --- Assertion Helpers (Synchronous) ---
// These helpers check the current state. They assume the caller has synchronized the control plane if necessary.

// assertFlowExistsNow verifies that a flow exists and is active at the expected priority across all active shards.
func (f *registryTestFixture) assertFlowExistsNow(flowID string, expectedPriority uint) {
	f.t.Helper()
	shards := f.fr.Shards()
	require.NotEmpty(f.t, shards, "Registry should have shards")

	activeShardCount := 0
	for _, shard := range shards {
		if !shard.IsActive() {
			continue
		}
		activeShardCount++

		mq, err := shard.ActiveManagedQueue(flowID)
		if assert.NoError(f.t, err, "Flow %s not active on active shard %s", flowID, shard.ID()) {
			assert.Equal(f.t, expectedPriority, mq.FlowQueueAccessor().FlowSpec().Priority,
				"Flow %s active at wrong priority on shard %s", flowID, shard.ID())
		}
	}
	assert.Positive(f.t, activeShardCount, "Registry should have active shards")
}

// assertFlowDoesNotExistNow verifies that a flow is garbage collected and removed from all shards (active and
// draining).
func (f *registryTestFixture) assertFlowDoesNotExistNow(flowID string) {
	f.t.Helper()
	shards := f.fr.Shards()
	for _, shard := range shards {
		// Check both active and draining shards.
		_, err := shard.ActiveManagedQueue(flowID)
		if assert.Error(f.t, err, "Flow %s should not have an active queue on shard %s", flowID, shard.ID()) {
			// Ensure the error is the expected "not found" error.
			assert.ErrorIs(f.t, err, contracts.ErrFlowInstanceNotFound,
				"Unexpected error when checking for flow existence on shard %s", shard.ID())
		}
	}

	// Grey-box check: Ensure it's truly gone from the central tracking map.
	f.fr.mu.Lock()
	_, exists := f.fr.flowStates[flowID]
	f.fr.mu.Unlock()
	assert.False(f.t, exists, "Flow %s should not exist in internal flowStates map", flowID)
}

// assertQueueIsDrainingNow verifies that a specific queue instance exists across all shards and exhibits draining
// behavior.
func (f *registryTestFixture) assertQueueIsDrainingNow(flowID string, priority uint) {
	f.t.Helper()
	shards := f.fr.Shards()
	require.NotEmpty(f.t, shards, "Registry has no shards")

	for _, shard := range shards {
		// 1. Verify the queue instance exists at the expected priority.
		mq, err := shard.ManagedQueue(flowID, priority)
		require.NoError(f.t, err, "Draining queue for flow %s at priority %d not found on shard %s",
			flowID, priority, shard.ID())

		// 2. Verify draining behavior: `Add` should fail.
		item := mocks.NewMockQueueItemAccessor(1, "test-drain", flowID)
		err = mq.Add(item)
		require.Error(f.t, err, "Add to a draining queue should fail on shard %s", shard.ID())
		assert.ErrorIs(f.t, err, contracts.ErrFlowInstanceNotFound,
			"Error type mismatch when adding to draining queue on shard %s", shard.ID())
	}
}

// assertStatsNow verifies that the global registry statistics match the expected values.
func (f *registryTestFixture) assertStatsNow(expectedLen, expectedBytes uint64) {
	f.t.Helper()
	stats := f.fr.Stats()
	assert.Equal(f.t, expectedLen, stats.TotalLen, "Global TotalLen mismatch")
	assert.Equal(f.t, expectedBytes, stats.TotalByteSize, "Global TotalByteSize mismatch")
}

// --- Test Functions: Initialization and Validation ---

func TestFlowRegistry_InitializationErrors(t *testing.T) {
	t.Parallel()

	t.Run("Invalid configuration (no bands)", func(t *testing.T) {
		t.Parallel()
		invalidConfig := &Config{} // No priority bands is invalid.
		_, err := NewFlowRegistry(invalidConfig, 1, logr.Discard())
		assert.Error(t, err, "NewFlowRegistry should fail with an invalid config")
		assert.Contains(t, err.Error(), "master configuration is invalid")
	})

	t.Run("Initial shard count is 0", func(t *testing.T) {
		t.Parallel()
		validConfig := &Config{
			PriorityBands: []PriorityBandConfig{{Priority: 1, PriorityName: "P1"}},
		}
		// `UpdateShardCount` is called internally and rejects 0 shards.
		_, err := NewFlowRegistry(validConfig, 0, logr.Discard())
		assert.Error(t, err, "NewFlowRegistry should fail if initial shard count is 0")
		assert.Contains(t, err.Error(), "failed to initialize shards")
		// Check that the specific error from UpdateShardCount bubbles up.
		// Note: This relies on the implementation detail of the error wrapping structure.
		// If the wrapping changes, this specific check might need adjustment.
		assert.ErrorIs(t, err, contracts.ErrInvalidShardCount, "error type mismatch")
	})
}

// --- Test Functions: Registration and Updates ---

func TestFlowRegistry_RegisterNewFlow(t *testing.T) {
	t.Parallel()
	f := newRegistryTestFixture(t, fixtureOptions{initialShardCount: 2, useFakeClock: true})
	const flowID = "test-flow-new"
	specHigh := types.FlowSpecification{ID: flowID, Priority: pHigh}

	err := f.fr.RegisterOrUpdateFlow(specHigh)
	require.NoError(t, err, "Registering a new flow should succeed")

	f.synchronizeControlPlane()
	f.assertFlowExistsNow(flowID, pHigh)
}

func TestFlowRegistry_RegisterInvalidFlow(t *testing.T) {
	t.Parallel()
	f := newRegistryTestFixture(t, fixtureOptions{initialShardCount: 1, useFakeClock: true})

	t.Run("Empty ID", func(t *testing.T) {
		err := f.fr.RegisterOrUpdateFlow(types.FlowSpecification{ID: "", Priority: pHigh})
		assert.ErrorIs(t, err, contracts.ErrFlowIDEmpty, "Should reject empty FlowID")
	})

	t.Run("Unknown Priority", func(t *testing.T) {
		err := f.fr.RegisterOrUpdateFlow(types.FlowSpecification{ID: "bad-flow", Priority: 999})
		assert.ErrorIs(t, err, contracts.ErrPriorityBandNotFound, "Should reject unknown priority")
	})
}

func TestFlowRegistry_Update_NoOp(t *testing.T) {
	t.Parallel()
	f := newRegistryTestFixture(t, fixtureOptions{initialShardCount: 2, useFakeClock: true})
	const flowID = "test-flow-noop"
	specHigh := types.FlowSpecification{ID: flowID, Priority: pHigh}

	// 1. Initial registration
	require.NoError(t, f.fr.RegisterOrUpdateFlow(specHigh))
	f.synchronizeControlPlane()

	// Capture initial queue instances for comparison later.
	initialQueues := make(map[string]contracts.ManagedQueue)
	for _, shard := range f.fr.Shards() {
		mq, err := shard.ActiveManagedQueue(flowID)
		require.NoError(t, err, "Active queue not found on shard %s before no-op update", shard.ID())
		initialQueues[shard.ID()] = mq
	}

	// 2. No-op update
	err := f.fr.RegisterOrUpdateFlow(specHigh)
	require.NoError(t, err, "No-op update should succeed")

	// 3. Verification
	f.synchronizeControlPlane()
	f.assertFlowExistsNow(flowID, pHigh)
	for _, shard := range f.fr.Shards() {
		mq, err := shard.ActiveManagedQueue(flowID)
		require.NoError(t, err, "Active queue not found on shard %s after no-op update", shard.ID())
		// Crucial check: Ensure the object reference hasn't changed (optimization).
		assert.Same(t, initialQueues[shard.ID()], mq, "Queue instance should not change on no-op update for shard %s",
			shard.ID())
	}
}

func TestFlowRegistry_Update_PriorityChange(t *testing.T) {
	t.Parallel()
	f := newRegistryTestFixture(t, fixtureOptions{initialShardCount: 2, useFakeClock: true})
	const flowID = "test-flow-prio-change"
	specHigh := types.FlowSpecification{ID: flowID, Priority: pHigh}
	specMed := types.FlowSpecification{ID: flowID, Priority: pMed}

	// 1. Setup: Register at `High` priority and add an item.
	require.NoError(t, f.fr.RegisterOrUpdateFlow(specHigh))
	shardID := f.getActiveShardIDs()[0]
	f.addItem(flowID, pHigh, shardID, 100)
	f.synchronizeControlPlane()
	f.assertStatsNow(1, 100)

	// 2. Action: Update priority (`High` -> `Med`).
	err := f.fr.RegisterOrUpdateFlow(specMed)
	require.NoError(t, err, "Priority update should succeed")

	// 3. Verification: `Med` is active, `High` is draining.
	f.synchronizeControlPlane()
	f.assertFlowExistsNow(flowID, pMed)
	f.assertQueueIsDrainingNow(flowID, pHigh)
	// Stats should be unchanged as the item is still in the draining queue.
	f.assertStatsNow(1, 100)
}

func TestFlowRegistry_Update_PriorityChangeReactivation(t *testing.T) {
	t.Parallel()
	f := newRegistryTestFixture(t, fixtureOptions{initialShardCount: 1, useFakeClock: true})
	const flowID = "test-flow-reactivation"
	specHigh := types.FlowSpecification{ID: flowID, Priority: pHigh}
	specMed := types.FlowSpecification{ID: flowID, Priority: pMed}
	shardID := f.getActiveShardIDs()[0] // We only have one shard in this test.

	// 1. Setup: Register `High`, add item, switch to `Med`.
	require.NoError(t, f.fr.RegisterOrUpdateFlow(specHigh))
	f.addItem(flowID, pHigh, shardID, 100)
	require.NoError(t, f.fr.RegisterOrUpdateFlow(specMed), "Priority update to Med should succeed")
	f.synchronizeControlPlane()

	// Verify intermediate state: `High` is draining, `Med` is active.
	f.assertQueueIsDrainingNow(flowID, pHigh)
	// Capture the draining queue instance for later comparison.
	drainingQueueAtHigh, err := f.getShardByID(shardID).ManagedQueue(flowID, pHigh)
	require.NoError(t, err, "Failed to get draining queue at High priority")

	// Add item to the currently active queue (`Med`).
	f.addItem(flowID, pMed, shardID, 50)
	f.synchronizeControlPlane()
	f.assertStatsNow(2, 150)

	// 2. Action: Switch back to `High` (`Med` -> `High`).
	err = f.fr.RegisterOrUpdateFlow(specHigh)
	require.NoError(t, err, "Priority rollback should succeed")

	// 3. Verification: `High` is active, `Med` is draining.
	f.synchronizeControlPlane()
	f.assertFlowExistsNow(flowID, pHigh)
	f.assertQueueIsDrainingNow(flowID, pMed)

	// Crucial check: The queue instance at High priority should be the SAME object that was previously draining.
	activeQueueAtHigh, err := f.getShardByID(shardID).ActiveManagedQueue(flowID)
	require.NoError(t, err, "Failed to get active queue at High priority after reactivation")
	assert.Same(t, drainingQueueAtHigh, activeQueueAtHigh,
		"The previously draining queue instance should be reactivated (optimization check)")
}

// TestFlowRegistry_RegistrationAtomicityOnFailure verifies that a failed registration (e.g., due to plugin failure)
// does not leave the registry in an inconsistent state.
func TestFlowRegistry_RegistrationAtomicityOnFailure(t *testing.T) {
	const pFail = uint(99)
	const flowID = "atomic-flow"
	policyName := intra.RegisteredPolicyName("mutable-policy-for-atomicity-test")

	// 1. Create a mock policy that is initially valid.
	mockPolicy := &frameworkmocks.MockIntraFlowDispatchPolicy{
		RequiredQueueCapabilitiesV: []framework.QueueCapability{}, // Initially, no requirements.
	}

	// 2. Register the mock policy via a closure.
	intra.MustRegisterPolicy(policyName, func() (framework.IntraFlowDispatchPolicy, error) {
		return mockPolicy, nil
	})

	// 3. Start with a valid configuration that uses the initially valid mock policy.
	config := &Config{
		PriorityBands: []PriorityBandConfig{
			{Priority: pHigh, PriorityName: "High"},
			{Priority: pFail, PriorityName: "Mutable-Policy-Band", IntraFlowDispatchPolicy: policyName},
		},
	}
	f := newRegistryTestFixture(t, fixtureOptions{initialShardCount: 1, customConfig: config, useFakeClock: true})

	// 4. AFTER initialization, mutate the mock policy to make it invalid.
	mockPolicy.RequiredQueueCapabilitiesV = []framework.QueueCapability{"impossible-capability"}

	shardID := f.getActiveShardIDs()[0]

	// 5. Setup: Register a flow at a valid priority.
	specHigh := types.FlowSpecification{ID: flowID, Priority: pHigh}
	require.NoError(t, f.fr.RegisterOrUpdateFlow(specHigh), "Initial valid registration should succeed")
	f.synchronizeControlPlane()
	f.assertFlowExistsNow(flowID, pHigh)

	// 6. Action: Attempt to update the flow to the now-failing priority.
	// This will fail during `prepareFlowSynchronization` due to the impossible capability requirement.
	specFail := types.FlowSpecification{ID: flowID, Priority: pFail}
	err := f.fr.RegisterOrUpdateFlow(specFail)

	// 7. Verification: Ensure failure occurred and the state remains unchanged.
	require.Error(t, err, "Registration should fail when policy is incompatible")
	assert.ErrorIs(t, err, contracts.ErrPolicyQueueIncompatible, "Error should be due to incompatibility")

	f.synchronizeControlPlane()
	// The flow should still be active at the original priority (`pHigh`).
	f.assertFlowExistsNow(flowID, pHigh)

	// No artifacts should exist at the failed priority (`pFail`).
	_, err = f.getShardByID(shardID).ManagedQueue(flowID, pFail)
	assert.ErrorIs(t, err, contracts.ErrFlowInstanceNotFound,
		"No queue instance should exist at the failed priority (pFail)")
}

// --- Test Functions: Garbage Collection ---

func TestFlowRegistry_GC_IdleFlow(t *testing.T) {
	t.Parallel()
	f := newRegistryTestFixture(t, fixtureOptions{initialShardCount: 2, useFakeClock: true})
	const flowID = "gc-idle"
	specHigh := types.FlowSpecification{ID: flowID, Priority: pHigh}

	// 1. Register an idle flow (starts GC timer).
	require.NoError(t, f.fr.RegisterOrUpdateFlow(specHigh), "Registering flow should succeed")
	f.synchronizeControlPlane()
	f.assertFlowExistsNow(flowID, pHigh)

	// 2. Advance time past the timeout.
	f.advanceClockToGCTimeout()

	// 3. Verify GC occurred.
	f.synchronizeControlPlane()
	f.assertFlowDoesNotExistNow(flowID)
}

func TestFlowRegistry_GC_ActiveFlowNotCollected(t *testing.T) {
	t.Parallel()
	f := newRegistryTestFixture(t, fixtureOptions{initialShardCount: 2, useFakeClock: true})
	const flowID = "gc-active"
	specHigh := types.FlowSpecification{ID: flowID, Priority: pHigh}

	// 1. Register flow (starts GC timer).
	require.NoError(t, f.fr.RegisterOrUpdateFlow(specHigh), "Registering flow should succeed")
	f.synchronizeControlPlane()

	// 2. Advance time partially.
	f.advanceTime(f.config.FlowGCTimeout / 2)

	// 3. Make the flow active (stops GC timer).
	shardID := f.getActiveShardIDs()[0]
	f.addItem(flowID, pHigh, shardID, 100)
	f.synchronizeControlPlane()
	f.assertStatsNow(1, 100)

	// 4. Advance time past the original timeout.
	f.advanceClockToGCTimeout()

	// 5. Verify flow still exists.
	f.synchronizeControlPlane()
	f.assertFlowExistsNow(flowID, pHigh)
}

func TestFlowRegistry_GC_TimerRestartsWhenIdleAgain(t *testing.T) {
	t.Parallel()
	f := newRegistryTestFixture(t, fixtureOptions{initialShardCount: 1, useFakeClock: true})
	const flowID = "gc-restart"
	specHigh := types.FlowSpecification{ID: flowID, Priority: pHigh}
	shardID := f.getActiveShardIDs()[0] // We only have one shard in this test.

	// 1. Setup: Active flow.
	require.NoError(t, f.fr.RegisterOrUpdateFlow(specHigh), "Registering flow should succeed")
	item := f.addItem(flowID, pHigh, shardID, 100)
	f.synchronizeControlPlane()

	// 2. Make flow idle (restarts GC timer).
	f.removeItem(flowID, pHigh, shardID, item)
	f.synchronizeControlPlane()
	f.assertStatsNow(0, 0)

	// 3. Advance time partially and verify it still exists.
	f.advanceTime(f.config.FlowGCTimeout / 2)
	f.synchronizeControlPlane()
	f.assertFlowExistsNow(flowID, pHigh)

	// 4. Advance time past timeout and verify GC.
	f.advanceClockToGCTimeout()
	f.synchronizeControlPlane()
	f.assertFlowDoesNotExistNow(flowID)
}

// TestFlowRegistry_GC_GenerationHandlesStaleTimers verifies that updates (which increment generation) correctly
// invalidate previous timers and start new ones, even if the flow remains idle.
func TestFlowRegistry_GC_GenerationHandlesStaleTimers(t *testing.T) {
	t.Parallel()
	f := newRegistryTestFixture(t, fixtureOptions{initialShardCount: 1, useFakeClock: true})
	const flowID = "gc-generation"
	specHigh := types.FlowSpecification{ID: flowID, Priority: pHigh}

	// 1. Register flow (Gen 1 timer starts).
	require.NoError(t, f.fr.RegisterOrUpdateFlow(specHigh), "Registering flow should succeed")
	f.synchronizeControlPlane()

	// 2. Advance time partially.
	f.advanceTime(f.config.FlowGCTimeout / 2)

	// 3. Update flow (Gen 1 timer stopped, Gen 2 timer starts).
	// A no-op update still increments the generation and resets the timer.
	require.NoError(t, f.fr.RegisterOrUpdateFlow(specHigh), "No-op update should succeed")
	f.synchronizeControlPlane()

	// 4. Advance time past the *original* timeout (Gen 1 timer would fire here if not stopped).
	f.advanceTime(f.config.FlowGCTimeout/2 + time.Second*2)
	f.synchronizeControlPlane()

	// 5. Verify flow still exists because the Gen 2 timer is still running.
	f.assertFlowExistsNow(flowID, pHigh)

	// 6. Advance time past the *new* timeout (Gen 2 timer fires).
	f.advanceClockToGCTimeout()
	f.synchronizeControlPlane()
	f.assertFlowDoesNotExistNow(flowID)
}

// TestFlowRegistry_GCRace_TimerFiresBeforeActivityProcessed tests the critical race condition where a flow becomes
// active just as its GC timer fires. The timer event might be processed before the activity event.
func TestFlowRegistry_GCRace_TimerFiresBeforeActivityProcessed(t *testing.T) {
	t.Parallel()
	f := newRegistryTestFixture(t, fixtureOptions{initialShardCount: 1, useFakeClock: true})
	const flowID = "gc-race-activity"
	spec := types.FlowSpecification{ID: flowID, Priority: pHigh}
	shardID := f.getActiveShardIDs()[0] // We only have one shard in this test.

	// 1. Register flow (starts GC timer).
	require.NoError(t, f.fr.RegisterOrUpdateFlow(spec), "Registering flow should succeed")
	f.synchronizeControlPlane()

	// 2. Advance clock. This fires the timer and queues the `gcTimerFiredEvent`.
	// The event is now sitting in the channel, waiting to be processed.
	f.advanceClockToGCTimeout()

	// 3. Add an item. This queues the `queueStateSignalBecameNonEmpty` event BEHIND the timer event.
	f.addItem(flowID, pHigh, shardID, 100)

	// 4. Process events. The control loop processes the timer event first.
	// The implementation of `onGCTimerFired` must correctly identify that the flow is no longer idle.
	f.synchronizeControlPlane()

	// 5. Verify the flow was NOT GC'd and stats are correct.
	f.assertFlowExistsNow(flowID, pHigh)
	f.assertStatsNow(1, 100)
}

// TestFlowRegistry_GC_DrainingQueue verifies that a draining queue (due to priority change) is only
// garbage collected when it is empty across ALL shards.
func TestFlowRegistry_GC_DrainingQueue(t *testing.T) {
	t.Parallel()
	f := newRegistryTestFixture(t, fixtureOptions{initialShardCount: 2, useFakeClock: true})
	const flowID = "gc-draining"
	specHigh := types.FlowSpecification{ID: flowID, Priority: pHigh}
	specMed := types.FlowSpecification{ID: flowID, Priority: pMed}

	shardIDs := f.getActiveShardIDs()
	require.Len(t, shardIDs, 2)
	shard0ID, shard1ID := shardIDs[0], shardIDs[1]

	// 1. Setup: Register `High`, add items to both shards.
	require.NoError(t, f.fr.RegisterOrUpdateFlow(specHigh))
	item0 := f.addItem(flowID, pHigh, shard0ID, 100)
	item1 := f.addItem(flowID, pHigh, shard1ID, 50)
	f.synchronizeControlPlane()
	f.assertStatsNow(2, 150)

	// 2. Action: Change priority (`High` -> `Med`). `High` is now draining.
	require.NoError(t, f.fr.RegisterOrUpdateFlow(specMed), "Priority update to Med should succeed")
	f.synchronizeControlPlane()
	f.assertQueueIsDrainingNow(flowID, pHigh)

	// 3. Drain Shard 0.
	f.removeItem(flowID, pHigh, shard0ID, item0)
	f.synchronizeControlPlane()
	f.assertStatsNow(1, 50)

	// 4. Verify: Queue `P_High` should still exist globally because Shard 1 is not empty.
	_, err := f.getShardByID(shard0ID).ManagedQueue(flowID, pHigh)
	assert.NoError(t, err, "Draining queue should still exist on shard 0 (even if empty locally)")
	_, err = f.getShardByID(shard1ID).ManagedQueue(flowID, pHigh)
	assert.NoError(t, err, "Draining queue should still exist on shard 1")

	// 5. Drain Shard 1.
	f.removeItem(flowID, pHigh, shard1ID, item1)
	f.synchronizeControlPlane()
	f.assertStatsNow(0, 0)

	// 6. Verify: Queue `P_High` should be garbage collected globally.
	for _, shardID := range shardIDs {
		_, err := f.getShardByID(shardID).ManagedQueue(flowID, pHigh)
		assert.ErrorIs(t, err, contracts.ErrFlowInstanceNotFound,
			"Draining queue at pHigh was not garbage collected on shard %s", shardID)
	}

	// `P_Med` should still be active.
	f.assertFlowExistsNow(flowID, pMed)
}

// --- Test Functions: Sharding and Scaling ---

// Helper config for sharding tests with defined capacities.
func newShardingTestConfig() *Config {
	return &Config{
		MaxBytes:      100,
		FlowGCTimeout: 1 * time.Minute,
		PriorityBands: []PriorityBandConfig{
			{Priority: pHigh, PriorityName: "High", MaxBytes: 50},
		},
	}
}

func TestFlowRegistry_Sharding_InitializationPartitioning(t *testing.T) {
	t.Parallel()
	// Total 100, Band 50. 3 Shards.
	// Global: 100/3 = 33, Rem 1. Shards get 34, 33, 33.
	// Band: 50/3 = 16, Rem 2. Shards get 17, 17, 16.
	opts := fixtureOptions{
		customConfig:      newShardingTestConfig(),
		initialShardCount: 3,
		useFakeClock:      true,
	}
	f := newRegistryTestFixture(t, opts)
	require.Len(t, f.fr.Shards(), 3)

	// We rely on the deterministic ordering returned by `Shards()` for partitioning validation.
	s0 := f.getShardByIndex(0).Stats()
	assert.Equal(t, uint64(34), s0.TotalCapacityBytes, "Shard 0 Global Capacity")
	assert.Equal(t, uint64(17), s0.PerPriorityBandStats[pHigh].CapacityBytes, "Shard 0 Band Capacity")

	s1 := f.getShardByIndex(1).Stats()
	assert.Equal(t, uint64(33), s1.TotalCapacityBytes, "Shard 1 Global Capacity")
	assert.Equal(t, uint64(17), s1.PerPriorityBandStats[pHigh].CapacityBytes, "Shard 1 Band Capacity")

	s2 := f.getShardByIndex(2).Stats()
	assert.Equal(t, uint64(33), s2.TotalCapacityBytes, "Shard 2 Global Capacity")
	assert.Equal(t, uint64(16), s2.PerPriorityBandStats[pHigh].CapacityBytes, "Shard 2 Band Capacity")
}

func TestFlowRegistry_Sharding_ScaleUp(t *testing.T) {
	t.Parallel()
	// Start with 1, scale to 3.
	opts := fixtureOptions{
		customConfig:      newShardingTestConfig(),
		initialShardCount: 1,
		useFakeClock:      true,
	}
	f := newRegistryTestFixture(t, opts)
	const flowID = "scale-up-flow"
	specHigh := types.FlowSpecification{ID: flowID, Priority: pHigh}

	// Register a flow before scaling to ensure it propagates to new shards.
	require.NoError(t, f.fr.RegisterOrUpdateFlow(specHigh))

	// Action: Scale Up (1 -> 3).
	err := f.fr.UpdateShardCount(3)
	require.NoError(t, err, "Scaling up should succeed")
	f.synchronizeControlPlane()

	// Verification.
	shards := f.fr.Shards()
	require.Len(t, shards, 3)

	// Check activity status and partitioning (re-partitioned to 34, 33, 33).
	for i, shard := range shards {
		assert.True(t, shard.IsActive(), "Shard %d (%s) should be active after scale up", i, shard.ID())
	}
	assert.Equal(t, uint64(34), f.getShardByIndex(0).Stats().TotalCapacityBytes, "Shard 0 Global Capacity")
	assert.Equal(t, uint64(33), f.getShardByIndex(1).Stats().TotalCapacityBytes, "Shard 1 Global Capacity")
	assert.Equal(t, uint64(33), f.getShardByIndex(2).Stats().TotalCapacityBytes, "Shard 2 Global Capacity")

	// Ensure the existing flow was synchronized to the new shards.
	f.assertFlowExistsNow(flowID, pHigh)
}

func TestFlowRegistry_Sharding_ScaleDown_Draining(t *testing.T) {
	t.Parallel()
	// Start with 3, scale to 1.
	opts := fixtureOptions{
		customConfig:      newShardingTestConfig(),
		initialShardCount: 3,
		useFakeClock:      true,
	}
	f := newRegistryTestFixture(t, opts)
	const flowID = "scale-down-flow"
	specHigh := types.FlowSpecification{ID: flowID, Priority: pHigh}

	require.NoError(t, f.fr.RegisterOrUpdateFlow(specHigh), "Initial registration should succeed")
	f.synchronizeControlPlane()

	// Identify shards. We rely on the implementation detail that the *last* shards in the slice are drained.
	shards := f.fr.Shards()
	require.Len(t, shards, 3)
	activeShardID := shards[0].ID()
	drainingShard1ID := shards[1].ID()
	drainingShard2ID := shards[2].ID()

	// Add items specifically to the shards destined for removal.
	// This prevents them from being immediately GC'd upon scale-down, allowing us to verify the draining state.
	f.addItem(flowID, pHigh, drainingShard1ID, 10)
	f.addItem(flowID, pHigh, drainingShard2ID, 10)
	f.synchronizeControlPlane()
	f.assertStatsNow(2, 20)

	// Action: Scale Down (3 -> 1).
	err := f.fr.UpdateShardCount(1)
	require.NoError(t, err, "Scaling down should succeed")
	f.synchronizeControlPlane()

	// Verification.
	currentShards := f.fr.Shards()
	// Total shards should still be 3 (1 active, 2 draining).
	require.Len(t, currentShards, 3)

	assert.True(t, f.getShardByID(activeShardID).IsActive(), "Shard 0 should remain active")
	assert.False(t, f.getShardByID(drainingShard1ID).IsActive(), "Shard 1 should be draining")
	assert.False(t, f.getShardByID(drainingShard2ID).IsActive(), "Shard 2 should be draining")

	// The active shard should have its capacity re-partitioned to the full amount.
	stats0 := f.getShardByID(activeShardID).Stats()
	assert.Equal(t, uint64(100), stats0.TotalCapacityBytes, "Active shard should have full global capacity")
	assert.Equal(t, uint64(50), stats0.PerPriorityBandStats[pHigh].CapacityBytes,
		"Active shard band should have full capacity")
}

// TestFlowRegistry_Sharding_ScaleDown_GCAndPurge verifies that a draining shard is fully garbage collected
// once it becomes empty, and crucially, that its ID is purged from all flow tracking maps (preventing memory leaks).
func TestFlowRegistry_Sharding_ScaleDown_GCAndPurge(t *testing.T) {
	t.Parallel()
	// Start with 2, scale to 1.
	opts := fixtureOptions{
		customConfig:      newShardingTestConfig(),
		initialShardCount: 2,
		useFakeClock:      true,
	}
	f := newRegistryTestFixture(t, opts)
	const flowID = "scale-down-gc"
	specHigh := types.FlowSpecification{ID: flowID, Priority: pHigh}

	require.NoError(t, f.fr.RegisterOrUpdateFlow(specHigh), "Initial registration should succeed")
	f.synchronizeControlPlane()

	// Identify the shard destined for draining (Shard 1, based on implementation detail).
	shards := f.fr.Shards()
	require.Len(t, shards, 2)
	drainingShardID := shards[1].ID()

	// Add an item to the draining shard.
	item := f.addItem(flowID, pHigh, drainingShardID, 10)
	f.synchronizeControlPlane()
	f.assertStatsNow(1, 10)

	// Action 1: Scale Down (2 -> 1).
	require.NoError(t, f.fr.UpdateShardCount(1))
	f.synchronizeControlPlane()

	// Verify Draining State.
	currentShards := f.fr.Shards()
	require.Len(t, currentShards, 2, "Shard count should be 2 (1 active, 1 draining)")
	require.False(t, f.getShardByID(drainingShardID).IsActive(), "Shard 1 should be draining")

	// Action 2: Empty the draining shard. This triggers the ShardBecameDrained signal.
	f.removeItem(flowID, pHigh, drainingShardID, item)
	f.synchronizeControlPlane()

	// Verification: Shard should be completely garbage collected.
	finalShards := f.fr.Shards()
	require.Len(t, finalShards, 1, "Drained shard was not GC'd")
	assert.NotEqual(t, drainingShardID, finalShards[0].ID(), "The wrong shard was garbage collected")

	// Crucial Check: Ensure the decommissioned shard ID is purged from flow state (prevents memory leak).
	f.fr.mu.Lock()
	flowState, ok := f.fr.flowStates[flowID]
	require.True(t, ok, "Flow should still exist")
	assert.Len(t, flowState.activeQueueEmptyOnShards, 1, "Flow state tracking map size incorrect after purge")
	assert.NotContains(t, flowState.activeQueueEmptyOnShards, drainingShardID,
		"Old shard ID still present in tracking map")
	f.fr.mu.Unlock()
}

func TestFlowRegistry_Sharding_ErrorHandling(t *testing.T) {
	t.Parallel()
	opts := fixtureOptions{initialShardCount: 1, useFakeClock: true}
	f := newRegistryTestFixture(t, opts)

	// Test invalid count (0).
	err := f.fr.UpdateShardCount(0)
	assert.Error(t, err, "Updating shard count to 0 should fail")
	assert.ErrorIs(t, err, contracts.ErrInvalidShardCount)

	// Test no-op update.
	err = f.fr.UpdateShardCount(1)
	assert.NoError(t, err, "Updating shard count to the same value should be a successful no-op")
}

// --- Test Functions: Statistics and Concurrency ---

func TestFlowRegistry_StatsAggregation(t *testing.T) {
	t.Parallel()
	config := &Config{
		MaxBytes: 10000,
		PriorityBands: []PriorityBandConfig{
			{Priority: pHigh, PriorityName: "High", MaxBytes: 5000},
			{Priority: pLow, PriorityName: "Low", MaxBytes: 3000},
		},
	}
	f := newRegistryTestFixture(t, fixtureOptions{initialShardCount: 2, customConfig: config, useFakeClock: true})
	// Get shard IDs for targeted additions.
	shardIDs := f.getActiveShardIDs()
	require.Len(t, shardIDs, 2)
	s0ID, s1ID := shardIDs[0], shardIDs[1]

	require.NoError(t, f.fr.RegisterOrUpdateFlow(types.FlowSpecification{ID: "flow1-high", Priority: pHigh}),
		"Registering flow1-high should succeed")
	require.NoError(t, f.fr.RegisterOrUpdateFlow(types.FlowSpecification{ID: "flow2-low", Priority: pLow}),
		"Registering flow2-low should succeed")

	// Add items distributed across flows, priorities, and shards.
	f.addItem("flow1-high", pHigh, s0ID, 100)
	f.addItem("flow1-high", pHigh, s1ID, 50)
	f.addItem("flow2-low", pLow, s0ID, 150)
	f.addItem("flow2-low", pLow, s0ID, 200) // Two items on the same flow/shard

	f.synchronizeControlPlane()

	// Verify Global Stats.
	f.assertStatsNow(4, 500) // 100+50+150+200 = 500
	stats := f.fr.Stats()
	assert.Equal(t, uint64(10000), stats.TotalCapacityBytes, "Global TotalCapacityBytes mismatch")

	// Verify Per-Priority Stats.
	statsHigh := stats.PerPriorityBandStats[pHigh]
	assert.Equal(t, uint64(5000), statsHigh.CapacityBytes, "High Priority CapacityBytes mismatch")
	assert.Equal(t, uint64(2), statsHigh.Len, "High Priority Len mismatch")
	assert.Equal(t, uint64(150), statsHigh.ByteSize, "High Priority ByteSize mismatch")

	statsLow := stats.PerPriorityBandStats[pLow]
	assert.Equal(t, uint64(3000), statsLow.CapacityBytes, "Low Priority CapacityBytes mismatch")
	assert.Equal(t, uint64(2), statsLow.Len, "Low Priority Len mismatch")
	assert.Equal(t, uint64(350), statsLow.ByteSize, "Low Priority ByteSize mismatch")

	// Verify Shard Stats. We use the index here specifically to verify the partitioning logic.
	// ShardStats() order matches the internal activeShards slice order.
	shardStats := f.fr.ShardStats()
	require.Len(t, shardStats, 2, "Expected 2 shard stats entries")

	// Find the stats corresponding to the IDs (order is deterministic).
	var statsS0, statsS1 contracts.ShardStats
	allShards := f.fr.Shards()
	if allShards[0].ID() == s0ID {
		statsS0, statsS1 = shardStats[0], shardStats[1]
	} else {
		// This branch should technically not happen given the deterministic initialization, but added for robustness.
		statsS1, statsS0 = shardStats[0], shardStats[1]
	}

	// Shard 0 Stats (3 items, 450 bytes)
	assert.Equal(t, uint64(3), statsS0.TotalLen, "Shard 0 TotalLen mismatch")
	assert.Equal(t, uint64(450), statsS0.TotalByteSize, "Shard 0 TotalByteSize mismatch")
	assert.Equal(t, uint64(5000), statsS0.TotalCapacityBytes, "Shard 0 TotalCapacityBytes mismatch") // 10000 / 2
	assert.Equal(t, uint64(1), statsS0.PerPriorityBandStats[pHigh].Len, "Shard 0 High Priority Len mismatch")
	assert.Equal(t, uint64(2500), statsS0.PerPriorityBandStats[pHigh].CapacityBytes,
		"Shard 0 High Priority CapacityBytes mismatch") // 5000 / 2
	assert.Equal(t, uint64(2), statsS0.PerPriorityBandStats[pLow].Len, "Shard 0 Low Priority Len mismatch")
	assert.Equal(t, uint64(1500), statsS0.PerPriorityBandStats[pLow].CapacityBytes,
		"Shard 0 Low Priority CapacityBytes mismatch") // 3000 / 2

	// Shard 1 Stats (1 item, 50 bytes)
	assert.Equal(t, uint64(1), statsS1.TotalLen, "Shard 1 TotalLen mismatch")
	assert.Equal(t, uint64(50), statsS1.TotalByteSize, "Shard 1 TotalByteSize mismatch")
}

// TestFlowRegistry_Backpressure verifies that the data path (e.g., `addItem`) blocks if the control plane event channel
// is full. This is critical for ensuring exactly-once event delivery required by the GC system.
func TestFlowRegistry_Backpressure(t *testing.T) {
	t.Parallel()

	// Configure a minimal buffer size.
	config := &Config{
		EventChannelBufferSize: 1, // Set buffer to 1
		PriorityBands: []PriorityBandConfig{
			{Priority: pHigh, PriorityName: "High"},
		},
	}
	// Use `RealClock` (`useFakeClock: false`) as we are testing blocking behavior over time.
	f := newRegistryTestFixture(t, fixtureOptions{initialShardCount: 1, customConfig: config, useFakeClock: false})

	const flow1, flow2, flow3 = "f1", "f2", "f3"
	require.NoError(t, f.fr.RegisterOrUpdateFlow(types.FlowSpecification{ID: flow1, Priority: pHigh}), "Registering flow1 should succeed")
	require.NoError(t, f.fr.RegisterOrUpdateFlow(types.FlowSpecification{ID: flow2, Priority: pHigh}), "Registering flow2 should succeed")
	require.NoError(t, f.fr.RegisterOrUpdateFlow(types.FlowSpecification{ID: flow3, Priority: pHigh}), "Registering flow3 should succeed")
	f.synchronizeControlPlane()

	// Get the shard and queue instances *before* pausing the control plane.
	shard := f.getShardByID(f.getActiveShardIDs()[0])
	mq1, err := shard.ManagedQueue(flow1, pHigh)
	require.NoError(t, err)
	mq2, err := shard.ManagedQueue(flow2, pHigh)
	require.NoError(t, err)
	mq3, err := shard.ManagedQueue(flow3, pHigh)
	require.NoError(t, err)

	// 1. Pause the control plane event loop by acquiring the main lock.
	// This prevents the loop from consuming events from the channel.
	f.fr.mu.Lock()

	// 2. Fill the buffer.
	// The first `addItem` generates a BecameNonEmpty event, filling the buffer (size 1).
	item1 := mocks.NewMockQueueItemAccessor(10, "req-1", flow1)
	require.NoError(t, mq1.Add(item1))

	// The second `addItem` generates another event. The `addItem` call itself returns quickly, but the underlying
	// `managedQueue`'s send to the channel will eventually fill the remaining space or block.
	item2 := mocks.NewMockQueueItemAccessor(10, "req-2", flow2)
	require.NoError(t, mq2.Add(item2))

	// Wait briefly to ensure the internal atomic operations and channel sends have occurred.
	time.Sleep(20 * time.Millisecond)

	// 3. Attempt an operation that generates a third event. This MUST block.
	operationCompleted := make(chan struct{})
	go func() {
		// This call will eventually block inside `propagateStatsDelta` when trying to send the BecameNonEmpty event because
		// the channel is full and the consumer (event loop) is paused.
		item3 := mocks.NewMockQueueItemAccessor(10, "req-3", flow3)
		err := mq3.Add(item3)
		assert.NoError(t, err, "Add in goroutine failed")
		close(operationCompleted) // Signal completion if it unblocks.
	}()

	// Verify that it blocks (does not complete within a short duration).
	select {
	case <-operationCompleted:
		f.fr.mu.Unlock() // Ensure unlock even on failure
		t.Fatal("addItem did not block when the event channel was full. Backpressure failed.")
	case <-time.After(100 * time.Millisecond):
		// Success: The operation is blocked as expected.
	}

	// 4. Unpause the control plane.
	f.fr.mu.Unlock()

	// 5. Verify the blocked operation completes.
	select {
	case <-operationCompleted:
		// Success: The operation unblocked after the control plane resumed.
	case <-time.After(syncTimeout):
		t.Fatal("addItem remained blocked after the control plane resumed.")
	}

	// 6. Verify system consistency.
	f.synchronizeControlPlane()
	f.assertStatsNow(3, 30)
}

// TestFlowRegistry_ConcurrencyStress performs concurrent administrative operations (registration, scaling) and data
// path operations (enqueue) to verify thread safety and statistical consistency.
func TestFlowRegistry_ConcurrencyStress(t *testing.T) {
	t.Parallel()
	const initialShards = 2
	f := newRegistryTestFixture(t, fixtureOptions{
		initialShardCount: initialShards,
		useFakeClock:      false, // Use `RealClock` for stress testing concurrency primitives.
		customConfig: &Config{
			PriorityBands: []PriorityBandConfig{
				{Priority: pHigh, PriorityName: "High"},
				{Priority: pMed, PriorityName: "Medium"},
				{Priority: pLow, PriorityName: "Low"},
			},
		},
	})

	const numAdminRoutines = 10
	const adminOpsPerRoutine = 50
	const numDataRoutines = 20
	const dataOpsPerRoutine = 200

	var wg sync.WaitGroup

	// 1. Concurrent Administrative Operations (Registration/Updates).
	wg.Add(numAdminRoutines)
	for i := range numAdminRoutines {
		go func(writerID int) {
			defer wg.Done()
			for j := range adminOpsPerRoutine {
				// Cycle through a few flow IDs to generate updates, not just new registrations.
				flowID := fmt.Sprintf("flow-admin-%d", (writerID+j)%5)
				// Alternate priorities to trigger draining/reactivation logic concurrently.
				priority := pHigh
				if j%2 == 0 {
					priority = pLow
				}
				spec := types.FlowSpecification{ID: flowID, Priority: priority}
				err := f.fr.RegisterOrUpdateFlow(spec)
				assert.NoError(t, err, "Concurrent RegisterOrUpdateFlow failed")
			}
		}(i)
	}

	// 2. Concurrent Shard Scaling.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := range 20 {
			// Cycle shard count between 2, 3, and 4.
			newCount := uint(initialShards + (j % 3))
			err := f.fr.UpdateShardCount(newCount)
			assert.NoError(t, err, "Concurrent UpdateShardCount failed")
			// Small sleep to allow other operations to proceed and increase contention.
			time.Sleep(1 * time.Millisecond)
		}
	}()

	// 3. Concurrent Data Path Operations (Enqueue).
	const dataFlowID = "data-path-flow"
	require.NoError(t, f.fr.RegisterOrUpdateFlow(types.FlowSpecification{ID: dataFlowID, Priority: pMed}))

	wg.Add(numDataRoutines)
	for i := range numDataRoutines {
		go func(routineID int) {
			defer wg.Done()
			for j := range dataOpsPerRoutine {
				// Attempt to enqueue on an available shard.
				shards := f.fr.Shards()
				if len(shards) == 0 {
					continue
				}
				// Select a shard based on the iteration count.
				shard := shards[j%len(shards)]

				// We must handle potential errors gracefully, as the shard or the flow might be draining due to concurrent
				// administrative operations (scaling or priority updates).
				mq, err := shard.ActiveManagedQueue(dataFlowID)
				if err != nil {
					// Flow might not exist on this shard yet, or the shard might be draining.
					continue
				}
				item := mocks.NewMockQueueItemAccessor(10, fmt.Sprintf("req-%d-%d", routineID, j), dataFlowID)
				// Add might also fail if the queue transitions to draining just before the call.
				_ = mq.Add(item)
			}
		}(i)
	}

	wg.Wait()

	// 4. Final Consistency Check.
	f.synchronizeControlPlane()

	globalStats := f.fr.Stats()
	shardStats := f.fr.ShardStats()

	// The critical check: Ensure the sum of the parts (shards) equals the whole (global stats).
	var aggregatedLen, aggregatedBytes uint64
	for _, s := range shardStats {
		aggregatedLen += s.TotalLen
		aggregatedBytes += s.TotalByteSize
	}

	assert.Equal(t, aggregatedLen, globalStats.TotalLen, "Global length should match the sum of shard lengths")
	assert.Equal(t, aggregatedBytes, globalStats.TotalByteSize,
		"Global byte size should match the sum of shard byte sizes")
}
