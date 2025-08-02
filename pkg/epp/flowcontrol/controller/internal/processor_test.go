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

//
// A Note on the Testing Strategy for `ShardProcessor`
//
// The `ShardProcessor` is a complex concurrent orchestrator. Testing it with concrete implementations would lead to
// flaky, non-deterministic tests. Therefore, we use a high-fidelity `testHarness` with stateful mocks to enable
// reliable and deterministic testing. This is a deliberate and necessary choice for several key reasons:
//
// 1.  Deterministic Race Simulation: The harness allows us to pause mock execution at critical moments, making it
//     possible to deterministically simulate and verify the processor's behavior during race conditions (e.g., the
//     dispatch vs. expiry race). This is impossible with concrete implementations without resorting to unreliable
//     sleeps.
//
// 2.  Failure Mode Simulation: We can trigger specific, on-demand errors from dependencies to verify the processor's
//     resilience and complex error-handling logic (e.g., the `errIntraFlow` circuit breaker).
//
// 3.  Interaction and Isolation Testing: Mocks allow us to isolate the `ShardProcessor` from its dependencies. This
//     ensures that tests are verifying the processor's orchestration logic (i.e., that it calls its dependencies
//     correctly) and are not affected by confounding bugs in those dependencies.
//
// In summary, this strategy is a prerequisite for reliably testing a concurrent engine, not just a simple data
// structure.
//

package internal

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	typesmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

const (
	testTTL         = 1 * time.Minute
	testShortTTL    = 20 * time.Millisecond
	testCleanupTick = 10 * time.Millisecond
	testWaitTimeout = 1 * time.Second
)

var testFlow = types.FlowSpecification{ID: "flow-a", Priority: 10}

// TestMain sets up the logger for all tests in the package.
func TestMain(m *testing.M) {
	log.SetLogger(zap.New(zap.WriteTo(os.Stderr), zap.UseDevMode(true)))
	os.Exit(m.Run())
}

// mockClock allows for controlling time in tests.
type mockClock struct {
	mu          sync.Mutex
	currentTime time.Time
}

func newMockClock() *mockClock {
	return &mockClock{currentTime: time.Now()}
}

func (c *mockClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.currentTime
}

func (c *mockClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.currentTime = c.currentTime.Add(d)
}

// testHarness provides a unified, mock-based testing environment for the ShardProcessor. It centralizes all mock state
// and provides helper methods for setting up tests and managing the processor's lifecycle.
type testHarness struct {
	t *testing.T
	*mocks.MockRegistryShard

	// Concurrency and Lifecycle
	ctx         context.Context
	cancel      context.CancelFunc
	wg          sync.WaitGroup
	startSignal chan struct{}

	// Core components under test
	processor *ShardProcessor
	mockClock *mockClock
	logger    logr.Logger

	// --- Centralized Mock State ---
	// The harness's mutex protects the single source of truth for all mock state.
	mu            sync.Mutex
	queues        map[string]*mocks.MockManagedQueue // Key: `flowID`
	priorityFlows map[uint][]string                  // Key: `priority`, Val: slice of `flowIDs`

	// Customizable policy logic for tests to override.
	interFlowPolicySelectQueue func(band framework.PriorityBandAccessor) (framework.FlowQueueAccessor, error)
	intraFlowPolicySelectItem  func(fqa framework.FlowQueueAccessor) (types.QueueItemAccessor, error)
}

// newTestHarness creates and wires up a complete testing harness.
func newTestHarness(t *testing.T, expiryCleanupInterval time.Duration) *testHarness {
	t.Helper()
	h := &testHarness{
		t:                 t,
		MockRegistryShard: &mocks.MockRegistryShard{},
		mockClock:         newMockClock(),
		logger:            logr.Discard(),
		startSignal:       make(chan struct{}),
		queues:            make(map[string]*mocks.MockManagedQueue),
		priorityFlows:     make(map[uint][]string),
	}

	// Wire up the harness to provide the mock implementations for the shard's dependencies.
	h.ActiveManagedQueueFunc = h.activeManagedQueue
	h.ManagedQueueFunc = h.managedQueue
	h.AllOrderedPriorityLevelsFunc = h.allOrderedPriorityLevels
	h.PriorityBandAccessorFunc = h.priorityBandAccessor
	h.InterFlowDispatchPolicyFunc = h.interFlowDispatchPolicy
	h.IntraFlowDispatchPolicyFunc = h.intraFlowDispatchPolicy

	// Provide a default stats implementation that is effectively infinite.
	h.StatsFunc = func() contracts.ShardStats {
		return contracts.ShardStats{
			TotalCapacityBytes: 1e9,
			PerPriorityBandStats: map[uint]contracts.PriorityBandStats{
				testFlow.Priority: {CapacityBytes: 1e9},
			},
		}
	}

	// Use a default pass-through filter.
	filter := func(
		ctx context.Context,
		band framework.PriorityBandAccessor,
		logger logr.Logger,
	) (map[string]struct{}, bool) {
		return nil, false
	}
	h.processor = NewShardProcessor(h, filter, h.mockClock, expiryCleanupInterval, h.logger)
	require.NotNil(t, h.processor, "NewShardProcessor should not return nil")
	return h
}

// --- Test Lifecycle and Helpers ---

// Start prepares the processor to run in a background goroutine but pauses it until Go() is called.
func (h *testHarness) Start() {
	h.t.Helper()
	h.ctx, h.cancel = context.WithCancel(context.Background())
	h.wg.Add(1)
	go func() {
		defer h.wg.Done()
		<-h.startSignal // Wait for the signal to begin execution.
		h.processor.Run(h.ctx)
	}()
}

// Go unpauses the processor's main Run loop.
func (h *testHarness) Go() {
	h.t.Helper()
	close(h.startSignal)
}

// Stop gracefully shuts down the processor and waits for it to terminate.
func (h *testHarness) Stop() {
	h.t.Helper()
	if h.cancel != nil {
		h.cancel()
	}
	h.wg.Wait()
}

// waitForFinalization blocks until an item is finalized or a timeout is reached.
func (h *testHarness) waitForFinalization(item *flowItem) (types.QueueOutcome, error) {
	h.t.Helper()
	select {
	case <-item.Done():
		return item.FinalState()
	case <-time.After(testWaitTimeout):
		h.t.Fatalf("Timed out waiting for item %q to be finalized", item.OriginalRequest().ID())
		return types.QueueOutcomeNotYetFinalized, nil
	}
}

// newTestItem creates a new flowItem for testing purposes.
func (h *testHarness) newTestItem(id, flowID string, ttl time.Duration) *flowItem {
	h.t.Helper()
	ctx := log.IntoContext(context.Background(), h.logger)
	req := typesmocks.NewMockFlowControlRequest(100, id, flowID, ctx)
	return NewItem(req, ttl, h.mockClock.Now())
}

// addQueue centrally registers a new mock queue for a given flow, ensuring all harness components are aware of it.
func (h *testHarness) addQueue(spec types.FlowSpecification) *mocks.MockManagedQueue {
	h.t.Helper()
	h.mu.Lock()
	defer h.mu.Unlock()

	mockQueue := &mocks.MockManagedQueue{FlowSpecV: spec}
	h.queues[spec.ID] = mockQueue

	// Add the `flowID` to the correct priority band, creating the band if needed.
	h.priorityFlows[spec.Priority] = append(h.priorityFlows[spec.Priority], spec.ID)

	return mockQueue
}

// --- Mock Interface Implementations ---

// activeManagedQueue provides the mock implementation for the `RegistryShard` interface.
func (h *testHarness) activeManagedQueue(flowID string) (contracts.ManagedQueue, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if q, ok := h.queues[flowID]; ok {
		return q, nil
	}
	return nil, fmt.Errorf("test setup error: no active queue for flow %q", flowID)
}

// managedQueue provides the mock implementation for the `RegistryShard` interface.
func (h *testHarness) managedQueue(flowID string, priority uint) (contracts.ManagedQueue, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if q, ok := h.queues[flowID]; ok && q.FlowSpec().Priority == priority {
		return q, nil
	}
	return nil, fmt.Errorf("test setup error: no queue for %q at priority %d", flowID, priority)
}

// allOrderedPriorityLevels provides the mock implementation for the `RegistryShard` interface.
func (h *testHarness) allOrderedPriorityLevels() []uint {
	h.mu.Lock()
	defer h.mu.Unlock()
	prios := make([]uint, 0, len(h.priorityFlows))
	for p := range h.priorityFlows {
		prios = append(prios, p)
	}
	slices.Sort(prios)
	return prios
}

// priorityBandAccessor provides the mock implementation for the `RegistryShard` interface. It acts as a factory for a
// fully-configured, stateless mock that is safe for concurrent use.
func (h *testHarness) priorityBandAccessor(p uint) (framework.PriorityBandAccessor, error) {
	band := &frameworkmocks.MockPriorityBandAccessor{PriorityV: p}

	// Safely get a snapshot of the flow IDs under a lock.
	h.mu.Lock()
	flowIDsForPriority := h.priorityFlows[p]
	h.mu.Unlock()

	// Configure the mock's behavior with a closure that reads from the harness's centralized, thread-safe state.
	band.IterateQueuesFunc = func(cb func(fqa framework.FlowQueueAccessor) bool) {
		// This closure safely iterates over the snapshot of flow IDs.
		for _, id := range flowIDsForPriority {
			// Get the queue using the thread-safe `managedQueue` method.
			q, err := h.managedQueue(id, p)
			if err == nil && q != nil {
				mq := q.(*mocks.MockManagedQueue)
				if !cb(mq.FlowQueueAccessor()) {
					break
				}
			}
		}
	}
	return band, nil
}

// interFlowDispatchPolicy provides the mock implementation for the `contracts.RegistryShard` interface.
func (h *testHarness) interFlowDispatchPolicy(p uint) (framework.InterFlowDispatchPolicy, error) {
	policy := &frameworkmocks.MockInterFlowDispatchPolicy{}
	// If the test provided a custom implementation, use it.
	if h.interFlowPolicySelectQueue != nil {
		policy.SelectQueueFunc = h.interFlowPolicySelectQueue
		return policy, nil
	}

	// Otherwise, use a default implementation that selects the first non-empty queue.
	policy.SelectQueueFunc = func(band framework.PriorityBandAccessor) (framework.FlowQueueAccessor, error) {
		var selectedQueue framework.FlowQueueAccessor
		band.IterateQueues(func(fqa framework.FlowQueueAccessor) bool {
			if fqa.Len() > 0 {
				selectedQueue = fqa
				return false // stop iterating
			}
			return true // continue
		})
		return selectedQueue, nil
	}
	return policy, nil
}

// intraFlowDispatchPolicy provides the mock implementation for the `contracts.RegistryShard` interface.
func (h *testHarness) intraFlowDispatchPolicy(flowID string, priority uint) (framework.IntraFlowDispatchPolicy, error) {
	policy := &frameworkmocks.MockIntraFlowDispatchPolicy{}
	// If the test provided a custom implementation, use it.
	if h.intraFlowPolicySelectItem != nil {
		policy.SelectItemFunc = h.intraFlowPolicySelectItem
		return policy, nil
	}

	// Otherwise, use a default implementation that selects the head of the queue.
	policy.SelectItemFunc = func(fqa framework.FlowQueueAccessor) (types.QueueItemAccessor, error) {
		return fqa.PeekHead()
	}
	return policy, nil
}

// TestShardProcessor contains all tests for the `ShardProcessor`.
func TestShardProcessor(t *testing.T) {
	t.Parallel()

	// Lifecycle tests use the processor's main `Run` loop to verify the complete end-to-end lifecycle of a request, from
	// `Enqueue` to its final outcome.
	t.Run("Lifecycle", func(t *testing.T) {
		t.Parallel()

		t.Run("should dispatch item successfully", func(t *testing.T) {
			t.Parallel()
			// --- ARRANGE ---
			h := newTestHarness(t, testCleanupTick)
			item := h.newTestItem("req-dispatch-success", testFlow.ID, testTTL)
			h.addQueue(types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority})

			// --- ACT ---
			h.Start()
			defer h.Stop()
			h.processor.Enqueue(item)
			h.Go()

			// --- ASSERT ---
			outcome, err := h.waitForFinalization(item)
			assert.Equal(t, types.QueueOutcomeDispatched, outcome, "The final outcome should be Dispatched")
			require.NoError(t, err, "A successful dispatch should not produce an error")
		})

		t.Run("should reject item when at capacity", func(t *testing.T) {
			t.Parallel()
			// --- ARRANGE ---
			h := newTestHarness(t, testCleanupTick)
			item := h.newTestItem("req-capacity-reject", testFlow.ID, testTTL)
			h.addQueue(types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority})
			h.StatsFunc = func() contracts.ShardStats {
				return contracts.ShardStats{PerPriorityBandStats: map[uint]contracts.PriorityBandStats{
					testFlow.Priority: {CapacityBytes: 50}, // 50 is less than item size of 100
				}}
			}

			// --- ACT ---
			h.Start()
			defer h.Stop()
			h.processor.Enqueue(item)
			h.Go()

			// --- ASSERT ---
			outcome, err := h.waitForFinalization(item)
			assert.Equal(t, types.QueueOutcomeRejectedCapacity, outcome, "The final outcome should be RejectedCapacity")
			require.Error(t, err, "A capacity rejection should produce an error")
			assert.ErrorIs(t, err, types.ErrQueueAtCapacity, "The error should be of type ErrQueueAtCapacity")
		})

		t.Run("should reject item on registry lookup failure", func(t *testing.T) {
			t.Parallel()
			// --- ARRANGE ---
			h := newTestHarness(t, testCleanupTick)
			item := h.newTestItem("req-lookup-fail-reject", testFlow.ID, testTTL)
			registryErr := errors.New("test registry lookup error")
			h.ActiveManagedQueueFunc = func(flowID string) (contracts.ManagedQueue, error) {
				return nil, registryErr
			}

			// --- ACT ---
			h.Start()
			defer h.Stop()
			h.processor.Enqueue(item)
			h.Go()

			// --- ASSERT ---
			outcome, err := h.waitForFinalization(item)
			assert.Equal(t, types.QueueOutcomeRejectedOther, outcome, "The final outcome should be RejectedOther")
			require.Error(t, err, "A rejection from a registry failure should produce an error")
			assert.ErrorIs(t, err, registryErr, "The underlying registry error should be preserved")
		})

		t.Run("should reject item if enqueued during shutdown", func(t *testing.T) {
			t.Parallel()
			// --- ARRANGE ---
			h := newTestHarness(t, testCleanupTick)
			item := h.newTestItem("req-shutdown-reject", testFlow.ID, testTTL)
			h.addQueue(types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority})

			// --- ACT ---
			h.Start()
			h.Go()
			// Stop the processor, then immediately try to enqueue.
			h.Stop()
			h.processor.Enqueue(item)

			// --- ASSERT ---
			outcome, err := h.waitForFinalization(item)
			assert.Equal(t, types.QueueOutcomeRejectedOther, outcome, "The outcome should be RejectedOther")
			require.Error(t, err, "An eviction on shutdown should produce an error")
			assert.ErrorIs(t, err, types.ErrFlowControllerShutdown, "The error should be of type ErrFlowControllerShutdown")
		})

		t.Run("should evict item on TTL expiry via background cleanup", func(t *testing.T) {
			t.Parallel()
			// --- ARRANGE ---
			h := newTestHarness(t, testCleanupTick)
			item := h.newTestItem("req-expired-evict", testFlow.ID, testShortTTL)
			h.addQueue(types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority})

			// --- ACT ---
			h.Start()
			defer h.Stop()
			h.processor.Enqueue(item)
			h.Go()

			// Let time pass for the item to expire and for the background cleanup to run.
			h.mockClock.Advance(testShortTTL * 2)
			time.Sleep(testCleanupTick * 3) // Allow the cleanup goroutine time to run.

			// --- ASSERT ---
			outcome, err := h.waitForFinalization(item)
			assert.Equal(t, types.QueueOutcomeEvictedTTL, outcome, "The final outcome should be EvictedTTL")
			require.Error(t, err, "A TTL eviction should produce an error")
			assert.ErrorIs(t, err, types.ErrTTLExpired, "The error should be of type ErrTTLExpired")
		})

		t.Run("should evict item at moment of dispatch if TTL has expired", func(t *testing.T) {
			t.Parallel()
			// --- ARRANGE ---
			h := newTestHarness(t, 1*time.Hour) // Disable background cleanup to isolate dispatch logic.
			item := h.newTestItem("req-expired-dispatch-evict", testFlow.ID, testShortTTL)
			mockQueue := h.addQueue(types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority})
			require.NoError(t, mockQueue.Add(item), "Adding item to mock queue should not fail")

			// Have the policy select the item, but then advance time so it's expired by the time dispatchItem actually runs.
			h.interFlowPolicySelectQueue = func(band framework.PriorityBandAccessor) (framework.FlowQueueAccessor, error) {
				h.mockClock.Advance(testShortTTL * 2)
				return mockQueue.FlowQueueAccessor(), nil
			}

			// --- ACT ---
			h.Start()
			defer h.Stop()
			h.Go()

			// The run loop will pick up the item and attempt dispatch, which will fail internally.
			// We need a small sleep to allow the non-blocking run loop to process.
			time.Sleep(50 * time.Millisecond)

			// --- ASSERT ---
			outcome, err := h.waitForFinalization(item)
			assert.Equal(t, types.QueueOutcomeEvictedTTL, outcome, "The final outcome should be EvictedTTL")
			require.Error(t, err, "An eviction at dispatch time should produce an error")
			assert.ErrorIs(t, err, types.ErrTTLExpired, "The error should be of type ErrTTLExpired")
		})

		t.Run("should evict item on context cancellation", func(t *testing.T) {
			t.Parallel()
			// --- ARRANGE ---
			h := newTestHarness(t, testCleanupTick)
			ctx, cancel := context.WithCancel(context.Background())
			req := typesmocks.NewMockFlowControlRequest(100, "req-ctx-cancel", testFlow.ID, ctx)
			item := NewItem(req, testTTL, h.mockClock.Now())
			h.addQueue(types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority})

			// --- ACT ---
			h.Start()
			defer h.Stop()
			h.processor.Enqueue(item)
			h.Go()
			cancel()                        // Cancel the context *after* the item is enqueued.
			time.Sleep(testCleanupTick * 3) // Allow the cleanup goroutine time to run.

			// --- ASSERT ---
			outcome, err := h.waitForFinalization(item)
			assert.Equal(t, types.QueueOutcomeEvictedContextCancelled, outcome,
				"The outcome should be EvictedContextCancelled")
			require.Error(t, err, "A context cancellation eviction should produce an error")
			assert.ErrorIs(t, err, types.ErrContextCancelled, "The error should be of type ErrContextCancelled")
		})

		t.Run("should evict a queued item on shutdown", func(t *testing.T) {
			t.Parallel()
			// --- ARRANGE ---
			h := newTestHarness(t, testCleanupTick)
			item := h.newTestItem("req-shutdown-evict", testFlow.ID, testTTL)
			mockQueue := h.addQueue(types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority})
			require.NoError(t, mockQueue.Add(item), "Adding item to mock queue should not fail")

			// Prevent dispatch to ensure we test shutdown eviction, not a successful dispatch.
			h.interFlowPolicySelectQueue = func(band framework.PriorityBandAccessor) (framework.FlowQueueAccessor, error) {
				return nil, nil
			}

			// --- ACT ---
			h.Start()
			h.Go()
			h.Stop() // Stop immediately to trigger eviction.

			// --- ASSERT ---
			outcome, err := h.waitForFinalization(item)
			assert.Equal(t, types.QueueOutcomeEvictedOther, outcome, "The outcome should be EvictedOther")
			require.Error(t, err, "An eviction on shutdown should produce an error")
			assert.ErrorIs(t, err, types.ErrFlowControllerShutdown, "The error should be of type ErrFlowControllerShutdown")
		})

		t.Run("should handle concurrent enqueues and dispatch all items", func(t *testing.T) {
			t.Parallel()
			// --- ARRANGE ---
			h := newTestHarness(t, testCleanupTick)
			const numConcurrentItems = 20
			q := h.addQueue(types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority})
			itemsToTest := make([]*flowItem, 0, numConcurrentItems)
			for i := 0; i < numConcurrentItems; i++ {
				item := h.newTestItem(fmt.Sprintf("req-concurrent-%d", i), testFlow.ID, testTTL)
				itemsToTest = append(itemsToTest, item)
			}

			// --- ACT ---
			h.Start()
			defer h.Stop()
			var wg sync.WaitGroup
			for _, item := range itemsToTest {
				wg.Add(1)
				go func(fi *flowItem) {
					defer wg.Done()
					h.processor.Enqueue(fi)
				}(item)
			}
			h.Go()
			wg.Wait() // Wait for all enqueues to finish.

			// --- ASSERT ---
			for _, item := range itemsToTest {
				outcome, err := h.waitForFinalization(item)
				assert.Equal(t, types.QueueOutcomeDispatched, outcome,
					"Item %q should have been dispatched", item.OriginalRequest().ID())
				assert.NoError(t, err,
					"A successful dispatch of item %q should not produce an error", item.OriginalRequest().ID())
			}
			assert.Equal(t, 0, q.Len(), "The mock queue should be empty at the end of the test")
		})

		t.Run("should guarantee exactly-once finalization during dispatch vs. expiry race", func(t *testing.T) {
			t.Parallel()

			// --- ARRANGE ---
			h := newTestHarness(t, 1*time.Hour) // Disable background cleanup to isolate the race.
			item := h.newTestItem("req-race", testFlow.ID, testShortTTL)
			q := h.addQueue(types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority})

			// Use channels to pause the dispatch cycle right before it would remove the item.
			policyCanProceed := make(chan struct{})
			itemIsBeingDispatched := make(chan struct{})

			require.NoError(t, q.Add(item)) // Add the item directly to the queue.

			// Override the queue's `RemoveFunc` to pause the dispatch goroutine at a critical moment.
			q.RemoveFunc = func(h types.QueueItemHandle) (types.QueueItemAccessor, error) {
				close(itemIsBeingDispatched) // 1. Signal that dispatch is happening.
				<-policyCanProceed           // 2. Wait for the test to tell us to continue.
				// 4. After we unblock, the item will have already been finalized by the cleanup logic, so we simulate the
				//    real-world outcome of a failed remove.
				return nil, fmt.Errorf("item with handle %v not found", h)
			}

			// --- ACT ---
			h.Start()
			defer h.Stop()
			h.Go()

			// Wait for the dispatch cycle to select our item and pause inside our mock `RemoveFunc`.
			<-itemIsBeingDispatched

			// 3. The dispatch goroutine is now paused. We can now safely win the "race" by running cleanup logic.
			h.mockClock.Advance(testShortTTL * 2)
			h.processor.cleanupExpired(h.mockClock.Now()) // This will remove and finalize the item.

			// 5. Un-pause the dispatch goroutine. It will now fail to remove the item and the `dispatchCycle` will
			//    correctly conclude without finalizing the item a second time.
			close(policyCanProceed)

			// --- ASSERT ---
			// The item's final state should be from the cleanup logic (EvictedTTL), not the dispatch logic.
			outcome, err := h.waitForFinalization(item)
			assert.Equal(t, types.QueueOutcomeEvictedTTL, outcome, "The outcome should be EvictedTTL from the cleanup routine")
			require.Error(t, err, "A TTL eviction should produce an error")
			assert.ErrorIs(t, err, types.ErrTTLExpired, "The error should be of type ErrTTLExpired")
		})

		t.Run("should shut down cleanly on context cancellation", func(t *testing.T) {
			t.Parallel()
			// --- ARRANGE ---
			h := newTestHarness(t, testCleanupTick)
			stopped := make(chan struct{})

			// --- ACT ---
			h.Start()
			h.Go()

			// Use a separate goroutine to wait for the processor to fully stop.
			go func() {
				h.Stop() // This cancels the context and waits on the WaitGroup.
				close(stopped)
			}()

			// --- ASSERT ---
			select {
			case <-stopped:
				// Success: The Stop() call completed without a deadlock.
			case <-time.After(testWaitTimeout):
				t.Fatal("Test timed out waiting for processor to stop")
			}
		})

		t.Run("should not panic on nil item from enqueue channel", func(t *testing.T) {
			t.Parallel()
			// --- ARRANGE ---
			h := newTestHarness(t, testCleanupTick)
			// This test is primarily checking that the processor doesn't panic or error on a nil input.

			// --- ACT ---
			h.Start()
			defer h.Stop()
			h.Go()
			h.processor.Enqueue(nil)

			// --- ASSERT ---
			// Allow a moment for the processor to potentially process the nil item.
			// A successful test is one that completes without panicking.
			time.Sleep(50 * time.Millisecond)
		})
	})

	t.Run("Unit", func(t *testing.T) {
		t.Parallel()

		t.Run("enqueue", func(t *testing.T) {
			t.Parallel()
			testErr := errors.New("something went wrong")

			testCases := []struct {
				name         string
				setupHarness func(h *testHarness)
				item         *flowItem
				assert       func(t *testing.T, h *testHarness, item *flowItem)
			}{
				{
					name: "should reject item on registry queue lookup failure",
					setupHarness: func(h *testHarness) {
						h.ActiveManagedQueueFunc = func(string) (contracts.ManagedQueue, error) { return nil, testErr }
					},
					assert: func(t *testing.T, h *testHarness, item *flowItem) {
						outcome, err := item.FinalState()
						assert.Equal(t, types.QueueOutcomeRejectedOther, outcome, "Outcome should be RejectedOther")
						require.Error(t, err, "An error should be returned")
						assert.ErrorIs(t, err, testErr, "The underlying error should be preserved")
					},
				},
				{
					name: "should reject item on registry priority band lookup failure",
					setupHarness: func(h *testHarness) {
						h.addQueue(types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority})
						h.PriorityBandAccessorFunc = func(uint) (framework.PriorityBandAccessor, error) { return nil, testErr }
					},
					assert: func(t *testing.T, h *testHarness, item *flowItem) {
						outcome, err := item.FinalState()
						assert.Equal(t, types.QueueOutcomeRejectedOther, outcome, "Outcome should be RejectedOther")
						require.Error(t, err, "An error should be returned")
						assert.ErrorIs(t, err, testErr, "The underlying error should be preserved")
					},
				},
				{
					name: "should reject item on queue add failure",
					setupHarness: func(h *testHarness) {
						mockQueue := h.addQueue(types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority})
						mockQueue.AddFunc = func(types.QueueItemAccessor) error { return testErr }
					},
					assert: func(t *testing.T, h *testHarness, item *flowItem) {
						outcome, err := item.FinalState()
						assert.Equal(t, types.QueueOutcomeRejectedOther, outcome, "Outcome should be RejectedOther")
						require.Error(t, err, "An error should be returned")
						assert.ErrorIs(t, err, testErr, "The underlying error should be preserved")
					},
				},
				{
					name: "should ignore an already-finalized item",
					setupHarness: func(h *testHarness) {
						mockQueue := h.addQueue(types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority})
						var addCallCount int
						mockQueue.AddFunc = func(item types.QueueItemAccessor) error {
							addCallCount++
							return nil
						}
						// Use Cleanup to assert after the test logic has run.
						t.Cleanup(func() {
							assert.Equal(t, 0, addCallCount, "Queue.Add should not have been called for a finalized item")
						})
					},
					item: func() *flowItem {
						// Create a pre-finalized item.
						item := newTestHarness(t, 0).newTestItem("req-finalized", testFlow.ID, testTTL)
						item.finalize(types.QueueOutcomeDispatched, nil)
						return item
					}(),
					assert: func(t *testing.T, h *testHarness, item *flowItem) {
						// The item was already finalized, so its state should not change.
						outcome, err := item.FinalState()
						assert.Equal(t, types.QueueOutcomeDispatched, outcome, "Outcome should remain unchanged")
						assert.NoError(t, err, "Error should remain unchanged")
					},
				},
			}

			for _, tc := range testCases {
				t.Run(tc.name, func(t *testing.T) {
					t.Parallel()
					h := newTestHarness(t, testCleanupTick)
					tc.setupHarness(h)
					item := tc.item
					if item == nil {
						item = h.newTestItem("req-enqueue-test", testFlow.ID, testTTL)
					}
					h.processor.enqueue(item)
					tc.assert(t, h, item)
				})
			}
		})

		t.Run("hasCapacity", func(t *testing.T) {
			t.Parallel()
			testCases := []struct {
				name         string
				itemByteSize uint64
				stats        contracts.ShardStats
				expectHasCap bool
			}{
				{
					name:         "should allow zero-size item even if full",
					itemByteSize: 0,
					stats:        contracts.ShardStats{TotalByteSize: 100, TotalCapacityBytes: 100},
					expectHasCap: true,
				},
				{
					name:         "should deny item if shard capacity exceeded",
					itemByteSize: 1,
					stats:        contracts.ShardStats{TotalByteSize: 100, TotalCapacityBytes: 100},
					expectHasCap: false,
				},
				{
					name:         "should deny item if band capacity exceeded",
					itemByteSize: 1,
					stats: contracts.ShardStats{
						TotalCapacityBytes: 200, TotalByteSize: 100,
						PerPriorityBandStats: map[uint]contracts.PriorityBandStats{
							testFlow.Priority: {ByteSize: 50, CapacityBytes: 50},
						},
					},
					expectHasCap: false,
				},
				{
					name:         "should deny item if band stats are missing",
					itemByteSize: 1,
					stats: contracts.ShardStats{
						TotalCapacityBytes: 200, TotalByteSize: 100,
						PerPriorityBandStats: map[uint]contracts.PriorityBandStats{}, // Missing stats for priority 10
					},
					expectHasCap: false,
				},
				{
					name:         "should allow item if both shard and band have capacity",
					itemByteSize: 10,
					stats: contracts.ShardStats{
						TotalCapacityBytes: 200, TotalByteSize: 100,
						PerPriorityBandStats: map[uint]contracts.PriorityBandStats{
							testFlow.Priority: {ByteSize: 50, CapacityBytes: 100},
						},
					},
					expectHasCap: true,
				},
			}

			for _, tc := range testCases {
				t.Run(tc.name, func(t *testing.T) {
					t.Parallel()
					h := newTestHarness(t, testCleanupTick)
					h.StatsFunc = func() contracts.ShardStats { return tc.stats }
					hasCap := h.processor.hasCapacity(testFlow.Priority, tc.itemByteSize)
					assert.Equal(t, tc.expectHasCap, hasCap, "Capacity check result should match expected value")
				})
			}
		})

		t.Run("dispatchCycle", func(t *testing.T) {
			t.Parallel()

			t.Run("should handle various policy and registry scenarios", func(t *testing.T) {
				t.Parallel()
				policyErr := errors.New("policy failure")
				registryErr := errors.New("registry error")
				specA := types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority}

				testCases := []struct {
					name              string
					setupHarness      func(h *testHarness)
					expectDidDispatch bool
				}{
					{
						name: "should do nothing if no items are queued",
						setupHarness: func(h *testHarness) {
							h.addQueue(specA) // Add a queue, but no items.
						},
						expectDidDispatch: false,
					},
					{
						name: "should stop dispatching when filter signals pause",
						setupHarness: func(h *testHarness) {
							// Add an item that *could* be dispatched to prove the pause is effective.
							q := h.addQueue(types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority})
							require.NoError(t, q.Add(h.newTestItem("item", testFlow.ID, testTTL)))
							h.processor.dispatchFilter = func(
								_ context.Context,
								_ framework.PriorityBandAccessor,
								_ logr.Logger,
							) (map[string]struct{}, bool) {
								return nil, true // Signal pause.
							}
						},
						expectDidDispatch: false,
					},
					{
						name: "should skip band on priority band accessor error",
						setupHarness: func(h *testHarness) {
							h.PriorityBandAccessorFunc = func(uint) (framework.PriorityBandAccessor, error) {
								return nil, registryErr
							}
						},
						expectDidDispatch: false,
					},
					{
						name: "should skip band on inter-flow policy error",
						setupHarness: func(h *testHarness) {
							h.addQueue(specA)
							h.interFlowPolicySelectQueue = func(
								_ framework.PriorityBandAccessor,
							) (framework.FlowQueueAccessor, error) {
								return nil, policyErr
							}
						},
						expectDidDispatch: false,
					},
					{
						name: "should skip band if inter-flow policy returns no queue",
						setupHarness: func(h *testHarness) {
							q := h.addQueue(specA)
							require.NoError(t, q.Add(h.newTestItem("item", testFlow.ID, testTTL)))
							h.interFlowPolicySelectQueue = func(
								_ framework.PriorityBandAccessor,
							) (framework.FlowQueueAccessor, error) {
								return nil, nil // Simulate band being empty or policy choosing to pause.
							}
						},
						expectDidDispatch: false,
					},
					{
						name: "should skip band on intra-flow policy error",
						setupHarness: func(h *testHarness) {
							q := h.addQueue(specA)
							require.NoError(t, q.Add(h.newTestItem("item", testFlow.ID, testTTL)))
							h.interFlowPolicySelectQueue = func(
								_ framework.PriorityBandAccessor,
							) (framework.FlowQueueAccessor, error) {
								return q.FlowQueueAccessor(), nil
							}
							h.intraFlowPolicySelectItem = func(_ framework.FlowQueueAccessor) (types.QueueItemAccessor, error) {
								return nil, policyErr
							}
						},
						expectDidDispatch: false,
					},
					{
						name: "should skip band if intra-flow policy returns no item",
						setupHarness: func(h *testHarness) {
							q := h.addQueue(specA)
							require.NoError(t, q.Add(h.newTestItem("item", testFlow.ID, testTTL)))
							h.interFlowPolicySelectQueue = func(
								_ framework.PriorityBandAccessor,
							) (framework.FlowQueueAccessor, error) {
								return q.FlowQueueAccessor(), nil
							}
							h.intraFlowPolicySelectItem = func(_ framework.FlowQueueAccessor) (types.QueueItemAccessor, error) {
								return nil, nil // Simulate queue being empty or policy choosing to pause.
							}
						},
						expectDidDispatch: false,
					},
					{
						name: "should continue to lower priority band on inter-flow policy error",
						setupHarness: func(h *testHarness) {
							// Create a failing high-priority queue and a working low-priority queue.
							specHigh := types.FlowSpecification{ID: "flow-high", Priority: testFlow.Priority}
							specLow := types.FlowSpecification{ID: "flow-low", Priority: 20}
							h.addQueue(specHigh)
							qLow := h.addQueue(specLow)

							itemLow := h.newTestItem("item-low", specLow.ID, testTTL)
							require.NoError(t, qLow.Add(itemLow))

							h.interFlowPolicySelectQueue = func(
								band framework.PriorityBandAccessor,
							) (framework.FlowQueueAccessor, error) {
								if band.Priority() == testFlow.Priority {
									return nil, errors.New("policy failure") // Fail high-priority.
								}
								// Succeed for low-priority.
								q, _ := h.managedQueue(specLow.ID, specLow.Priority)
								return q.FlowQueueAccessor(), nil
							}
						},
						expectDidDispatch: true,
					},
				}

				for _, tc := range testCases {
					t.Run(tc.name, func(t *testing.T) {
						t.Parallel()
						h := newTestHarness(t, testCleanupTick)
						tc.setupHarness(h)
						dispatched := h.processor.dispatchCycle(context.Background())
						assert.Equal(t, tc.expectDidDispatch, dispatched, "Dispatch result should match expected value")
					})
				}
			})

			t.Run("should use filtered view of queues when filter is active", func(t *testing.T) {
				t.Parallel()
				// --- ARRANGE ---
				h := newTestHarness(t, testCleanupTick)
				specA := types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority}
				specB := types.FlowSpecification{ID: "flow-b", Priority: testFlow.Priority}
				h.addQueue(specA)
				qB := h.addQueue(specB)
				itemB := h.newTestItem("item-b", specB.ID, testTTL)
				require.NoError(t, qB.Add(itemB))

				// This filter only allows flow-b.
				h.processor.dispatchFilter = func(
					_ context.Context,
					_ framework.PriorityBandAccessor,
					_ logr.Logger,
				) (map[string]struct{}, bool) {
					return map[string]struct{}{specB.ID: {}}, false
				}

				// This policy will be given the filtered view, so it should only see flow-b.
				h.interFlowPolicySelectQueue = func(band framework.PriorityBandAccessor) (framework.FlowQueueAccessor, error) {
					var flowIDs []string
					band.IterateQueues(func(fqa framework.FlowQueueAccessor) bool {
						flowIDs = append(flowIDs, fqa.FlowSpec().ID)
						return true
					})
					// This is the core assertion of the test.
					require.ElementsMatch(t, []string{specB.ID}, flowIDs, "Policy should only see the filtered flow")

					// Select flow-b to prove the chain works.
					q, _ := h.managedQueue(specB.ID, specB.Priority)
					return q.FlowQueueAccessor(), nil
				}

				// --- ACT ---
				dispatched := h.processor.dispatchCycle(context.Background())

				// --- ASSERT ---
				assert.True(t, dispatched, "An item should have been dispatched from the filtered flow")
				outcome, err := itemB.FinalState()
				assert.Equal(t, types.QueueOutcomeDispatched, outcome, "The dispatched item's outcome should be correct")
				assert.NoError(t, err, "The dispatched item should not have an error")
			})

			t.Run("should guarantee strict priority by starving lower priority items", func(t *testing.T) {
				t.Parallel()
				// --- ARRANGE ---
				h := newTestHarness(t, testCleanupTick)
				specHigh := types.FlowSpecification{ID: "flow-high", Priority: 10}
				specLow := types.FlowSpecification{ID: "flow-low", Priority: 20}
				qHigh := h.addQueue(specHigh)
				qLow := h.addQueue(specLow)

				const numItems = 3
				highPrioItems := make([]*flowItem, numItems)
				lowPrioItems := make([]*flowItem, numItems)
				for i := range numItems {
					// Add high priority items.
					itemH := h.newTestItem(fmt.Sprintf("req-high-%d", i), specHigh.ID, testTTL)
					require.NoError(t, qHigh.Add(itemH))
					highPrioItems[i] = itemH

					// Add low priority items.
					itemL := h.newTestItem(fmt.Sprintf("req-low-%d", i), specLow.ID, testTTL)
					require.NoError(t, qLow.Add(itemL))
					lowPrioItems[i] = itemL
				}

				// --- ACT & ASSERT ---
				// First, dispatch all high-priority items.
				for i := range numItems {
					dispatched := h.processor.dispatchCycle(context.Background())
					require.True(t, dispatched, "Expected a high-priority dispatch on cycle %d", i+1)
				}

				// Verify all high-priority items are gone and low-priority items remain.
				for _, item := range highPrioItems {
					outcome, err := item.FinalState()
					assert.Equal(t, types.QueueOutcomeDispatched, outcome, "High-priority item should be dispatched")
					assert.NoError(t, err, "Dispatched high-priority item should not have an error")
				}
				assert.Equal(t, numItems, qLow.Len(), "Low-priority queue should still be full")

				// Next, dispatch all low-priority items.
				for i := range numItems {
					dispatched := h.processor.dispatchCycle(context.Background())
					require.True(t, dispatched, "Expected a low-priority dispatch on cycle %d", i+1)
				}
				assert.Equal(t, 0, qLow.Len(), "Low-priority queue should be empty")
			})
		})

		t.Run("dispatchItem", func(t *testing.T) {
			t.Parallel()

			t.Run("should fail on registry errors", func(t *testing.T) {
				t.Parallel()
				registryErr := errors.New("registry error")

				testCases := []struct {
					name        string
					setupMocks  func(h *testHarness)
					expectedErr error
				}{
					{
						name: "on ManagedQueue lookup failure",
						setupMocks: func(h *testHarness) {
							h.ManagedQueueFunc = func(string, uint) (contracts.ManagedQueue, error) { return nil, registryErr }
						},
						expectedErr: registryErr,
					},
					{
						name: "on queue remove failure",
						setupMocks: func(h *testHarness) {
							h.ManagedQueueFunc = func(string, uint) (contracts.ManagedQueue, error) {
								return &mocks.MockManagedQueue{
									RemoveFunc: func(types.QueueItemHandle) (types.QueueItemAccessor, error) {
										return nil, registryErr
									},
								}, nil
							}
						},
						expectedErr: registryErr,
					},
				}

				for _, tc := range testCases {
					t.Run(tc.name, func(t *testing.T) {
						t.Parallel()
						h := newTestHarness(t, testCleanupTick)
						tc.setupMocks(h)
						item := h.newTestItem("req-dispatch-fail", testFlow.ID, testTTL)
						err := h.processor.dispatchItem(item, testFlow.Priority, h.logger)
						require.Error(t, err, "dispatchItem should return an error")
						assert.ErrorIs(t, err, tc.expectedErr, "The underlying registry error should be preserved")
					})
				}
			})

			t.Run("should evict item that expires at moment of dispatch", func(t *testing.T) {
				t.Parallel()
				// --- ARRANGE ---
				h := newTestHarness(t, testCleanupTick)
				item := h.newTestItem("req-expired-dispatch", testFlow.ID, testShortTTL)

				h.ManagedQueueFunc = func(string, uint) (contracts.ManagedQueue, error) {
					return &mocks.MockManagedQueue{
						RemoveFunc: func(types.QueueItemHandle) (types.QueueItemAccessor, error) {
							return item, nil
						},
					}, nil
				}

				// --- ACT ---
				h.mockClock.Advance(testShortTTL * 2) // Make the item expire.
				err := h.processor.dispatchItem(item, testFlow.Priority, h.logger)

				// --- ASSERT ---
				// First, check the error returned by `dispatchItem`.
				require.Error(t, err, "dispatchItem should return an error for an expired item")
				assert.ErrorIs(t, err, types.ErrTTLExpired, "The error should be of type ErrTTLExpired")

				// Second, check the final state of the item itself.
				outcome, finalErr := item.FinalState()
				assert.Equal(t, types.QueueOutcomeEvictedTTL, outcome, "The item's final outcome should be EvictedTTL")
				require.Error(t, finalErr, "The item's final state should contain an error")
				assert.ErrorIs(t, finalErr, types.ErrTTLExpired, "The item's final error should be of type ErrTTLExpired")
			})

			t.Run("should panic if queue returns item of wrong type", func(t *testing.T) {
				t.Parallel()
				// --- ARRANGE ---
				h := newTestHarness(t, testCleanupTick)
				badItem := &typesmocks.MockQueueItemAccessor{
					OriginalRequestV: typesmocks.NewMockFlowControlRequest(0, "bad-item", "", context.Background()),
				}

				h.ManagedQueueFunc = func(string, uint) (contracts.ManagedQueue, error) {
					return &mocks.MockManagedQueue{
						RemoveFunc: func(types.QueueItemHandle) (types.QueueItemAccessor, error) {
							return badItem, nil
						},
					}, nil
				}

				itemToDispatch := h.newTestItem("req-dispatch-panic", testFlow.ID, testTTL)
				expectedPanicMsg := fmt.Sprintf("%s: internal error: item %q of type %T is not a *flowItem",
					errIntraFlow, "bad-item", badItem)

				// --- ACT & ASSERT ---
				assert.PanicsWithError(t, expectedPanicMsg, func() {
					_ = h.processor.dispatchItem(itemToDispatch, testFlow.Priority, h.logger)
				}, "A type mismatch from a queue should cause a panic")
			})
		})

		t.Run("cleanup and utility methods", func(t *testing.T) {
			t.Parallel()

			t.Run("should remove and finalize expired items", func(t *testing.T) {
				t.Parallel()
				// --- ARRANGE ---
				h := newTestHarness(t, testCleanupTick)
				spec := types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority}
				// Create an item that is already expired relative to the cleanup time.
				item := h.newTestItem("req-expired", testFlow.ID, 1*time.Millisecond)
				q := h.addQueue(spec)
				require.NoError(t, q.Add(item))
				cleanupTime := h.mockClock.Now().Add(10 * time.Millisecond)

				// --- ACT ---
				h.processor.cleanupExpired(cleanupTime)

				// --- ASSERT ---
				outcome, err := item.FinalState()
				assert.Equal(t, types.QueueOutcomeEvictedTTL, outcome, "Item outcome should be EvictedTTL")
				require.Error(t, err, "Item should have an error")
				assert.ErrorIs(t, err, types.ErrTTLExpired, "Item error should be ErrTTLExpired")
			})

			t.Run("should evict all items on shutdown", func(t *testing.T) {
				t.Parallel()
				// --- ARRANGE ---
				h := newTestHarness(t, testCleanupTick)
				spec := types.FlowSpecification{ID: testFlow.ID, Priority: testFlow.Priority}
				item := h.newTestItem("req-pending", testFlow.ID, testTTL)
				q := h.addQueue(spec)
				require.NoError(t, q.Add(item))

				// --- ACT ---
				h.processor.evictAll()

				// --- ASSERT ---
				outcome, err := item.FinalState()
				assert.Equal(t, types.QueueOutcomeEvictedOther, outcome, "Item outcome should be EvictedOther")
				require.Error(t, err, "Item should have an error")
				assert.ErrorIs(t, err, types.ErrFlowControllerShutdown, "Item error should be ErrFlowControllerShutdown")
			})

			t.Run("should handle registry errors gracefully during concurrent processing", func(t *testing.T) {
				t.Parallel()
				// --- ARRANGE ---
				h := newTestHarness(t, testCleanupTick)
				h.AllOrderedPriorityLevelsFunc = func() []uint { return []uint{testFlow.Priority} }
				h.PriorityBandAccessorFunc = func(p uint) (framework.PriorityBandAccessor, error) {
					return nil, errors.New("registry error")
				}

				// --- ACT & ASSERT ---
				// The test passes if this call completes without panicking.
				assert.NotPanics(t, func() {
					h.processor.processAllQueuesConcurrently("test", func(mq contracts.ManagedQueue, logger logr.Logger) {})
				}, "processAllQueuesConcurrently should not panic on registry errors")
			})

			t.Run("should handle items of an unexpected type gracefully during finalization", func(t *testing.T) {
				t.Parallel()
				// --- ARRANGE ---
				h := newTestHarness(t, testCleanupTick)
				item := &typesmocks.MockQueueItemAccessor{
					OriginalRequestV: typesmocks.NewMockFlowControlRequest(0, "bad-item", testFlow.ID, context.Background()),
				}
				items := []types.QueueItemAccessor{item}

				// --- ACT & ASSERT ---
				// The test passes if this call completes without panicking.
				assert.NotPanics(t, func() {
					getOutcome := func(types.QueueItemAccessor) (types.QueueOutcome, error) {
						return types.QueueOutcomeEvictedOther, nil
					}
					h.processor.finalizeItems(items, h.logger, getOutcome)
				}, "finalizeItems should not panic on unexpected item types")
			})

			t.Run("should process all queues with a worker pool", func(t *testing.T) {
				t.Parallel()
				// --- ARRANGE ---
				h := newTestHarness(t, testCleanupTick)

				// Create more queues than the fixed number of cleanup workers to ensure the pooling logic is exercised.
				const numQueues = maxCleanupWorkers + 5
				var processedCount atomic.Int32

				for i := range numQueues {
					spec := types.FlowSpecification{
						ID:       fmt.Sprintf("flow-%d", i),
						Priority: testFlow.Priority,
					}
					h.addQueue(spec)
				}

				processFn := func(mq contracts.ManagedQueue, logger logr.Logger) {
					processedCount.Add(1)
				}

				// --- ACT ---
				h.processor.processAllQueuesConcurrently("test-worker-pool", processFn)

				// --- ASSERT ---
				assert.Equal(t, int32(numQueues), processedCount.Load(),
					"The number of processed queues should match the number created")
			})
		})
	})
}

func TestCheckItemExpiry(t *testing.T) {
	t.Parallel()

	// --- ARRANGE ---
	now := time.Now()
	ctxCancelled, cancel := context.WithCancel(context.Background())
	cancel() // Cancel the context immediately.

	testCases := []struct {
		name          string
		item          types.QueueItemAccessor
		now           time.Time
		expectExpired bool
		expectOutcome types.QueueOutcome
		expectErr     error
	}{
		{
			name: "should not be expired if TTL is not reached and context is active",
			item: NewItem(
				typesmocks.NewMockFlowControlRequest(100, "req-not-expired", "", context.Background()),
				testTTL,
				now),
			now:           now.Add(30 * time.Second),
			expectExpired: false,
			expectOutcome: types.QueueOutcomeNotYetFinalized,
			expectErr:     nil,
		},
		{
			name: "should not be expired if TTL is disabled (0)",
			item: NewItem(
				typesmocks.NewMockFlowControlRequest(100, "req-not-expired-no-ttl", "", context.Background()),
				0,
				now),
			now:           now.Add(30 * time.Second),
			expectExpired: false,
			expectOutcome: types.QueueOutcomeNotYetFinalized,
			expectErr:     nil,
		},
		{
			name: "should be expired if TTL is exceeded",
			item: NewItem(
				typesmocks.NewMockFlowControlRequest(100, "req-ttl-expired", "", context.Background()),
				time.Second,
				now),
			now:           now.Add(2 * time.Second),
			expectExpired: true,
			expectOutcome: types.QueueOutcomeEvictedTTL,
			expectErr:     types.ErrTTLExpired,
		},
		{
			name: "should be expired if context is cancelled",
			item: NewItem(
				typesmocks.NewMockFlowControlRequest(100, "req-ctx-cancelled", "", ctxCancelled),
				testTTL,
				now),
			now:           now,
			expectExpired: true,
			expectOutcome: types.QueueOutcomeEvictedContextCancelled,
			expectErr:     types.ErrContextCancelled,
		},
		{
			name: "should be expired if already finalized",
			item: func() types.QueueItemAccessor {
				i := NewItem(typesmocks.NewMockFlowControlRequest(100, "req-finalized", "", context.Background()), testTTL, now)
				i.finalize(types.QueueOutcomeDispatched, nil)
				return i
			}(),
			now:           now,
			expectExpired: true,
			expectOutcome: types.QueueOutcomeDispatched,
			expectErr:     nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			// --- ACT ---
			isExpired, outcome, err := checkItemExpiry(tc.item, tc.now)

			// --- ASSERT ---
			assert.Equal(t, tc.expectExpired, isExpired, "Expired status should match expected value")
			assert.Equal(t, tc.expectOutcome, outcome, "Outcome should match expected value")

			if tc.expectErr != nil {
				require.Error(t, err, "An error was expected")
				// Use ErrorIs for sentinel errors, ErrorContains for general messages.
				if errors.Is(tc.expectErr, types.ErrTTLExpired) || errors.Is(tc.expectErr, types.ErrContextCancelled) {
					assert.ErrorIs(t, err, tc.expectErr, "The specific error type should be correct")
				} else {
					assert.ErrorContains(t, err, tc.expectErr.Error(), "The error message should contain the expected text")
				}
			} else {
				assert.NoError(t, err, "No error was expected")
			}
		})
	}

	t.Run("should panic on item of an unexpected type", func(t *testing.T) {
		t.Parallel()
		// --- ARRANGE ---
		badItem := &typesmocks.MockQueueItemAccessor{
			OriginalRequestV: typesmocks.NewMockFlowControlRequest(0, "item-bad-type", "", context.Background()),
		}

		expectedPanicMsg := fmt.Sprintf("internal error: item %q of type %T is not a *flowItem",
			badItem.OriginalRequestV.ID(), badItem)

		// --- ACT & ASSERT ---
		assert.PanicsWithError(t, expectedPanicMsg, func() {
			_, _, _ = checkItemExpiry(badItem, time.Now())
		}, "A type mismatch from a queue should cause a panic")
	})
}
