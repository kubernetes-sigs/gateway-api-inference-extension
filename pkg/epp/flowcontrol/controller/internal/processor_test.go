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
// In summary, this is a prerequisite for reliably testing a concurrent engine, not just a simple data
// structure.
//

package internal

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	testclock "k8s.io/utils/clock/testing"
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

var testFlow = types.FlowKey{ID: "flow-a", Priority: 10}

// TestMain sets up the logger for all tests in the package.
func TestMain(m *testing.M) {
	log.SetLogger(zap.New(zap.WriteTo(os.Stderr), zap.UseDevMode(true)))
	os.Exit(m.Run())
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
	clock     *testclock.FakeClock
	logger    logr.Logger

	// --- Centralized Mock State ---
	// The harness's mutex protects the single source of truth for all mock state.
	mu            sync.Mutex
	queues        map[types.FlowKey]*mocks.MockManagedQueue
	priorityFlows map[int][]types.FlowKey // Key: `priority`

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
		clock:             testclock.NewFakeClock(time.Now()),
		logger:            logr.Discard(),
		startSignal:       make(chan struct{}),
		queues:            make(map[types.FlowKey]*mocks.MockManagedQueue),
		priorityFlows:     make(map[int][]types.FlowKey),
	}

	// Wire up the harness to provide the mock implementations for the shard's dependencies.
	h.ManagedQueueFunc = h.managedQueue
	h.AllOrderedPriorityLevelsFunc = h.allOrderedPriorityLevels
	h.PriorityBandAccessorFunc = h.priorityBandAccessor
	h.InterFlowDispatchPolicyFunc = h.interFlowDispatchPolicy
	h.IntraFlowDispatchPolicyFunc = h.intraFlowDispatchPolicy

	// Provide a default stats implementation that is effectively infinite.
	h.StatsFunc = func() contracts.ShardStats {
		return contracts.ShardStats{
			TotalCapacityBytes: 1e9,
			PerPriorityBandStats: map[int]contracts.PriorityBandStats{
				testFlow.Priority: {CapacityBytes: 1e9},
			},
		}
	}

	// Use a default pass-through filter.
	filter := func(
		ctx context.Context,
		band framework.PriorityBandAccessor,
		logger logr.Logger,
	) (framework.PriorityBandAccessor, bool) {
		return nil, false
	}
	h.processor = NewShardProcessor(h, filter, h.clock, expiryCleanupInterval, 100, h.logger)
	require.NotNil(t, h.processor, "NewShardProcessor should not return nil")

	t.Cleanup(func() { h.Stop() })

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
func (h *testHarness) waitForFinalization(item *FlowItem) (types.QueueOutcome, error) {
	h.t.Helper()
	select {
	case finalState := <-item.Done():
		return finalState.Outcome, finalState.Err
	case <-time.After(testWaitTimeout):
		h.t.Fatalf("Timed out waiting for item %q to be finalized", item.OriginalRequest().ID())
		return types.QueueOutcomeNotYetFinalized, nil
	}
}

// newTestItem creates a new FlowItem for testing purposes.
func (h *testHarness) newTestItem(id string, key types.FlowKey, ttl time.Duration) *FlowItem {
	h.t.Helper()
	ctx := log.IntoContext(context.Background(), h.logger)
	req := typesmocks.NewMockFlowControlRequest(100, id, key, ctx)
	return NewItem(req, ttl, h.clock.Now())
}

// addQueue centrally registers a new mock queue for a given flow, ensuring all harness components are aware of it.
func (h *testHarness) addQueue(key types.FlowKey) *mocks.MockManagedQueue {
	h.t.Helper()
	h.mu.Lock()
	defer h.mu.Unlock()
	mockQueue := &mocks.MockManagedQueue{FlowKeyV: key}
	h.queues[key] = mockQueue
	h.priorityFlows[key.Priority] = append(h.priorityFlows[key.Priority], key)
	return mockQueue
}

// --- Mock Interface Implementations ---

// managedQueue provides the mock implementation for the `RegistryShard` interface.
func (h *testHarness) managedQueue(key types.FlowKey) (contracts.ManagedQueue, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if q, ok := h.queues[key]; ok {
		return q, nil
	}
	return nil, fmt.Errorf("test setup error: no queue for %q", key)
}

// allOrderedPriorityLevels provides the mock implementation for the `RegistryShard` interface.
func (h *testHarness) allOrderedPriorityLevels() []int {
	h.mu.Lock()
	defer h.mu.Unlock()
	prios := make([]int, 0, len(h.priorityFlows))
	for p := range h.priorityFlows {
		prios = append(prios, p)
	}
	sort.Slice(prios, func(i, j int) bool {
		return prios[i] > prios[j]
	})

	return prios
}

// priorityBandAccessor provides the mock implementation for the `RegistryShard` interface. It acts as a factory for a
// fully-configured, stateless mock that is safe for concurrent use.
func (h *testHarness) priorityBandAccessor(p int) (framework.PriorityBandAccessor, error) {
	band := &frameworkmocks.MockPriorityBandAccessor{PriorityV: p}

	// Safely get a snapshot of the flow IDs under a lock.
	h.mu.Lock()
	flowKeysForPriority := h.priorityFlows[p]
	h.mu.Unlock()

	// Configure the mock's behavior with a closure that reads from the harness's centralized, thread-safe state.
	band.IterateQueuesFunc = func(cb func(fqa framework.FlowQueueAccessor) bool) {
		// This closure safely iterates over the snapshot of flow IDs.
		for _, key := range flowKeysForPriority {
			// Get the queue using the thread-safe `managedQueue` method.
			q, err := h.managedQueue(key)
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
func (h *testHarness) interFlowDispatchPolicy(p int) (framework.InterFlowDispatchPolicy, error) {
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
func (h *testHarness) intraFlowDispatchPolicy(types.FlowKey) (framework.IntraFlowDispatchPolicy, error) {
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

	// Integration tests use the processor's main `Run` loop to verify the complete end-to-end lifecycle of a request, from
	// `Enqueue` to its final outcome.
	t.Run("Integration", func(t *testing.T) {
		t.Parallel()

		t.Run("should dispatch item successfully", func(t *testing.T) {
			t.Parallel()
			// --- ARRANGE ---
			h := newTestHarness(t, testCleanupTick)
			item := h.newTestItem("req-dispatch-success", testFlow, testTTL)
			h.addQueue(testFlow)

			// --- ACT ---
			h.Start()
			require.NoError(t, h.processor.Submit(item), "precondition: Submit should not fail")
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
			item := h.newTestItem("req-capacity-reject", testFlow, testTTL)
			h.addQueue(testFlow)
			h.StatsFunc = func() contracts.ShardStats {
				return contracts.ShardStats{PerPriorityBandStats: map[int]contracts.PriorityBandStats{
					testFlow.Priority: {CapacityBytes: 50}, // 50 is less than item size of 100
				}}
			}

			// --- ACT ---
			h.Start()
			require.NoError(t, h.processor.Submit(item), "precondition: Submit should not fail")
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
			item := h.newTestItem("req-lookup-fail-reject", testFlow, testTTL)
			registryErr := errors.New("test registry lookup error")
			h.ManagedQueueFunc = func(types.FlowKey) (contracts.ManagedQueue, error) {
				return nil, registryErr
			}

			// --- ACT ---
			h.Start()
			defer h.Stop()
			require.NoError(t, h.processor.Submit(item), "precondition: Submit should not fail")
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
			item := h.newTestItem("req-shutdown-reject", testFlow, testTTL)
			h.addQueue(testFlow)

			// --- ACT ---
			h.Start()
			h.Go()
			h.Stop() // Stop the processor, then immediately try to enqueue.
			require.NoError(t, h.processor.Submit(item), "precondition: Submit should not fail, even on shutdown")

			// --- ASSERT ---
			outcome, err := h.waitForFinalization(item)
			assert.Equal(t, types.QueueOutcomeRejectedOther, outcome, "The outcome should be RejectedOther")
			require.Error(t, err, "An eviction on shutdown should produce an error")
			assert.ErrorIs(t, err, types.ErrFlowControllerNotRunning, "The error should be of type ErrFlowControllerNotRunning")
		})

		t.Run("should evict item on TTL expiry via background cleanup", func(t *testing.T) {
			t.Parallel()
			// --- ARRANGE ---
			h := newTestHarness(t, testCleanupTick)
			item := h.newTestItem("req-expired-evict", testFlow, testShortTTL)
			h.addQueue(testFlow)

			// --- ACT ---
			h.Start()
			require.NoError(t, h.processor.Submit(item), "precondition: Submit should not fail")
			h.Go()

			h.clock.Step(testShortTTL * 2) // Let time pass for the item to expire.
			// Manually invoke the cleanup logic to simulate a tick of the cleanup loop deterministically.
			h.processor.cleanupExpired(h.clock.Now())

			// --- ASSERT ---
			outcome, err := h.waitForFinalization(item)
			assert.Equal(t, types.QueueOutcomeEvictedTTL, outcome, "The final outcome should be EvictedTTL")
			require.Error(t, err, "A TTL eviction should produce an error")
			assert.ErrorIs(t, err, types.ErrTTLExpired, "The error should be of type ErrTTLExpired")
		})

		t.Run("should evict item on context cancellation", func(t *testing.T) {
			t.Parallel()
			// --- ARRANGE ---
			h := newTestHarness(t, testCleanupTick)
			ctx, cancel := context.WithCancel(context.Background())
			req := typesmocks.NewMockFlowControlRequest(100, "req-ctx-cancel", testFlow, ctx)
			item := NewItem(req, testTTL, h.clock.Now())
			h.addQueue(testFlow)

			// --- ACT ---
			h.Start()
			require.NoError(t, h.processor.Submit(item), "precondition: Submit should not fail")
			h.Go()
			cancel() // Cancel the context after the item is enqueued.
			// Manually invoke the cleanup logic to deterministically check for the cancelled context.
			h.processor.cleanupExpired(h.clock.Now())

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
			item := h.newTestItem("req-shutdown-evict", testFlow, testTTL)
			mockQueue := h.addQueue(testFlow)
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
			assert.ErrorIs(t, err, types.ErrFlowControllerNotRunning, "The error should be of type ErrFlowControllerNotRunning")
		})

		t.Run("should handle concurrent enqueues and dispatch all items", func(t *testing.T) {
			t.Parallel()
			// --- ARRANGE ---
			h := newTestHarness(t, testCleanupTick)
			const numConcurrentItems = 20
			q := h.addQueue(testFlow)
			itemsToTest := make([]*FlowItem, 0, numConcurrentItems)
			for i := 0; i < numConcurrentItems; i++ {
				item := h.newTestItem(fmt.Sprintf("req-concurrent-%d", i), testFlow, testTTL)
				itemsToTest = append(itemsToTest, item)
			}

			// --- ACT ---
			h.Start()
			defer h.Stop()
			var wg sync.WaitGroup
			for _, item := range itemsToTest {
				wg.Add(1)
				go func(fi *FlowItem) {
					defer wg.Done()
					require.NoError(t, h.processor.Submit(fi), "Submit should not fail")
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
			item := h.newTestItem("req-race", testFlow, testShortTTL)
			q := h.addQueue(testFlow)

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
			h.clock.Step(testShortTTL * 2)
			h.processor.cleanupExpired(h.clock.Now()) // This will remove and finalize the item.

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
			require.NoError(t, h.processor.Submit(nil), "Submit should not fail")

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
				item         *FlowItem
				assert       func(t *testing.T, h *testHarness, item *FlowItem)
			}{
				{
					name: "should reject item on registry queue lookup failure",
					setupHarness: func(h *testHarness) {
						h.ManagedQueueFunc = func(types.FlowKey) (contracts.ManagedQueue, error) { return nil, testErr }
					},
					assert: func(t *testing.T, h *testHarness, item *FlowItem) {
						assert.Equal(t, types.QueueOutcomeRejectedOther, item.finalState.Outcome, "Outcome should be RejectedOther")
						require.Error(t, item.finalState.Err, "An error should be returned")
						assert.ErrorIs(t, item.finalState.Err, testErr, "The underlying error should be preserved")
					},
				},
				{
					name: "should reject item on registry priority band lookup failure",
					setupHarness: func(h *testHarness) {
						h.addQueue(testFlow)
						h.PriorityBandAccessorFunc = func(int) (framework.PriorityBandAccessor, error) { return nil, testErr }
					},
					assert: func(t *testing.T, h *testHarness, item *FlowItem) {
						assert.Equal(t, types.QueueOutcomeRejectedOther, item.finalState.Outcome, "Outcome should be RejectedOther")
						require.Error(t, item.finalState.Err, "An error should be returned")
						assert.ErrorIs(t, item.finalState.Err, testErr, "The underlying error should be preserved")
					},
				},
				{
					name: "should reject item on queue add failure",
					setupHarness: func(h *testHarness) {
						mockQueue := h.addQueue(testFlow)
						mockQueue.AddFunc = func(types.QueueItemAccessor) error { return testErr }
					},
					assert: func(t *testing.T, h *testHarness, item *FlowItem) {
						assert.Equal(t, types.QueueOutcomeRejectedOther, item.finalState.Outcome, "Outcome should be RejectedOther")
						require.Error(t, item.finalState.Err, "An error should be returned")
						assert.ErrorIs(t, item.finalState.Err, testErr, "The underlying error should be preserved")
					},
				},
				{
					name: "should ignore an already-finalized item",
					setupHarness: func(h *testHarness) {
						mockQueue := h.addQueue(testFlow)
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
					item: func() *FlowItem {
						// Create a pre-finalized item.
						item := newTestHarness(t, 0).newTestItem("req-finalized", testFlow, testTTL)
						item.Finalize(types.QueueOutcomeDispatched, nil)
						return item
					}(),
					assert: func(t *testing.T, h *testHarness, item *FlowItem) {
						// The item was already finalized, so its state should not change.
						assert.Equal(t, types.QueueOutcomeDispatched, item.finalState.Outcome, "Outcome should remain unchanged")
						assert.NoError(t, item.finalState.Err, "Error should remain unchanged")
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
						item = h.newTestItem("req-enqueue-test", testFlow, testTTL)
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
						PerPriorityBandStats: map[int]contracts.PriorityBandStats{
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
						PerPriorityBandStats: map[int]contracts.PriorityBandStats{}, // Missing stats for priority 10
					},
					expectHasCap: false,
				},
				{
					name:         "should allow item if both shard and band have capacity",
					itemByteSize: 10,
					stats: contracts.ShardStats{
						TotalCapacityBytes: 200, TotalByteSize: 100,
						PerPriorityBandStats: map[int]contracts.PriorityBandStats{
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

				testCases := []struct {
					name              string
					setupHarness      func(h *testHarness)
					expectDidDispatch bool
				}{
					{
						name: "should do nothing if no items are queued",
						setupHarness: func(h *testHarness) {
							h.addQueue(testFlow) // Add a queue, but no items.
						},
						expectDidDispatch: false,
					},
					{
						name: "should stop dispatching when filter signals pause",
						setupHarness: func(h *testHarness) {
							// Add an item that *could* be dispatched to prove the pause is effective.
							q := h.addQueue(testFlow)
							require.NoError(t, q.Add(h.newTestItem("item", testFlow, testTTL)))
							h.processor.dispatchFilter = func(
								_ context.Context,
								_ framework.PriorityBandAccessor,
								_ logr.Logger,
							) (framework.PriorityBandAccessor, bool) {
								return nil, true // Signal pause.
							}
						},
						expectDidDispatch: false,
					},
					{
						name: "should skip band on priority band accessor error",
						setupHarness: func(h *testHarness) {
							h.PriorityBandAccessorFunc = func(int) (framework.PriorityBandAccessor, error) {
								return nil, registryErr
							}
						},
						expectDidDispatch: false,
					},
					{
						name: "should skip band on inter-flow policy error",
						setupHarness: func(h *testHarness) {
							h.addQueue(testFlow)
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
							q := h.addQueue(testFlow)
							require.NoError(t, q.Add(h.newTestItem("item", testFlow, testTTL)))
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
							q := h.addQueue(testFlow)
							require.NoError(t, q.Add(h.newTestItem("item", testFlow, testTTL)))
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
							q := h.addQueue(testFlow)
							require.NoError(t, q.Add(h.newTestItem("item", testFlow, testTTL)))
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
							keyHigh := types.FlowKey{ID: "flow-high", Priority: testFlow.Priority}
							keyLow := types.FlowKey{ID: "flow-low", Priority: 20}
							h.addQueue(keyHigh)
							qLow := h.addQueue(keyLow)

							itemLow := h.newTestItem("item-low", keyLow, testTTL)
							require.NoError(t, qLow.Add(itemLow))

							h.interFlowPolicySelectQueue = func(
								band framework.PriorityBandAccessor,
							) (framework.FlowQueueAccessor, error) {
								if band.Priority() == testFlow.Priority {
									return nil, errors.New("policy failure") // Fail high-priority.
								}
								// Succeed for low-priority.
								q, _ := h.managedQueue(keyLow)
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
				flowB := types.FlowKey{ID: "flow-b", Priority: testFlow.Priority}
				h.addQueue(testFlow)
				qB := h.addQueue(flowB)
				itemB := h.newTestItem("item-b", flowB, testTTL)
				require.NoError(t, qB.Add(itemB))

				// This filter only allows flow-b.
				h.processor.dispatchFilter = func(
					_ context.Context,
					originalBand framework.PriorityBandAccessor,
					_ logr.Logger,
				) (framework.PriorityBandAccessor, bool) {
					return newSubsetPriorityBandAccessor(originalBand, []types.FlowKey{flowB}), false
				}

				// This policy will be given the filtered view, so it should only see flow-b.
				h.interFlowPolicySelectQueue = func(band framework.PriorityBandAccessor) (framework.FlowQueueAccessor, error) {
					var flowIDs []string
					band.IterateQueues(func(fqa framework.FlowQueueAccessor) bool {
						flowIDs = append(flowIDs, fqa.FlowKey().ID)
						return true
					})
					// This is the core assertion of the test.
					require.ElementsMatch(t, []string{flowB.ID}, flowIDs, "Policy should only see the filtered flow")

					// Select flow-b to prove the chain works.
					q, _ := h.managedQueue(flowB)
					return q.FlowQueueAccessor(), nil
				}

				// --- ACT ---
				dispatched := h.processor.dispatchCycle(context.Background())

				// --- ASSERT ---
				assert.True(t, dispatched, "An item should have been dispatched from the filtered flow")
				assert.Equal(t, types.QueueOutcomeDispatched, itemB.finalState.Outcome,
					"The dispatched item's outcome should be correct")
				assert.NoError(t, itemB.finalState.Err, "The dispatched item should not have an error")
			})

			t.Run("should guarantee strict priority by starving lower priority items", func(t *testing.T) {
				t.Parallel()
				// --- ARRANGE ---
				h := newTestHarness(t, testCleanupTick)
				keyHigh := types.FlowKey{ID: "flow-high", Priority: 20}
				keyLow := types.FlowKey{ID: "flow-low", Priority: 10}
				qHigh := h.addQueue(keyHigh)
				qLow := h.addQueue(keyLow)

				const numItems = 3
				highPrioItems := make([]*FlowItem, numItems)
				lowPrioItems := make([]*FlowItem, numItems)
				for i := range numItems {
					// Add high priority items.
					itemH := h.newTestItem(fmt.Sprintf("req-high-%d", i), keyHigh, testTTL)
					require.NoError(t, qHigh.Add(itemH))
					highPrioItems[i] = itemH

					// Add low priority items.
					itemL := h.newTestItem(fmt.Sprintf("req-low-%d", i), keyLow, testTTL)
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
					assert.Equal(t, types.QueueOutcomeDispatched, item.finalState.Outcome,
						"High-priority item should be dispatched")
					assert.NoError(t, item.finalState.Err, "Dispatched high-priority item should not have an error")
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
							h.ManagedQueueFunc = func(types.FlowKey) (contracts.ManagedQueue, error) { return nil, registryErr }
						},
						expectedErr: registryErr,
					},
					{
						name: "on queue remove failure",
						setupMocks: func(h *testHarness) {
							h.ManagedQueueFunc = func(types.FlowKey) (contracts.ManagedQueue, error) {
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
						item := h.newTestItem("req-dispatch-fail", testFlow, testTTL)
						err := h.processor.dispatchItem(item, h.logger)
						require.Error(t, err, "dispatchItem should return an error")
						assert.ErrorIs(t, err, tc.expectedErr, "The underlying registry error should be preserved")
					})
				}
			})

			t.Run("should evict item that expires at moment of dispatch", func(t *testing.T) {
				t.Parallel()
				// --- ARRANGE ---
				h := newTestHarness(t, testCleanupTick)
				item := h.newTestItem("req-expired-dispatch", testFlow, testShortTTL)

				h.ManagedQueueFunc = func(types.FlowKey) (contracts.ManagedQueue, error) {
					return &mocks.MockManagedQueue{
						RemoveFunc: func(types.QueueItemHandle) (types.QueueItemAccessor, error) {
							return item, nil
						},
					}, nil
				}

				// --- ACT ---
				h.clock.Step(testShortTTL * 2) // Make the item expire.
				err := h.processor.dispatchItem(item, h.logger)

				// --- ASSERT ---
				// First, check the error returned by `dispatchItem`.
				require.Error(t, err, "dispatchItem should return an error for an expired item")
				assert.ErrorIs(t, err, types.ErrTTLExpired, "The error should be of type ErrTTLExpired")

				// Second, check the final state of the item itself.
				assert.Equal(t, types.QueueOutcomeEvictedTTL, item.finalState.Outcome,
					"The item's final outcome should be EvictedTTL")
				require.Error(t, item.finalState.Err, "The item's final state should contain an error")
				assert.ErrorIs(t, item.finalState.Err, types.ErrTTLExpired,
					"The item's final error should be of type ErrTTLExpired")
			})
		})

		t.Run("cleanup and utility methods", func(t *testing.T) {
			t.Parallel()

			t.Run("should remove and finalize expired items", func(t *testing.T) {
				t.Parallel()
				// --- ARRANGE ---
				h := newTestHarness(t, testCleanupTick)
				// Create an item that is already expired relative to the cleanup time.
				item := h.newTestItem("req-expired", testFlow, 1*time.Millisecond)
				q := h.addQueue(testFlow)
				require.NoError(t, q.Add(item))
				cleanupTime := h.clock.Now().Add(10 * time.Millisecond)

				// --- ACT ---
				h.processor.cleanupExpired(cleanupTime)

				// --- ASSERT ---
				assert.Equal(t, types.QueueOutcomeEvictedTTL, item.finalState.Outcome, "Item outcome should be EvictedTTL")
				require.Error(t, item.finalState.Err, "Item should have an error")
				assert.ErrorIs(t, item.finalState.Err, types.ErrTTLExpired, "Item error should be ErrTTLExpired")
			})

			t.Run("should evict all items on shutdown", func(t *testing.T) {
				t.Parallel()
				// --- ARRANGE ---
				h := newTestHarness(t, testCleanupTick)
				item := h.newTestItem("req-pending", testFlow, testTTL)
				q := h.addQueue(testFlow)
				require.NoError(t, q.Add(item))

				// --- ACT ---
				h.processor.evictAll()

				// --- ASSERT ---
				assert.Equal(t, types.QueueOutcomeEvictedOther, item.finalState.Outcome, "Item outcome should be EvictedOther")
				require.Error(t, item.finalState.Err, "Item should have an error")
				assert.ErrorIs(t, item.finalState.Err, types.ErrFlowControllerNotRunning,
					"Item error should be ErrFlowControllerNotRunning")
			})

			t.Run("should handle registry errors gracefully during concurrent processing", func(t *testing.T) {
				t.Parallel()
				// --- ARRANGE ---
				h := newTestHarness(t, testCleanupTick)
				h.AllOrderedPriorityLevelsFunc = func() []int { return []int{testFlow.Priority} }
				h.PriorityBandAccessorFunc = func(p int) (framework.PriorityBandAccessor, error) {
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
					OriginalRequestV: typesmocks.NewMockFlowControlRequest(0, "bad-item", testFlow, context.Background()),
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
					key := types.FlowKey{
						ID:       fmt.Sprintf("flow-%d", i),
						Priority: testFlow.Priority,
					}
					h.addQueue(key)
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

	t.Run("Public API", func(t *testing.T) {
		t.Parallel()

		t.Run("Submit", func(t *testing.T) {
			t.Parallel()

			t.Run("should return ErrProcessorBusy when channel is full", func(t *testing.T) {
				t.Parallel()
				h := newTestHarness(t, testCleanupTick)
				h.processor.enqueueChan = make(chan *FlowItem, 1)
				h.processor.enqueueChan <- h.newTestItem("item-filler", testFlow, testTTL) // Fill the channel to capacity.

				// The next submit should be non-blocking and fail immediately.
				err := h.processor.Submit(h.newTestItem("item-to-reject", testFlow, testTTL))
				require.Error(t, err, "Submit must return an error when the channel is full")
				assert.ErrorIs(t, err, ErrProcessorBusy, "The returned error must be ErrProcessorBusy")
			})
		})

		t.Run("SubmitOrBlock", func(t *testing.T) {
			t.Parallel()

			t.Run("should block when channel is full and succeed when space becomes available", func(t *testing.T) {
				t.Parallel()
				h := newTestHarness(t, testCleanupTick)
				h.processor.enqueueChan = make(chan *FlowItem, 1)
				h.processor.enqueueChan <- h.newTestItem("item-filler", testFlow, testTTL) // Fill the channel to capacity.

				itemToSubmit := h.newTestItem("item-to-block", testFlow, testTTL)
				submitErr := make(chan error, 1)

				// Run `SubmitOrBlock` in a separate goroutine, as it will block.
				go func() {
					submitErr <- h.processor.SubmitOrBlock(context.Background(), itemToSubmit)
				}()

				// Prove that the call is blocking by ensuring it hasn't returned an error yet.
				time.Sleep(20 * time.Millisecond)
				require.Len(t, submitErr, 0, "SubmitOrBlock should be blocking and not have returned yet")
				<-h.processor.enqueueChan // Make space in the channel. This should unblock the goroutine.

				select {
				case err := <-submitErr:
					require.NoError(t, err, "SubmitOrBlock should succeed and return no error after being unblocked")
				case <-time.After(testWaitTimeout):
					t.Fatal("SubmitOrBlock did not return after space was made in the channel")
				}
			})

			t.Run("should unblock and return context error on cancellation", func(t *testing.T) {
				t.Parallel()
				h := newTestHarness(t, testCleanupTick)
				h.processor.enqueueChan = make(chan *FlowItem) // Use an unbuffered channel to guarantee the first send blocks.
				itemToSubmit := h.newTestItem("item-to-cancel", testFlow, testTTL)
				submitErr := make(chan error, 1)
				ctx, cancel := context.WithCancel(context.Background())

				// Run `SubmitOrBlock` in a separate goroutine, as it will block.
				go func() {
					submitErr <- h.processor.SubmitOrBlock(ctx, itemToSubmit)
				}()

				// Prove that the call is blocking.
				time.Sleep(20 * time.Millisecond)
				require.Len(t, submitErr, 0, "SubmitOrBlock should be blocking and not have returned yet")
				cancel() // Cancel the context. This should unblock the goroutine.

				select {
				case err := <-submitErr:
					require.Error(t, err, "SubmitOrBlock should return an error after context cancellation")
					assert.ErrorIs(t, err, context.Canceled, "The returned error must be context.Canceled")
				case <-time.After(testWaitTimeout):
					t.Fatal("SubmitOrBlock did not return after context was cancelled")
				}
			})

			t.Run("should reject immediately if shutting down", func(t *testing.T) {
				t.Parallel()
				h := newTestHarness(t, testCleanupTick)
				item := h.newTestItem("req-shutdown-reject", testFlow, testTTL)
				h.addQueue(testFlow)

				h.Start()
				h.Go()
				h.Stop() // Stop the processor, then immediately try to enqueue.
				err := h.processor.SubmitOrBlock(context.Background(), item)

				require.Error(t, err, "SubmitOrBlock should return an error when shutting down")
				assert.ErrorIs(t, err, types.ErrFlowControllerNotRunning, "The error should be ErrFlowControllerNotRunning")

				outcome, err := h.waitForFinalization(item)
				assert.Equal(t, types.QueueOutcomeRejectedOther, outcome, "The outcome should be RejectedOther")
				require.Error(t, err, "Finalization should include an error")
				assert.ErrorIs(t, err, types.ErrFlowControllerNotRunning,
					"The finalization error should be ErrFlowControllerNotRunning")
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
				typesmocks.NewMockFlowControlRequest(100, "req-not-expired", testFlow, context.Background()),
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
				typesmocks.NewMockFlowControlRequest(100, "req-not-expired-no-ttl", testFlow, context.Background()),
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
				typesmocks.NewMockFlowControlRequest(100, "req-ttl-expired", testFlow, context.Background()),
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
				typesmocks.NewMockFlowControlRequest(100, "req-ctx-cancelled", testFlow, ctxCancelled),
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
				i := NewItem(
					typesmocks.NewMockFlowControlRequest(100, "req-finalized", testFlow, context.Background()),
					testTTL,
					now)
				i.Finalize(types.QueueOutcomeDispatched, nil)
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
}
