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
	"sync"
	"sync/atomic"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue/listqueue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	typesmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

// testStatsReconciler is a mock implementation of the `parentStatsReconciler` function.
// It captures the deltas it's called with, allowing tests to assert on them.
type testStatsReconciler struct {
	mu              sync.Mutex
	lenDelta        int64
	byteSizeDelta   int64
	invocationCount int
}

func (r *testStatsReconciler) reconcile(lenDelta, byteSizeDelta int64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.lenDelta += lenDelta
	r.byteSizeDelta += byteSizeDelta
	r.invocationCount++
}

func (r *testStatsReconciler) getStats() (lenDelta, byteSizeDelta int64, count int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.lenDelta, r.byteSizeDelta, r.invocationCount
}

// testFixture holds the components needed for a `managedQueue` test.
type testFixture struct {
	mq             *managedQueue
	mockQueue      *frameworkmocks.MockSafeQueue
	mockPolicy     *frameworkmocks.MockIntraFlowDispatchPolicy
	reconciler     *testStatsReconciler
	flowSpec       types.FlowSpecification
	mockComparator *frameworkmocks.MockItemComparator
}

// setupTestManagedQueue creates a new test fixture for testing the `managedQueue`.
func setupTestManagedQueue(t *testing.T) *testFixture {
	t.Helper()

	mockQueue := &frameworkmocks.MockSafeQueue{}
	reconciler := &testStatsReconciler{}
	flowSpec := types.FlowSpecification{ID: "test-flow", Priority: 1}
	mockComparator := &frameworkmocks.MockItemComparator{}
	mockPolicy := &frameworkmocks.MockIntraFlowDispatchPolicy{
		ComparatorV: mockComparator,
	}

	mq := newManagedQueue(
		mockQueue,
		mockPolicy,
		flowSpec,
		logr.Discard(),
		reconciler.reconcile,
	)
	require.NotNil(t, mq, "newManagedQueue should not return nil")

	return &testFixture{
		mq:             mq,
		mockQueue:      mockQueue,
		mockPolicy:     mockPolicy,
		reconciler:     reconciler,
		flowSpec:       flowSpec,
		mockComparator: mockComparator,
	}
}

func TestManagedQueue_New(t *testing.T) {
	t.Parallel()
	f := setupTestManagedQueue(t)

	assert.Zero(t, f.mq.Len(), "A new managedQueue should have a length of 0")
	assert.Zero(t, f.mq.ByteSize(), "A new managedQueue should have a byte size of 0")
	assert.True(t, f.mq.isActive.Load(), "A new managedQueue should be active")
}

func TestManagedQueue_Add(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name                  string
		itemByteSize          uint64
		mockAddError          error
		markAsDraining        bool
		expectError           bool
		expectedErrorIs       error
		expectedLen           int
		expectedByteSize      uint64
		expectedLenDelta      int64
		expectedByteSizeDelta int64
		expectedReconcile     bool
	}{
		{
			name:                  "Success",
			itemByteSize:          100,
			expectError:           false,
			expectedLen:           1,
			expectedByteSize:      100,
			expectedLenDelta:      1,
			expectedByteSizeDelta: 100,
			expectedReconcile:     true,
		},
		{
			name:                  "Error from underlying queue",
			itemByteSize:          100,
			mockAddError:          errors.New("queue full"),
			expectError:           true,
			expectedLen:           0,
			expectedByteSize:      0,
			expectedLenDelta:      0,
			expectedByteSizeDelta: 0,
			expectedReconcile:     false,
		},
		{
			name:                  "Error on inactive queue",
			itemByteSize:          100,
			markAsDraining:        true,
			expectError:           true,
			expectedErrorIs:       contracts.ErrFlowInstanceNotFound,
			expectedLen:           0,
			expectedByteSize:      0,
			expectedLenDelta:      0,
			expectedByteSizeDelta: 0,
			expectedReconcile:     false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			f := setupTestManagedQueue(t)

			// Configure mock
			f.mockQueue.AddFunc = func(item types.QueueItemAccessor) error {
				return tc.mockAddError
			}

			if tc.markAsDraining {
				f.mq.markAsDraining()
				assert.False(t, f.mq.isActive.Load(), "Setup: queue should be marked as inactive")
			}

			item := typesmocks.NewMockQueueItemAccessor(tc.itemByteSize, "req-1", "test-flow")
			err := f.mq.Add(item)

			if tc.expectError {
				require.Error(t, err, "Add should have returned an error")
				if tc.expectedErrorIs != nil {
					assert.ErrorIs(t, err, tc.expectedErrorIs, "Error should wrap the expected sentinel error")
				}
			} else {
				require.NoError(t, err, "Add should not have returned an error")
			}

			// Assert final state
			assert.Equal(t, tc.expectedLen, f.mq.Len(), "Final length should be as expected")
			assert.Equal(t, tc.expectedByteSize, f.mq.ByteSize(), "Final byte size should be as expected")

			// Assert reconciler state
			lenDelta, byteSizeDelta, count := f.reconciler.getStats()
			assert.Equal(t, tc.expectedLenDelta, lenDelta, "Reconciler length delta should be as expected")
			assert.Equal(t, tc.expectedByteSizeDelta, byteSizeDelta, "Reconciler byte size delta should be as expected")
			if tc.expectedReconcile {
				assert.Equal(t, 1, count, "Reconciler should have been called once")
			} else {
				assert.Zero(t, count, "Reconciler should not have been called")
			}
		})
	}
}

func TestManagedQueue_Remove(t *testing.T) {
	t.Parallel()
	f := setupTestManagedQueue(t)

	// Setup initial state
	initialItem := typesmocks.NewMockQueueItemAccessor(100, "req-1", "test-flow")
	f.mockQueue.AddFunc = func(item types.QueueItemAccessor) error { return nil }
	err := f.mq.Add(initialItem)
	require.NoError(t, err, "Setup: Adding an item should not fail")
	require.Equal(t, 1, f.mq.Len(), "Setup: Length should be 1 after adding an item")

	// --- Test Success ---
	t.Run("Success", func(t *testing.T) {
		// Configure mock for Remove
		f.mockQueue.RemoveFunc = func(handle types.QueueItemHandle) (types.QueueItemAccessor, error) {
			return initialItem, nil
		}

		// Perform Remove
		removedItem, err := f.mq.Remove(initialItem.Handle())
		require.NoError(t, err, "Remove should not return an error")
		assert.Equal(t, initialItem, removedItem, "Remove should return the correct item")

		// Assert final state
		assert.Zero(t, f.mq.Len(), "Length should be 0 after removing the only item")
		assert.Zero(t, f.mq.ByteSize(), "ByteSize should be 0 after removing the only item")

		// Assert reconciler state
		lenDelta, byteSizeDelta, count := f.reconciler.getStats()
		assert.Equal(t, int64(0), lenDelta, "Net length delta should be 0 after add and remove")
		assert.Equal(t, int64(0), byteSizeDelta, "Net byte size delta should be 0 after add and remove")
		assert.Equal(t, 2, count, "Reconciler should have been called for both Add and Remove")
	})

	// --- Test Error ---
	t.Run("Error", func(t *testing.T) {
		f := setupTestManagedQueue(t)
		require.NoError(t, f.mq.Add(initialItem), "Setup: Adding an item should not fail")

		// Configure mock to return an error
		expectedErr := errors.New("item not found")
		f.mockQueue.RemoveFunc = func(handle types.QueueItemHandle) (types.QueueItemAccessor, error) {
			return nil, expectedErr
		}

		_, err := f.mq.Remove(initialItem.Handle())
		require.ErrorIs(t, err, expectedErr, "Remove should propagate the error")

		// Assert state did not change
		assert.Equal(t, 1, f.mq.Len(), "Length should not change on a failed remove")
		assert.Equal(t, uint64(100), f.mq.ByteSize(), "ByteSize should not change on a failed remove")

		// Assert reconciler was not called for the remove
		_, _, count := f.reconciler.getStats()
		assert.Equal(t, 1, count, "Reconciler should only have been called for the initial Add")
	})
}

func TestManagedQueue_CleanupAndDrain(t *testing.T) {
	t.Parallel()
	item1 := typesmocks.NewMockQueueItemAccessor(10, "req-1", "test-flow")
	item2 := typesmocks.NewMockQueueItemAccessor(20, "req-2", "test-flow")
	item3 := typesmocks.NewMockQueueItemAccessor(30, "req-3", "test-flow")

	// --- Test Cleanup ---
	t.Run("Cleanup", func(t *testing.T) {
		t.Parallel()
		f := setupTestManagedQueue(t)
		// Add initial items
		require.NoError(t, f.mq.Add(item1), "Setup: Add item1 should not fail")
		require.NoError(t, f.mq.Add(item2), "Setup: Add item2 should not fail")
		require.NoError(t, f.mq.Add(item3), "Setup: Add item3 should not fail")
		require.Equal(t, 3, f.mq.Len(), "Setup: Initial length should be 3")
		require.Equal(t, uint64(60), f.mq.ByteSize(), "Setup: Initial byte size should be 60")

		// Configure mock to clean up item2
		f.mockQueue.CleanupFunc = func(p framework.PredicateFunc) ([]types.QueueItemAccessor, error) {
			return []types.QueueItemAccessor{item2}, nil
		}

		cleaned, err := f.mq.Cleanup(func(i types.QueueItemAccessor) bool { return true })
		require.NoError(t, err, "Cleanup should not return an error")
		require.Len(t, cleaned, 1, "Cleanup should return one item")

		// Assert final state
		assert.Equal(t, 2, f.mq.Len(), "Length should be 2 after cleanup")
		assert.Equal(t, uint64(40), f.mq.ByteSize(), "ByteSize should be 40 after cleanup")

		// Assert reconciler state (3 adds, 1 cleanup)
		lenDelta, byteSizeDelta, count := f.reconciler.getStats()
		assert.Equal(t, int64(2), lenDelta, "Net length delta should be 2")
		assert.Equal(t, int64(40), byteSizeDelta, "Net byte size delta should be 40")
		assert.Equal(t, 4, count, "Reconciler should have been called 4 times")
	})

	// --- Test Drain ---
	t.Run("Drain", func(t *testing.T) {
		t.Parallel()
		f := setupTestManagedQueue(t)
		// Add initial items
		require.NoError(t, f.mq.Add(item1), "Setup: Add item1 should not fail")
		require.NoError(t, f.mq.Add(item2), "Setup: Add item2 should not fail")
		require.Equal(t, 2, f.mq.Len(), "Setup: Initial length should be 2")
		require.Equal(t, uint64(30), f.mq.ByteSize(), "Setup: Initial byte size should be 30")

		// Configure mock to drain both items
		f.mockQueue.DrainFunc = func() ([]types.QueueItemAccessor, error) {
			return []types.QueueItemAccessor{item1, item2}, nil
		}

		drained, err := f.mq.Drain()
		require.NoError(t, err, "Drain should not return an error")
		require.Len(t, drained, 2, "Drain should return two items")

		// Assert final state
		assert.Zero(t, f.mq.Len(), "Length should be 0 after drain")
		assert.Zero(t, f.mq.ByteSize(), "ByteSize should be 0 after drain")

		// Assert reconciler state (2 adds, 1 drain)
		lenDelta, byteSizeDelta, count := f.reconciler.getStats()
		assert.Equal(t, int64(0), lenDelta, "Net length delta should be 0")
		assert.Equal(t, int64(0), byteSizeDelta, "Net byte size delta should be 0")
		assert.Equal(t, 3, count, "Reconciler should have been called 3 times")
	})

	// --- Test Error Paths ---
	t.Run("ErrorPaths", func(t *testing.T) {
		f := setupTestManagedQueue(t)
		require.NoError(t, f.mq.Add(item1), "Setup: Adding an item should not fail")
		initialLen, initialByteSize := f.mq.Len(), f.mq.ByteSize()

		expectedErr := errors.New("internal error")

		// Cleanup error
		f.mockQueue.CleanupFunc = func(p framework.PredicateFunc) ([]types.QueueItemAccessor, error) {
			return nil, expectedErr
		}
		_, err := f.mq.Cleanup(func(i types.QueueItemAccessor) bool { return true })
		require.ErrorIs(t, err, expectedErr, "Cleanup should propagate error")
		assert.Equal(t, initialLen, f.mq.Len(), "Len should not change on Cleanup error")
		assert.Equal(t, initialByteSize, f.mq.ByteSize(), "ByteSize should not change on Cleanup error")

		// Drain error
		f.mockQueue.DrainFunc = func() ([]types.QueueItemAccessor, error) {
			return nil, expectedErr
		}
		_, err = f.mq.Drain()
		require.ErrorIs(t, err, expectedErr, "Drain should propagate error")
		assert.Equal(t, initialLen, f.mq.Len(), "Len should not change on Drain error")
		assert.Equal(t, initialByteSize, f.mq.ByteSize(), "ByteSize should not change on Drain error")
	})
}

func TestManagedQueue_FlowQueueAccessor(t *testing.T) {
	t.Parallel()
	f := setupTestManagedQueue(t)
	item := typesmocks.NewMockQueueItemAccessor(100, "req-1", "test-flow")

	// Setup underlying queue state
	f.mockQueue.PeekHeadV = item
	f.mockQueue.PeekTailV = item
	f.mockQueue.NameV = "MockQueue"
	f.mockQueue.CapabilitiesV = []framework.QueueCapability{framework.CapabilityFIFO}

	// Add an item to populate the managed queue's stats
	require.NoError(t, f.mq.Add(item), "Setup: Adding an item should not fail")

	// Get the accessor
	accessor := f.mq.FlowQueueAccessor()
	require.NotNil(t, accessor, "FlowQueueAccessor should not be nil")

	// Assert that the accessor methods reflect the underlying state
	assert.Equal(t, f.mq.Name(), accessor.Name(), "Accessor Name() should match managed queue")
	assert.Equal(t, f.mq.Capabilities(), accessor.Capabilities(), "Accessor Capabilities() should match managed queue")
	assert.Equal(t, f.mq.Len(), accessor.Len(), "Accessor Len() should match managed queue")
	assert.Equal(t, f.mq.ByteSize(), accessor.ByteSize(), "Accessor ByteSize() should match managed queue")
	assert.Equal(t, f.flowSpec, accessor.FlowSpec(), "Accessor FlowSpec() should match managed queue")
	assert.Equal(t, f.mockComparator, accessor.Comparator(), "Accessor Comparator() should match the one from the policy")
	assert.Equal(t, f.mockComparator, f.mq.Comparator(), "ManagedQueue Comparator() should match the one from the policy")

	peekedHead, err := accessor.PeekHead()
	require.NoError(t, err, "Accessor PeekHead() should not return an error")
	assert.Equal(t, item, peekedHead, "Accessor PeekHead() should return the correct item")

	peekedTail, err := accessor.PeekTail()
	require.NoError(t, err, "Accessor PeekTail() should not return an error")
	assert.Equal(t, item, peekedTail, "Accessor PeekTail() should return the correct item")
}

func TestManagedQueue_Concurrency(t *testing.T) {
	t.Parallel()

	// Use a real `listqueue` since it's concurrent-safe.
	lq, err := queue.NewQueueFromName(listqueue.ListQueueName, nil)
	require.NoError(t, err, "Setup: creating a real listqueue should not fail")

	reconciler := &testStatsReconciler{}
	flowSpec := types.FlowSpecification{ID: "conc-test-flow", Priority: 1}
	mq := newManagedQueue(lq, nil, flowSpec, logr.Discard(), reconciler.reconcile)

	const (
		numGoroutines   = 20
		opsPerGoroutine = 200
		itemByteSize    = 10
		initialItems    = 500
	)

	var wg sync.WaitGroup
	var successfulAdds, successfulRemoves atomic.Int64

	// Use a channel as a concurrent-safe pool of handles for removal.
	handles := make(chan types.QueueItemHandle, initialItems+(numGoroutines*opsPerGoroutine))

	// Pre-fill the queue to give removers something to do immediately.
	for range initialItems {
		item := typesmocks.NewMockQueueItemAccessor(uint64(itemByteSize), "initial", "flow")
		require.NoError(t, mq.Add(item), "Setup: pre-filling queue should not fail")
		handles <- item.Handle()
	}
	// Reset reconciler stats after setup so we only measure the concurrent phase.
	*reconciler = testStatsReconciler{}

	// Launch goroutines to perform a mix of concurrent operations.
	wg.Add(numGoroutines)
	for i := range numGoroutines {
		go func(routineID int) {
			defer wg.Done()
			for j := range opsPerGoroutine {
				// Mix up operations between adding and removing.
				if (routineID+j)%2 == 0 {
					// Add operation
					item := typesmocks.NewMockQueueItemAccessor(uint64(itemByteSize), "req", "flow")
					if err := mq.Add(item); err == nil {
						successfulAdds.Add(1)
						handles <- item.Handle()
					}
				} else {
					// Remove operation
					select {
					case handle := <-handles:
						if _, err := mq.Remove(handle); err == nil {
							successfulRemoves.Add(1)
						}
					default:
						// No handles available, do nothing. This can happen if removers are faster than adders.
					}
				}
			}
		}(i)
	}
	wg.Wait()

	// All concurrent operations are complete. Drain any remaining items to get the final count.
	drainedItems, err := mq.Drain()
	require.NoError(t, err, "Draining the queue at the end should not fail")

	// Final consistency checks.
	finalItemCount := len(drainedItems)
	finalByteSize := mq.ByteSize()

	// The number of items left in the queue should match our tracking.
	expectedFinalItemCount := initialItems + int(successfulAdds.Load()) - int(successfulRemoves.Load())
	assert.Equal(t, expectedFinalItemCount, finalItemCount, "Final item count should match initial + adds - removes")

	// The queue's internal stats must be zero after draining.
	assert.Zero(t, mq.Len(), "Managed queue length must be zero after drain")
	assert.Zero(t, finalByteSize, "Managed queue byte size must be zero after drain")

	// The net change reported to the reconciler must match the net change from the concurrent phase,
	// plus the final drain.
	netLenChangeDuringConcurrentPhase := successfulAdds.Load() - successfulRemoves.Load()
	netByteSizeChangeDuringConcurrentPhase := netLenChangeDuringConcurrentPhase * itemByteSize

	lenDelta, byteSizeDelta, _ := reconciler.getStats()

	expectedLenDelta := netLenChangeDuringConcurrentPhase - int64(finalItemCount)
	expectedByteSizeDelta := netByteSizeChangeDuringConcurrentPhase - int64(uint64(finalItemCount)*itemByteSize)

	assert.Equal(t, expectedLenDelta, lenDelta,
		"Net length delta in reconciler should match the net change from all operations")
	assert.Equal(t, expectedByteSizeDelta, byteSizeDelta,
		"Net byte size delta in reconciler should match the net change from all operations")
}
