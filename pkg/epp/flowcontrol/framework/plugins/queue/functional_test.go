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

package queue

import (
	"fmt"
	"slices"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	typesmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

// enqueueTimeComparator orders items by their enqueue time (earlier first).
// Used as the default comparator for basic FIFO-like ordering tests.
var enqueueTimeComparator = &frameworkmocks.MockItemComparator{
	ScoreTypeV: "enqueue_time_ns_asc",
	FuncV: func(a, b types.QueueItemAccessor) bool {
		return a.EnqueueTime().Before(b.EnqueueTime())
	},
}

// byteSizeComparator orders items by their byte size (smaller first).
var byteSizeComparator = &frameworkmocks.MockItemComparator{
	ScoreTypeV: "byte_size_asc",
	FuncV: func(a, b types.QueueItemAccessor) bool {
		return a.OriginalRequest().ByteSize() < b.OriginalRequest().ByteSize()
	},
}

// reverseEnqueueTimeComparator orders items by their enqueue time (later first - LIFO).
// Used to test CapabilityPriorityConfigurable queues with a non-FIFO ordering.
var reverseEnqueueTimeComparator = &frameworkmocks.MockItemComparator{
	ScoreTypeV: "enqueue_time_ns_desc",
	FuncV: func(a, b types.QueueItemAccessor) bool {
		return a.EnqueueTime().After(b.EnqueueTime())
	},
}

// testLifecycleAndOrdering is a helper function to execute a standard sequence of Add, PeekHead, and Remove operations
// on a queue. It verifies the queue's state (length, byte size) and item ordering based on the provided `itemsInOrder`
// slice, which should be pre-sorted according to the `comparatorName` (which is just a string for
// logging/identification).
// This function is crucial for testing different ordering logic (FIFO, custom priority).
func testLifecycleAndOrdering(
	t *testing.T,
	q framework.SafeQueue,
	itemsInOrder []*typesmocks.MockQueueItemAccessor,
	comparatorName string,
) {
	t.Helper()

	// PeekHead/PeekTail on empty queue
	peeked := q.PeekHead()
	assert.Nil(t, peeked, "[%s] PeekHead on empty queue should return a nil item", comparatorName)
	peeked = q.PeekTail()
	assert.Nil(t, peeked, "[%s] PeekTail on empty queue should return a nil item", comparatorName)

	// Add items
	currentExpectedLen := 0
	var currentExpectedByteSize uint64
	for i, item := range itemsInOrder {
		q.Add(item)
		require.NotNil(t, item.Handle(), "[%s] Add must assign a non-nil handle to the item (item %d, ID: %s)",
			comparatorName, i, item.OriginalRequest().ID())
		require.False(t, item.Handle().IsInvalidated(),
			"[%s] A new handle from Add must not be invalidated (item %d, ID: %s)",
			comparatorName, i, item.OriginalRequest().ID())

		currentExpectedLen++
		currentExpectedByteSize += item.OriginalRequest().ByteSize()
		assert.Equal(t, currentExpectedLen, q.Len(), "[%s] Len() must be correct after Add (item %d, ID: %s)",
			comparatorName, i, item.OriginalRequest().ID())
		assert.Equal(t, currentExpectedByteSize, q.ByteSize(),
			"[%s] ByteSize() must be correct after Add (item %d, ID: %s)",
			comparatorName, i, item.OriginalRequest().ID())
	}

	// Check final state after adds
	initialLen := len(itemsInOrder)
	var expectedTotalByteSize uint64
	for _, item := range itemsInOrder {
		expectedTotalByteSize += item.OriginalRequest().ByteSize()
	}
	assert.Equal(t, initialLen, q.Len(), "[%s] Len() should return the correct count after all items are added",
		comparatorName)
	assert.Equal(t, expectedTotalByteSize, q.ByteSize(),
		"[%s] ByteSize() should return the correct sum after all items are added", comparatorName)

	// Peek and Remove cycle to verify ordering
	expectedLen := initialLen
	expectedByteSize := expectedTotalByteSize
	for i, expectedItem := range itemsInOrder {
		// Verify PeekHead
		peeked = q.PeekHead()
		require.NotNil(t, peeked, "[%s] PeekHead should return a non-nil item (iteration %d)", comparatorName, i)
		assert.Equal(t, expectedItem.OriginalRequest().ID(), peeked.OriginalRequest().ID(),
			"[%s] PeekHead must return the item (ID: %s) at the head of the queue (iteration %d)",
			comparatorName, expectedItem.OriginalRequest().ID(), i)
		peekedHandle := peeked.Handle()
		require.NotNil(t, peekedHandle, "[%s] Handle from a peeked item must not be nil (iteration %d)", comparatorName, i)
		require.False(t, peekedHandle.IsInvalidated(),
			"[%s] Handle from a peeked item must not be invalidated (iteration %d)", comparatorName, i)
		assert.Equal(t, expectedLen, q.Len(), "[%s] Len() must be unchanged after PeekHead (iteration %d)",
			comparatorName, i)
		assert.Equal(t, expectedByteSize, q.ByteSize(),
			"[%s] ByteSize() must be unchanged after PeekHead (iteration %d)", comparatorName, i)

		// Verify PeekTail
		peekedTail := q.PeekTail()
		require.NotNil(t, peekedTail, "[%s] PeekTail should return a non-nil item (iteration %d)", comparatorName, i)
		// The tail is the last item in the *remaining* ordered slice.
		expectedTailItem := itemsInOrder[len(itemsInOrder)-1]
		assert.Equal(t, expectedTailItem.OriginalRequest().ID(), peekedTail.OriginalRequest().ID(),
			"[%s] PeekTail must return the item with the lowest priority (iteration %d)", comparatorName, i)

		// Remove the head item
		removed, removeErr := q.Remove(peekedHandle)
		require.NoError(t, removeErr, "[%s] Remove with a valid handle should not fail (iteration %d, item ID: %s)",
			comparatorName, i, expectedItem.OriginalRequest().ID())
		require.NotNil(t, removed, "[%s] Remove should return the removed item (iteration %d)", comparatorName, i)
		assert.Equal(t, expectedItem.OriginalRequest().ID(), removed.OriginalRequest().ID(),
			"[%s] Remove should return the correct item (iteration %d)", comparatorName, i)
		assert.True(t, peekedHandle.IsInvalidated(),
			"[%s] Remove must invalidate the handle of the removed item (iteration %d)", comparatorName, i)

		expectedLen--
		expectedByteSize -= removed.OriginalRequest().ByteSize()
		assert.Equal(t, expectedLen, q.Len(), "[%s] Len() should be correctly updated after Remove (iteration %d)",
			comparatorName, i)
		assert.Equal(t, expectedByteSize, q.ByteSize(),
			"[%s] ByteSize() should be correctly updated after Remove (iteration %d)", comparatorName, i)
	}

	assert.Zero(t, q.Len(), "[%s] Queue length should be 0 after all items are removed", comparatorName)
	assert.Zero(t, q.ByteSize(), "[%s] Queue byte size should be 0 after all items are removed", comparatorName)

	peeked = q.PeekHead()
	assert.Nil(t, peeked, "[%s] PeekHead on an empty queue should return a nil item again", comparatorName)
}

// TestQueueConformance is the main conformance test suite for `framework.SafeQueue` implementations.
// It iterates over all queue implementations registered via `queue.MustRegisterQueue` and runs a series of sub-tests to
// ensure they adhere to the `framework.SafeQueue` contract.
func TestQueueConformance(t *testing.T) {
	t.Parallel()

	for queueName, constructor := range RegisteredQueues {
		t.Run(string(queueName), func(t *testing.T) {
			t.Parallel()
			flowKey := types.FlowKey{ID: "test-flow-1", Priority: 0}

			t.Run("Initialization", func(t *testing.T) {
				t.Parallel()
				q, err := constructor(enqueueTimeComparator)
				require.NoError(t, err, "Setup: creating queue for test should not fail")

				require.NotNil(t, q, "Constructor should return a non-nil queue instance")
				assert.Zero(t, q.Len(), "A new queue should have a length of 0")
				assert.Zero(t, q.ByteSize(), "A new queue should have a byte size of 0")
				assert.Equal(t, string(queueName), q.Name(), "Name() should return the registered name of the queue")
				assert.NotNil(t, q.Capabilities(), "Capabilities() should not return a nil slice")
				assert.NotEmpty(t, q.Capabilities(), "Capabilities() should return at least one capability")
			})

			t.Run("LifecycleAndOrdering_DefaultFIFO", func(t *testing.T) {
				t.Parallel()
				q, err := constructor(enqueueTimeComparator)
				require.NoError(t, err, "Setup: creating queue with enqueueTimeComparator should not fail")

				now := time.Now()

				item1 := typesmocks.NewMockQueueItemAccessor(100, "item1_fifo", flowKey)
				item1.EnqueueTimeV = now.Add(-2 * time.Second) // Earliest
				item2 := typesmocks.NewMockQueueItemAccessor(50, "item2_fifo", flowKey)
				item2.EnqueueTimeV = now.Add(-1 * time.Second) // Middle
				item3 := typesmocks.NewMockQueueItemAccessor(20, "item3_fifo", flowKey)
				item3.EnqueueTimeV = now // Latest

				itemsInFIFOOrder := []*typesmocks.MockQueueItemAccessor{item1, item2, item3}
				testLifecycleAndOrdering(t, q, itemsInFIFOOrder, "DefaultFIFO")
			})

			qForCapCheck, err := constructor(enqueueTimeComparator)
			if err == nil && slices.Contains(qForCapCheck.Capabilities(), framework.CapabilityPriorityConfigurable) {
				t.Run("LifecycleAndOrdering_PriorityConfigurable_ByteSize", func(t *testing.T) {
					t.Parallel()
					q, err := constructor(byteSizeComparator)
					require.NoError(t, err, "Setup: creating queue with byteSizeComparator should not fail")

					itemLarge := typesmocks.NewMockQueueItemAccessor(100, "itemLarge_prio", flowKey)
					itemSmall := typesmocks.NewMockQueueItemAccessor(20, "itemSmall_prio", flowKey)
					itemMedium := typesmocks.NewMockQueueItemAccessor(50, "itemMedium_prio", flowKey)

					itemsInByteSizeOrder := []*typesmocks.MockQueueItemAccessor{itemSmall, itemMedium, itemLarge}
					testLifecycleAndOrdering(t, q, itemsInByteSizeOrder, "PriorityByteSize")
				})

				t.Run("LifecycleAndOrdering_PriorityConfigurable_LIFO", func(t *testing.T) {
					t.Parallel()
					q, err := constructor(reverseEnqueueTimeComparator)
					require.NoError(t, err, "Setup: creating queue with reverseEnqueueTimeComparator should not fail")

					now := time.Now()
					item1 := typesmocks.NewMockQueueItemAccessor(100, "item1_lifo", flowKey)
					item1.EnqueueTimeV = now.Add(-2 * time.Second) // Earliest
					item2 := typesmocks.NewMockQueueItemAccessor(50, "item2_lifo", flowKey)
					item2.EnqueueTimeV = now.Add(-1 * time.Second) // Middle
					item3 := typesmocks.NewMockQueueItemAccessor(20, "item3_lifo", flowKey)
					item3.EnqueueTimeV = now // Latest

					itemsInLIFOOrder := []*typesmocks.MockQueueItemAccessor{item3, item2, item1}
					testLifecycleAndOrdering(t, q, itemsInLIFOOrder, "PriorityLIFO")
				})
			}

			t.Run("Remove_InvalidHandle", func(t *testing.T) {
				t.Parallel()
				q, err := constructor(enqueueTimeComparator)
				require.NoError(t, err, "Setup: creating queue for test should not fail")

				item := typesmocks.NewMockQueueItemAccessor(100, "item", flowKey)
				q.Add(item)

				otherQ, err := constructor(enqueueTimeComparator) // A different queue instance
				require.NoError(t, err, "Setup: creating otherQ should succeed")
				otherItem := typesmocks.NewMockQueueItemAccessor(10, "other_item", types.FlowKey{ID: "other-flow"})
				otherQ.Add(otherItem)
				alienHandle := otherItem.Handle()
				require.NotNil(t, alienHandle, "Setup: alien handle should not be nil")

				invalidatedHandle := &typesmocks.MockQueueItemHandle{}
				invalidatedHandle.Invalidate()

				foreignHandle := &typesmocks.MockQueueItemHandle{} // Different type

				testCases := []struct {
					name      string
					handle    types.QueueItemHandle
					expectErr error
				}{
					{name: "nil handle", handle: nil, expectErr: framework.ErrInvalidQueueItemHandle},
					{name: "invalidated handle", handle: invalidatedHandle, expectErr: framework.ErrInvalidQueueItemHandle},
					{name: "alien handle from other queue", handle: alienHandle, expectErr: framework.ErrQueueItemNotFound},
					{name: "foreign handle type", handle: foreignHandle, expectErr: framework.ErrInvalidQueueItemHandle},
				}

				for _, tc := range testCases {
					t.Run(tc.name, func(t *testing.T) {
						t.Parallel()
						currentLen := q.Len()
						currentByteSize := q.ByteSize()

						_, removeErr := q.Remove(tc.handle)
						assert.ErrorIs(t, removeErr, tc.expectErr, "Remove with %s should produce %v", tc.name, tc.expectErr)
						assert.Equal(t, currentLen, q.Len(), "The queue's length must not change after a failed Remove with %s",
							tc.name)
						assert.Equal(t, currentByteSize, q.ByteSize(),
							"The queue's byte size must not change after a failed Remove with %s", tc.name)
					})
				}
			})

			t.Run("Remove_NonHead", func(t *testing.T) {
				t.Parallel()
				q, err := constructor(enqueueTimeComparator)
				require.NoError(t, err, "Setup: creating queue for test should not fail")

				now := time.Now()
				item1 := typesmocks.NewMockQueueItemAccessor(10, "item1_nonhead", flowKey)
				item1.EnqueueTimeV = now.Add(-3 * time.Second)
				item2 := typesmocks.NewMockQueueItemAccessor(20, "item2_nonhead_TARGET", flowKey)
				item2.EnqueueTimeV = now.Add(-2 * time.Second)
				item3 := typesmocks.NewMockQueueItemAccessor(30, "item3_nonhead", flowKey)
				item3.EnqueueTimeV = now.Add(-1 * time.Second)

				q.Add(item1)
				q.Add(item2)
				q.Add(item3)
				require.Equal(t, 3, q.Len(), "Queue should have 3 items before removing non-head")
				handleNonHead := item2.Handle()

				removed, err := q.Remove(handleNonHead)
				require.NoError(t, err, "It should be possible to remove an item that is not the head")
				require.NotNil(t, removed, "Remove should return the removed item")
				assert.Equal(t, item2.OriginalRequest().ID(), removed.OriginalRequest().ID(),
					"Remove should return the correct item (item2)")
				assert.True(t, handleNonHead.IsInvalidated(), "Remove must invalidate the handle of the removed item")
				assert.Equal(t, 2, q.Len(), "Queue length should be 2 after removing non-head")
				assert.Equal(t, item1.OriginalRequest().ByteSize()+item3.OriginalRequest().ByteSize(), q.ByteSize(),
					"Byte size should be correct after removing non-head")

				// Attempt to remove again with the now-stale handle
				_, errStaleNonHead := q.Remove(handleNonHead)
				assert.ErrorIs(t, errStaleNonHead, framework.ErrInvalidQueueItemHandle,
					"Removing with a stale handle must fail with ErrInvalidQueueItemHandle")
			})

			predicateRemoveOddSizes := func(item types.QueueItemAccessor) bool {
				return item.OriginalRequest().ByteSize()%2 != 0
			}

			t.Run("Cleanup_EmptyQueue", func(t *testing.T) {
				t.Parallel()
				emptyQ, _ := constructor(enqueueTimeComparator)
				cleanedItems := emptyQ.Cleanup(predicateRemoveOddSizes)
				assert.Empty(t, cleanedItems, "Cleanup on an empty queue should return an empty slice")
				assert.Zero(t, emptyQ.Len(), "Len() should be 0 after Cleanup on an empty queue")
				assert.Zero(t, emptyQ.ByteSize(), "ByteSize() should be 0 after Cleanup on an empty queue")
			})

			t.Run("Cleanup_PredicateMatchesNone", func(t *testing.T) {
				t.Parallel()
				q, _ := constructor(enqueueTimeComparator)
				itemK1 := typesmocks.NewMockQueueItemAccessor(10, "k1_matchNone", flowKey)
				itemK2 := typesmocks.NewMockQueueItemAccessor(12, "k2_matchNone", flowKey)
				q.Add(itemK1)
				q.Add(itemK2)
				initialLen := q.Len()
				initialBs := q.ByteSize()

				cleanedItems := q.Cleanup(func(item types.QueueItemAccessor) bool { return false })
				assert.Empty(t, cleanedItems, "Cleanup should return an empty slice when no items match the predicate")
				assert.Equal(t, initialLen, q.Len(), "Len() should not change after Cleanup when no items match thepredicate")
				assert.Equal(t, initialBs, q.ByteSize(),
					"ByteSize() should not change after Cleanup when no items match the predicate")
				assert.False(t, itemK1.Handle().IsInvalidated(), "Handle for kept item 1 must NOT be invalidated")
				assert.False(t, itemK2.Handle().IsInvalidated(), "Handle for kept item 2 must NOT be invalidated")
			})

			t.Run("Cleanup_PredicateMatchesAll", func(t *testing.T) {
				t.Parallel()
				q, _ := constructor(enqueueTimeComparator)
				itemR1 := typesmocks.NewMockQueueItemAccessor(11, "r1_matchAll", flowKey)
				itemR2 := typesmocks.NewMockQueueItemAccessor(13, "r2_matchAll", flowKey)
				q.Add(itemR1)
				q.Add(itemR2)

				cleanedItems := q.Cleanup(func(item types.QueueItemAccessor) bool { return true })
				assert.Len(t, cleanedItems, 2, "Cleanup should return all items that matched the predicate")
				assert.Zero(t, q.Len(), "Len() should be 0 after Cleanup")
				assert.Zero(t, q.ByteSize(), "ByteSize() should be 0 after Cleanup")
				assert.True(t, itemR1.Handle().IsInvalidated(), "Handle for removed item 1 must be invalidated")
				assert.True(t, itemR2.Handle().IsInvalidated(), "Handle for removed item 2 must be invalidated")
			})

			t.Run("Cleanup_PredicateMatchesSubset_VerifyHandles", func(t *testing.T) {
				t.Parallel()
				q, _ := constructor(enqueueTimeComparator)
				iK1 := typesmocks.NewMockQueueItemAccessor(20, "k1_subset", flowKey)
				iR1 := typesmocks.NewMockQueueItemAccessor(11, "r1_subset", flowKey)
				iK2 := typesmocks.NewMockQueueItemAccessor(22, "k2_subset", flowKey)
				iR2 := typesmocks.NewMockQueueItemAccessor(33, "r2_subset", flowKey)
				q.Add(iK1)
				q.Add(iR1)
				q.Add(iK2)
				q.Add(iR2)

				expectedKeptByteSize := iK1.OriginalRequest().ByteSize() + iK2.OriginalRequest().ByteSize()

				cleanedItems := q.Cleanup(predicateRemoveOddSizes)
				assert.Len(t, cleanedItems, 2, "Cleanup should return 2 items that matched the predicate")
				assert.Equal(t, 2, q.Len(), "Len() should be 2 after Cleanup")
				assert.Equal(t, expectedKeptByteSize, q.ByteSize(), "ByteSize() should be sum of kept items after Cleanup")

				foundR1, foundR2 := false, false
				for _, item := range cleanedItems {
					if item.OriginalRequest().ID() == iR1.OriginalRequest().ID() {
						foundR1 = true
						assert.True(t, iR1.Handle().IsInvalidated(), "Handle for removed item iR1 must be invalidated")
					}
					if item.OriginalRequest().ID() == iR2.OriginalRequest().ID() {
						foundR2 = true
						assert.True(t, iR2.Handle().IsInvalidated(), "Handle for removed item iR2 must be invalidated")
					}
				}
				assert.True(t, foundR1, "iR1 should have been returned by Cleanup")
				assert.True(t, foundR2, "iR2 should have been returned by Cleanup")

				assert.False(t, iK1.Handle().IsInvalidated(), "Handle for kept item iK1 must NOT be invalidated")
				assert.False(t, iK2.Handle().IsInvalidated(), "Handle for kept item iK2 must NOT be invalidated")

				// Verify remaining items are correct
				var remainingIDs []string
				for q.Len() > 0 {
					peeked := q.PeekHead()
					item, _ := q.Remove(peeked.Handle())
					remainingIDs = append(remainingIDs, item.OriginalRequest().ID())
				}
				sort.Strings(remainingIDs) // Sort for stable comparison
				expectedRemainingIDs := []string{iK1.OriginalRequest().ID(), iK2.OriginalRequest().ID()}
				sort.Strings(expectedRemainingIDs)
				assert.Equal(t, expectedRemainingIDs, remainingIDs, "Remaining items in queue are not as expected")
			})

			t.Run("Drain_NonEmptyQueue_VerifyHandles", func(t *testing.T) {
				t.Parallel()
				q, err := constructor(enqueueTimeComparator)
				require.NoError(t, err, "Setup: creating queue for drain test should not fail")

				itemD1 := typesmocks.NewMockQueueItemAccessor(10, "ditem1", flowKey)
				itemD2 := typesmocks.NewMockQueueItemAccessor(20, "ditem2", flowKey)
				q.Add(itemD1)
				q.Add(itemD2)

				drainedItems := q.Drain()
				assert.Len(t, drainedItems, 2, "Drain should return all items that were in the queue")
				assert.Zero(t, q.Len(), "Queue length must be 0 after Drain")
				assert.Zero(t, q.ByteSize(), "Queue byte size must be 0 after Drain")

				assert.True(t, itemD1.Handle().IsInvalidated(), "Handle for drained itemD1 must be invalidated")
				assert.True(t, itemD2.Handle().IsInvalidated(), "Handle for drained itemD2 must be invalidated")

				var foundD1, foundD2 bool
				for _, item := range drainedItems {
					if item.OriginalRequest().ID() == itemD1.OriginalRequest().ID() {
						foundD1 = true
					}
					if item.OriginalRequest().ID() == itemD2.OriginalRequest().ID() {
						foundD2 = true
					}
				}
				assert.True(t, foundD1, "itemD1 should be in drainedItems")
				assert.True(t, foundD2, "itemD2 should be in drainedItems")
			})

			t.Run("Drain_EmptyQueue_DrainTwice", func(t *testing.T) {
				t.Parallel()
				q, err := constructor(enqueueTimeComparator)
				require.NoError(t, err, "Setup: creating queue for empty drain test should not fail")

				drainedItems := q.Drain() // First drain on empty
				assert.Empty(t, drainedItems, "Drain on an empty queue should return an empty slice")

				drainedAgain := q.Drain() // Second drain on already empty
				assert.Empty(t, drainedAgain, "Second drain on an already empty queue should return an empty slice")
				assert.Zero(t, q.Len())
				assert.Zero(t, q.ByteSize())
			})

			t.Run("Concurrency", func(t *testing.T) {
				t.Parallel()
				q, err := constructor(enqueueTimeComparator)
				require.NoError(t, err, "Setup: creating queue for concurrency test should not fail")

				const (
					numGoroutines   = 10
					initialItems    = 200
					opsPerGoroutine = 50
				)

				// handleChan acts as a concurrent-safe pool of handles that goroutines can pull from to test Remove.
				handleChan := make(chan types.QueueItemHandle, initialItems+(numGoroutines*opsPerGoroutine))

				// Pre-populate the queue with an initial set of items.
				for i := range initialItems {
					item := typesmocks.NewMockQueueItemAccessor(1, fmt.Sprintf("%s_conc_init_%d", flowKey, i), flowKey)
					q.Add(item)
					require.NoError(t, err, "Setup: pre-populating the queue should not fail")
					handleChan <- item.Handle()
				}

				var wg sync.WaitGroup
				wg.Add(numGoroutines)
				var successfulAdds, successfulRemoves atomic.Uint64

				// Start goroutines to perform a mix of concurrent operations.
				for i := range numGoroutines {
					go func(routineID int) {
						defer wg.Done()
						for j := range opsPerGoroutine {
							opType := (j + routineID) % 4 // Vary operations more across goroutines
							switch opType {
							case 0: // Add
								item := typesmocks.NewMockQueueItemAccessor(1,
									fmt.Sprintf("%s_conc_init_%d_%d", flowKey, routineID, j), flowKey)
								q.Add(item)
								successfulAdds.Add(1)
								handleChan <- item.Handle()
							case 1: // Remove
								select {
								case handle := <-handleChan:
									if handle != nil && !handle.IsInvalidated() { // Check before trying to prevent known-to-fail calls
										_, removeErr := q.Remove(handle)
										if removeErr == nil {
											successfulRemoves.Add(1)
										} else {
											// It's okay if it's ErrInvalidQueueItemHandle or ErrQueueItemNotFound due to races
											assert.ErrorIs(t, removeErr, framework.ErrInvalidQueueItemHandle,
												"Expected invalid handle or not found if raced")
										}
									}
								default:
									// No handles available to remove
								}
							case 2: // Inspect
								_ = q.Len()
								_ = q.ByteSize()
								peeked := q.PeekHead()
								if q.Len() == 0 {
									assert.Nil(t, peeked, "PeekHead on empty queue expected nil")
								}
								peeked = q.PeekTail()
								if q.Len() == 0 {
									assert.Nil(t, peeked, "PeekTail on empty queue expected nil")
								}
							case 3: // Cleanup
								q.Cleanup(func(item types.QueueItemAccessor) bool { return false })
							}
						}
					}(i)
				}

				wg.Wait()
				close(handleChan)

				// Drain the queue to verify all handles are invalidated and to count remaining items accurately.
				drainedItems := q.Drain()

				for _, item := range drainedItems {
					require.True(t, item.Handle().IsInvalidated(), "All handles from final drain must be invalidated")
				}

				// The number of items successfully added minus those successfully removed should equal the number of items
				// drained.
				assert.Equal(t, int(initialItems)+int(successfulAdds.Load())-int(successfulRemoves.Load()), len(drainedItems),
					"Number of items drained (%d) should match initial (%d) + successful adds (%d) - successful removes (%d).",
					len(drainedItems), initialItems, successfulAdds.Load(), successfulRemoves.Load())

				assert.Zero(t, q.Len(), "Queue length should be 0 after final drain")
				assert.Zero(t, q.ByteSize(), "Queue byte size should be 0 after final drain")
			})
		})
	}
}
