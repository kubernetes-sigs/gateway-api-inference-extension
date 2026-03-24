/*
Copyright 2026 The Kubernetes Authors.

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

package eviction

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// --- Test policy implementations ---

type testOrdering struct{}

func (t *testOrdering) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "test-ordering", Name: "test"}
}

func (t *testOrdering) Less(a, b *flowcontrol.EvictionItem) bool {
	if a.Priority != b.Priority {
		return a.Priority < b.Priority
	}
	return a.DispatchTime.After(b.DispatchTime)
}

type testFilter struct {
	threshold int
}

func (t *testFilter) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "test-filter", Name: "test"}
}

func (t *testFilter) Accept(item *flowcontrol.EvictionItem) bool {
	return item.Priority < t.threshold
}

type acceptAllFilter struct{}

func (a *acceptAllFilter) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "accept-all", Name: "test"}
}

func (a *acceptAllFilter) Accept(_ *flowcontrol.EvictionItem) bool { return true }

// --- Helpers ---

func newItem(id string, priority int, dispatchOffset time.Duration) *flowcontrol.EvictionItem {
	return &flowcontrol.EvictionItem{
		RequestID:    id,
		Priority:     priority,
		DispatchTime: time.Now().Add(dispatchOffset),
	}
}

// --- Tests ---

func TestEvictionQueue_TrackAndUntrack(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &acceptAllFilter{})

	item := newItem("req-1", -1, 0)
	q.Track(item)

	assert.Equal(t, 1, q.InFlightLen())
	assert.Equal(t, 1, q.EvictableLen())

	q.Untrack("req-1")
	assert.Equal(t, 0, q.InFlightLen())
	assert.Equal(t, 0, q.EvictableLen())

	// PopN also cleans up both maps.
	q.Track(newItem("req-2", -1, 0))
	q.Track(newItem("req-3", -2, 0))
	evicted := q.PopN(1)
	require.Len(t, evicted, 1)
	assert.Equal(t, 1, q.EvictableLen(), "One item should remain in heap after PopN(1)")
	assert.Equal(t, 1, q.InFlightLen(), "PopN should remove from allInFlight too")

	// Re-untracking a popped item is a no-op.
	q.Untrack(evicted[0].RequestID)
	assert.Equal(t, 1, q.InFlightLen(), "Untracking already-popped item should be a no-op")
}

func TestEvictionQueue_FilterRejectsHighPriority(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &testFilter{threshold: 0})

	sheddable := newItem("sheddable", -1, 0)
	normal := newItem("normal", 5, 0)

	q.Track(sheddable)
	q.Track(normal)

	assert.Equal(t, 2, q.InFlightLen(), "Both should be tracked as in-flight")
	assert.Equal(t, 1, q.EvictableLen(), "Only sheddable should be in the eviction heap")

	// Pop should return only the sheddable one.
	evicted := q.PopN(5)
	require.Len(t, evicted, 1)
	assert.Equal(t, "sheddable", evicted[0].RequestID)
}

func TestEvictionQueue_PopN_OrderByPriority(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &acceptAllFilter{})

	q.Track(newItem("p5", 5, 0))
	q.Track(newItem("p1", 1, 0))
	q.Track(newItem("p3", 3, 0))
	q.Track(newItem("p-1", -1, 0))
	q.Track(newItem("p0", 0, 0))

	evicted := q.PopN(3)
	require.Len(t, evicted, 3)

	// Should be in ascending priority order (most evictable first).
	assert.Equal(t, "p-1", evicted[0].RequestID)
	assert.Equal(t, "p0", evicted[1].RequestID)
	assert.Equal(t, "p1", evicted[2].RequestID)

	assert.Equal(t, 2, q.EvictableLen(), "Two items should remain in the heap")
	assert.Equal(t, 2, q.InFlightLen(), "Two items should remain in-flight")
}

func TestEvictionQueue_PopN_TiebreakByDispatchTime(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &acceptAllFilter{})

	q.Track(newItem("newer", 0, 100*time.Millisecond))
	q.Track(newItem("oldest", 0, -100*time.Millisecond))
	q.Track(newItem("middle", 0, 0))

	evicted := q.PopN(3)
	require.Len(t, evicted, 3)

	assert.Equal(t, "newer", evicted[0].RequestID)
	assert.Equal(t, "middle", evicted[1].RequestID)
	assert.Equal(t, "oldest", evicted[2].RequestID)
}

func TestEvictionQueue_PopN_Bounds(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &acceptAllFilter{})

	// PopN on empty queue.
	assert.Empty(t, q.PopN(5), "PopN on empty queue should return empty slice")

	// PopN with n > heap size.
	q.Track(newItem("req-1", 0, 0))
	q.Track(newItem("req-2", 0, time.Millisecond))
	evicted := q.PopN(10)
	assert.Len(t, evicted, 2, "PopN should return all items when n > heap size")
	assert.Equal(t, 0, q.EvictableLen())
	assert.Equal(t, 0, q.InFlightLen())
}

func TestEvictionQueue_UntrackNonExistent(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &acceptAllFilter{})

	q.Track(newItem("req-1", -1, 0))

	// Untrack a non-existent ID should not affect existing state.
	q.Untrack("does-not-exist")
	assert.Equal(t, 1, q.InFlightLen(), "Existing items should be unaffected")
	assert.Equal(t, 1, q.EvictableLen(), "Existing items should be unaffected")
}

func TestEvictionQueue_UntrackRemovesFromHeap(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &acceptAllFilter{})

	q.Track(newItem("req-1", -1, 0))
	q.Track(newItem("req-2", -2, 0))
	q.Track(newItem("req-3", -3, 0))

	// Untrack the most evictable one before eviction.
	q.Untrack("req-3")

	evicted := q.PopN(1)
	require.Len(t, evicted, 1)
	assert.Equal(t, "req-2", evicted[0].RequestID, "After untracking req-3, req-2 should be most evictable")
}

func TestEvictionQueue_DuplicateTrack(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &acceptAllFilter{})

	// Track same ID twice — should replace, not double-add.
	q.Track(newItem("req-1", -1, 0))
	q.Track(newItem("req-1", -1, 0))

	assert.Equal(t, 1, q.InFlightLen(), "Duplicate track should not increase in-flight count")
	assert.Equal(t, 1, q.EvictableLen(), "Duplicate track should not increase evictable count")

	evicted := q.PopN(5)
	require.Len(t, evicted, 1, "Should only pop one item after duplicate track")

	// Re-track with different priority — heap should reflect the new priority.
	q.Track(newItem("req-a", 5, 0))
	q.Track(newItem("req-a", -10, 0)) // re-track with lower priority

	q.Track(newItem("req-b", -5, 0))

	evicted = q.PopN(1)
	require.Len(t, evicted, 1)
	assert.Equal(t, "req-a", evicted[0].RequestID, "Re-tracked req-a with priority -10 should be popped before req-b with priority -5")
	assert.Equal(t, -10, evicted[0].Priority, "Re-tracked item should have the updated priority")
}

func TestEvictionQueue_Peek(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &acceptAllFilter{})

	// Peek on empty queue.
	assert.Nil(t, q.Peek(), "Peek on empty queue should return nil")

	q.Track(newItem("high", 5, 0))
	q.Track(newItem("low", -1, 0))
	q.Track(newItem("mid", 2, 0))

	peeked := q.Peek()
	require.NotNil(t, peeked)
	assert.Equal(t, "low", peeked.RequestID, "Peek should return the most-evictable item")

	// Peek should not remove the item.
	assert.Equal(t, 3, q.EvictableLen(), "Peek should not change evictable count")

	// Mutating the copy should not corrupt the heap.
	peeked.Priority = 999
	peeked2 := q.Peek()
	assert.Equal(t, -1, peeked2.Priority, "Mutating Peek result should not affect the heap")
}

func TestEvictionQueue_UntrackNonEvictable(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &testFilter{threshold: 0})

	// Track a non-evictable request (priority >= 0).
	q.Track(newItem("normal", 5, 0))
	assert.Equal(t, 1, q.InFlightLen())
	assert.Equal(t, 0, q.EvictableLen(), "Non-evictable request should not be in the heap")

	// Untrack should clean up allInFlight without touching the heap.
	q.Untrack("normal")
	assert.Equal(t, 0, q.InFlightLen())
	assert.Equal(t, 0, q.EvictableLen())
}

func TestEvictionQueue_MixedOperations(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &acceptAllFilter{})

	// Track 7 items with distinct priorities.
	for _, p := range []int{3, 7, 1, 5, 2, 6, 4} {
		q.Track(newItem(fmt.Sprintf("req-%d", p), p, 0))
	}
	assert.Equal(t, 7, q.EvictableLen())

	// Untrack 3 from the middle (priorities 3, 5, 6).
	q.Untrack("req-3")
	q.Untrack("req-5")
	q.Untrack("req-6")
	assert.Equal(t, 4, q.EvictableLen())

	// Pop 2 — should get the two lowest remaining priorities (1, 2).
	evicted := q.PopN(2)
	require.Len(t, evicted, 2)
	assert.Equal(t, 1, evicted[0].Priority)
	assert.Equal(t, 2, evicted[1].Priority)

	// Remaining 2 should come out in order (4, 7).
	evicted = q.PopN(5)
	require.Len(t, evicted, 2)
	assert.Equal(t, 4, evicted[0].Priority)
	assert.Equal(t, 7, evicted[1].Priority)

	assert.Equal(t, 0, q.EvictableLen())
	assert.Equal(t, 0, q.InFlightLen())
}

func TestEvictionQueue_RetrackAfterPop(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &acceptAllFilter{})

	q.Track(newItem("req-1", -1, 0))
	evicted := q.PopN(1)
	require.Len(t, evicted, 1)

	// Re-track the same requestID after it was popped.
	q.Track(newItem("req-1", -2, 0))
	assert.Equal(t, 1, q.InFlightLen())
	assert.Equal(t, 1, q.EvictableLen())

	evicted = q.PopN(1)
	require.Len(t, evicted, 1)
	assert.Equal(t, "req-1", evicted[0].RequestID)
	assert.Equal(t, -2, evicted[0].Priority, "Re-tracked item should have the new priority")
}

func TestEvictionQueue_Concurrency(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &acceptAllFilter{})

	const goroutines = 10
	const opsPerGoroutine = 100

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := range goroutines {
		go func(id int) {
			defer wg.Done()
			for i := range opsPerGoroutine {
				reqID := fmt.Sprintf("req-%d-%d", id, i)
				item := newItem(reqID, id, time.Duration(i)*time.Millisecond)

				switch i % 4 {
				case 0:
					q.Track(item)
				case 1:
					q.Track(item)
					q.Untrack(reqID)
				case 2:
					q.PopN(1)
				case 3:
					q.EvictableLen()
					q.InFlightLen()
					q.Peek()
				}
			}
		}(g)
	}

	wg.Wait()

	// Verify invariants after concurrent operations.
	inFlight := q.InFlightLen()
	evictable := q.EvictableLen()
	assert.GreaterOrEqual(t, inFlight, 0)
	assert.GreaterOrEqual(t, evictable, 0)
	assert.GreaterOrEqual(t, inFlight, evictable,
		"In-flight count should always be >= evictable count")

	// Drain remaining items and verify they come out in order.
	remaining := q.PopN(inFlight + 1)
	for i := 1; i < len(remaining); i++ {
		assert.LessOrEqual(t, remaining[i-1].Priority, remaining[i].Priority,
			"Remaining items should pop in priority order")
	}
	assert.Equal(t, 0, q.EvictableLen(), "Queue should be empty after draining")
}
