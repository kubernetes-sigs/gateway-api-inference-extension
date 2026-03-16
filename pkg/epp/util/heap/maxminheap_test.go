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

package heap

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// intItem is a trivial item type for testing.
type intItem struct {
	value int
}

// intLess orders by value ascending: higher value = higher priority = head.
func intLess(a, b intItem) bool {
	return a.value > b.value
}

func TestMaxMinHeap_AddAndPeek(t *testing.T) {
	t.Parallel()
	h := New(intLess)

	// Empty heap.
	_, ok := h.PeekMax()
	assert.False(t, ok, "PeekMax on empty heap should return false")
	_, ok = h.PeekMin()
	assert.False(t, ok, "PeekMin on empty heap should return false")

	h.Add(intItem{value: 5})
	h.Add(intItem{value: 1})
	h.Add(intItem{value: 9})
	h.Add(intItem{value: 3})
	h.Add(intItem{value: 7})

	assert.Equal(t, 5, h.Len())

	max, ok := h.PeekMax()
	require.True(t, ok)
	assert.Equal(t, 9, max.value, "PeekMax should return the item with the highest value")

	min, ok := h.PeekMin()
	require.True(t, ok)
	assert.Equal(t, 1, min.value, "PeekMin should return the item with the lowest value")
}

func TestMaxMinHeap_Remove(t *testing.T) {
	t.Parallel()
	h := New(intLess)

	h1 := h.Add(intItem{value: 10})
	h2 := h.Add(intItem{value: 20})
	h3 := h.Add(intItem{value: 30})

	// Remove the middle item.
	removed, ok := h.Remove(h2)
	require.True(t, ok)
	assert.Equal(t, 20, removed.value)
	assert.True(t, h2.IsInvalidated())
	assert.Equal(t, 2, h.Len())

	// Double-remove should fail.
	_, ok = h.Remove(h2)
	assert.False(t, ok, "Removing an invalidated handle should return false")

	// Remove nil handle.
	_, ok = h.Remove(nil)
	assert.False(t, ok, "Removing nil handle should return false")

	// Verify max/min after removal.
	max, _ := h.PeekMax()
	assert.Equal(t, 30, max.value)
	min, _ := h.PeekMin()
	assert.Equal(t, 10, min.value)

	// Remove remaining.
	h.Remove(h1)
	h.Remove(h3)
	assert.Equal(t, 0, h.Len())
}

func TestMaxMinHeap_RemoveHead(t *testing.T) {
	t.Parallel()
	h := New(intLess)

	handles := make([]*Handle, 5)
	for i, v := range []int{3, 1, 4, 1, 5} {
		handles[i] = h.Add(intItem{value: v})
	}

	// Remove from head repeatedly — should yield items in descending order.
	expected := []int{5, 4, 3, 1, 1}
	for i, exp := range expected {
		max, ok := h.PeekMax()
		require.True(t, ok, "PeekMax should succeed at iteration %d", i)
		assert.Equal(t, exp, max.value)

		// Find the handle for this max item by checking which handle's index is 0.
		// Since PeekMax is the root, its handle index is 0.
		removed, ok := h.Remove(h.entries[0].handle)
		require.True(t, ok)
		assert.Equal(t, exp, removed.value)
	}

	assert.Equal(t, 0, h.Len())
}

func TestMaxMinHeap_RemoveTail(t *testing.T) {
	t.Parallel()
	h := New(intLess)

	for _, v := range []int{3, 1, 4, 1, 5} {
		h.Add(intItem{value: v})
	}

	// Remove from tail repeatedly — should yield items in ascending order.
	expected := []int{1, 1, 3, 4, 5}
	for i, exp := range expected {
		min, ok := h.PeekMin()
		require.True(t, ok, "PeekMin should succeed at iteration %d", i)
		assert.Equal(t, exp, min.value)

		// Find the min entry's handle.
		minIdx := h.findMinIndex()
		removed, ok := h.Remove(h.entries[minIdx].handle)
		require.True(t, ok)
		assert.Equal(t, exp, removed.value)
	}

	assert.Equal(t, 0, h.Len())
}

// findMinIndex returns the index of the minimum element (helper for tests).
func (h *MaxMinHeap[T]) findMinIndex() int {
	n := len(h.entries)
	if n <= 1 {
		return 0
	}
	if n == 2 {
		return 1
	}
	if h.less(h.entries[1].item, h.entries[2].item) {
		return 2
	}
	return 1
}

func TestMaxMinHeap_Cleanup(t *testing.T) {
	t.Parallel()
	h := New(intLess)

	for _, v := range []int{1, 2, 3, 4, 5, 6} {
		h.Add(intItem{value: v})
	}

	// Remove odd values.
	removed := h.Cleanup(func(item intItem) bool {
		return item.value%2 != 0
	})

	assert.Len(t, removed, 3, "Should remove 3 odd items")
	assert.Equal(t, 3, h.Len(), "Should have 3 even items remaining")

	max, _ := h.PeekMax()
	assert.Equal(t, 6, max.value)
	min, _ := h.PeekMin()
	assert.Equal(t, 2, min.value)

	// Verify heap property after cleanup.
	assertHeapProperty(t, h)
}

func TestMaxMinHeap_Drain(t *testing.T) {
	t.Parallel()
	h := New(intLess)

	var handles []*Handle
	for _, v := range []int{10, 20, 30} {
		handles = append(handles, h.Add(intItem{value: v}))
	}

	drained := h.Drain()
	assert.Len(t, drained, 3)
	assert.Equal(t, 0, h.Len())

	// All handles should be invalidated.
	for _, handle := range handles {
		assert.True(t, handle.IsInvalidated())
	}

	// Drain on empty should return empty.
	drained = h.Drain()
	assert.Empty(t, drained)
}

func TestMaxMinHeap_HeapProperty(t *testing.T) {
	t.Parallel()
	h := New(intLess)

	// Add 20 items in a scattered order and verify property after each.
	values := []int{15, 3, 18, 7, 12, 1, 20, 9, 5, 14, 2, 17, 6, 11, 4, 19, 8, 13, 16, 10}
	handles := make([]*Handle, len(values))
	for i, v := range values {
		handles[i] = h.Add(intItem{value: v})
		assertHeapProperty(t, h)
	}

	// Remove a few from the middle.
	for _, i := range []int{15, 7, 11} {
		_, ok := h.Remove(handles[i])
		require.True(t, ok, "Remove should succeed for item at index %d", i)
		assertHeapProperty(t, h)
	}

	// Remove all from head.
	for h.Len() > 0 {
		_, ok := h.Remove(h.entries[0].handle)
		require.True(t, ok)
		assertHeapProperty(t, h)
	}
}

func TestMaxMinHeap_Concurrency(t *testing.T) {
	t.Parallel()
	h := New(intLess)

	const (
		numGoroutines   = 10
		initialItems    = 200
		opsPerGoroutine = 50
	)

	handleChan := make(chan *Handle, initialItems+(numGoroutines*opsPerGoroutine))

	for i := range initialItems {
		handle := h.Add(intItem{value: i})
		handleChan <- handle
	}

	var wg sync.WaitGroup
	wg.Add(numGoroutines)
	var successfulAdds, successfulRemoves atomic.Uint64

	for i := range numGoroutines {
		go func(routineID int) {
			defer wg.Done()
			for j := range opsPerGoroutine {
				switch (j + routineID) % 4 {
				case 0: // Add
					handle := h.Add(intItem{value: routineID*1000 + j})
					successfulAdds.Add(1)
					handleChan <- handle
				case 1: // Remove
					select {
					case handle := <-handleChan:
						if handle != nil && !handle.IsInvalidated() {
							if _, ok := h.Remove(handle); ok {
								successfulRemoves.Add(1)
							}
						}
					default:
					}
				case 2: // PeekMax/PeekMin
					h.PeekMax()
					h.PeekMin()
				case 3: // Len
					h.Len()
				}
			}
		}(i)
	}

	wg.Wait()
	close(handleChan)

	drained := h.Drain()
	assert.Equal(t,
		int(initialItems)+int(successfulAdds.Load())-int(successfulRemoves.Load()),
		len(drained),
		fmt.Sprintf("drained=%d, initial=%d, adds=%d, removes=%d",
			len(drained), initialItems, successfulAdds.Load(), successfulRemoves.Load()),
	)
	assert.Equal(t, 0, h.Len())
}

func TestMaxMinHeap_SingleItem(t *testing.T) {
	t.Parallel()
	h := New(intLess)

	handle := h.Add(intItem{value: 42})

	max, ok := h.PeekMax()
	require.True(t, ok)
	assert.Equal(t, 42, max.value)

	min, ok := h.PeekMin()
	require.True(t, ok)
	assert.Equal(t, 42, min.value, "Single item should be both max and min")

	removed, ok := h.Remove(handle)
	require.True(t, ok)
	assert.Equal(t, 42, removed.value)
	assert.Equal(t, 0, h.Len())
}

func TestMaxMinHeap_TwoItems(t *testing.T) {
	t.Parallel()
	h := New(intLess)

	h.Add(intItem{value: 10})
	h.Add(intItem{value: 20})

	max, _ := h.PeekMax()
	assert.Equal(t, 20, max.value)

	min, _ := h.PeekMin()
	assert.Equal(t, 10, min.value)
}

func TestMaxMinHeap_DuplicateValues(t *testing.T) {
	t.Parallel()
	h := New(intLess)

	h1 := h.Add(intItem{value: 5})
	h2 := h.Add(intItem{value: 5})
	h3 := h.Add(intItem{value: 5})

	assert.Equal(t, 3, h.Len())

	// Remove one — the other two should remain.
	h.Remove(h2)
	assert.Equal(t, 2, h.Len())

	max, _ := h.PeekMax()
	assert.Equal(t, 5, max.value)
	min, _ := h.PeekMin()
	assert.Equal(t, 5, min.value)

	h.Remove(h1)
	h.Remove(h3)
	assert.Equal(t, 0, h.Len())
}

// assertHeapProperty validates the max-min heap invariant on the entire heap.
func assertHeapProperty[T any](t *testing.T, h *MaxMinHeap[T]) {
	t.Helper()
	h.mu.RLock()
	defer h.mu.RUnlock()

	n := len(h.entries)
	for i := 0; i < n; i++ {
		level := int(math.Floor(math.Log2(float64(i + 1))))
		isMin := level%2 != 0

		leftChild := 2*i + 1
		rightChild := 2*i + 2

		if leftChild < n {
			if isMin {
				assert.False(t, h.less(h.entries[i].item, h.entries[leftChild].item),
					"min-level node %d has child %d with smaller value", i, leftChild)
			} else {
				assert.False(t, h.less(h.entries[leftChild].item, h.entries[i].item),
					"max-level node %d has child %d with larger value", i, leftChild)
			}
		}
		if rightChild < n {
			if isMin {
				assert.False(t, h.less(h.entries[i].item, h.entries[rightChild].item),
					"min-level node %d has child %d with smaller value", i, rightChild)
			} else {
				assert.False(t, h.less(h.entries[rightChild].item, h.entries[i].item),
					"max-level node %d has child %d with larger value", i, rightChild)
			}
		}
	}
}
