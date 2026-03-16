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

// Package heap provides a generic, concurrent-safe max-min heap implementation.
//
// A max-min heap is a binary tree structure that maintains a specific ordering property: for any node, if it is at an
// even level (e.g., 0, 2, ...), its value is greater than all values in its subtree (max level). If it is at an odd
// level (e.g., 1, 3, ...), its value is smaller than all values in its subtree (min level). This structure allows for
// efficient O(1) retrieval of both the maximum and minimum priority items.
//
// The core heap maintenance logic (up, down, and grandchild finding) is adapted from the public domain implementation
// at https://github.com/esote/minmaxheap, which is licensed under CC0-1.0.
package heap

import (
	"math"
	"sync"
)

// Handle is an opaque reference to an item in the heap.
// It enables O(log n) removal of arbitrary items by tracking the item's position in the backing array.
// A Handle is bound to the heap instance that created it and MUST NOT be used with a different heap.
type Handle struct {
	index       int
	invalidated bool
}

// IsInvalidated returns true if this handle has been invalidated (the item was removed from the heap).
func (h *Handle) IsInvalidated() bool { return h.invalidated }

// Invalidate marks this handle as no longer valid.
func (h *Handle) Invalidate() { h.invalidated = true }

// LessFunc reports whether item a has higher priority than item b.
// Returning true means a should be closer to the head (max) of the heap.
type LessFunc[T any] func(a, b T) bool

// PredicateFunc returns true if the given item matches a condition.
type PredicateFunc[T any] func(item T) bool

// entry pairs an item with its handle for internal bookkeeping.
type entry[T any] struct {
	item   T
	handle *Handle
}

// MaxMinHeap is a generic, concurrent-safe max-min heap.
//
// It provides O(1) access to both the maximum and minimum elements, and O(log n) insertion and arbitrary removal.
// The ordering is determined by the LessFunc provided at construction time.
//
// All exported methods are goroutine-safe.
type MaxMinHeap[T any] struct {
	entries []*entry[T]
	mu      sync.RWMutex
	less    LessFunc[T]
}

// New creates a new MaxMinHeap ordered by the given comparator.
// less(a, b) should return true if a has higher priority than b (a is "greater").
func New[T any](less LessFunc[T]) *MaxMinHeap[T] {
	return &MaxMinHeap[T]{
		entries: make([]*entry[T], 0),
		less:    less,
	}
}

// Len returns the number of items in the heap.
func (h *MaxMinHeap[T]) Len() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.entries)
}

// Add inserts an item into the heap and returns a Handle that can be used for subsequent removal.
// Time complexity: O(log n).
func (h *MaxMinHeap[T]) Add(item T) *Handle {
	h.mu.Lock()
	defer h.mu.Unlock()

	e := &entry[T]{
		item:   item,
		handle: &Handle{index: len(h.entries)},
	}
	h.entries = append(h.entries, e)
	h.up(len(h.entries) - 1)
	return e.handle
}

// Remove removes the item identified by the given Handle from the heap.
// Returns the removed item and true on success, or the zero value and false if the handle is nil, invalidated,
// or out of range.
// Time complexity: O(log n).
func (h *MaxMinHeap[T]) Remove(handle *Handle) (T, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()

	var zero T
	if handle == nil || handle.invalidated {
		return zero, false
	}

	i := handle.index
	if i < 0 || i >= len(h.entries) {
		return zero, false
	}

	// Verify this handle actually belongs to the entry at this index (guards against stale/alien handles).
	if h.entries[i].handle != handle {
		return zero, false
	}

	item := h.entries[i].item
	n := len(h.entries) - 1

	if i < n {
		h.swap(i, n)
		h.entries = h.entries[:n]
		// The swapped-in element may need to move in either direction.
		h.down(i)
		h.up(i)
	} else {
		h.entries = h.entries[:n]
	}

	handle.Invalidate()
	return item, true
}

// PeekMax returns the item with the highest priority without removing it.
// Returns the zero value and false if the heap is empty.
// Time complexity: O(1).
func (h *MaxMinHeap[T]) PeekMax() (T, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	var zero T
	if len(h.entries) == 0 {
		return zero, false
	}
	return h.entries[0].item, true
}

// PeekMin returns the item with the lowest priority without removing it.
// Returns the zero value and false if the heap is empty.
// Time complexity: O(1).
func (h *MaxMinHeap[T]) PeekMin() (T, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	n := len(h.entries)
	var zero T
	if n == 0 {
		return zero, false
	}
	if n == 1 {
		return h.entries[0].item, true
	}
	if n == 2 {
		return h.entries[1].item, true
	}

	// With three or more items, the minimum is one of the two children of the root.
	if h.less(h.entries[1].item, h.entries[2].item) {
		return h.entries[2].item, true
	}
	return h.entries[1].item, true
}

// Cleanup removes all items for which the predicate returns true, returning them in a slice.
// Handles for removed items are invalidated.
// Time complexity: O(n).
func (h *MaxMinHeap[T]) Cleanup(predicate PredicateFunc[T]) []T {
	h.mu.Lock()
	defer h.mu.Unlock()

	var removed []T
	var kept []*entry[T]

	for _, e := range h.entries {
		if predicate(e.item) {
			removed = append(removed, e.item)
			e.handle.Invalidate()
		} else {
			kept = append(kept, e)
		}
	}

	if len(removed) > 0 {
		h.entries = kept
		// Update indices.
		for i, e := range h.entries {
			e.handle.index = i
		}
		// Re-establish the heap property via heapify.
		for i := len(h.entries)/2 - 1; i >= 0; i-- {
			h.down(i)
		}
	}

	return removed
}

// Drain removes and returns all items. Handles are invalidated.
func (h *MaxMinHeap[T]) Drain() []T {
	h.mu.Lock()
	defer h.mu.Unlock()

	result := make([]T, len(h.entries))
	for i, e := range h.entries {
		result[i] = e.item
		e.handle.Invalidate()
	}
	h.entries = h.entries[:0]
	return result
}

// --- Internal heap operations ---

func (h *MaxMinHeap[T]) swap(i, j int) {
	h.entries[i], h.entries[j] = h.entries[j], h.entries[i]
	h.entries[i].handle.index = i
	h.entries[j].handle.index = j
}

func (h *MaxMinHeap[T]) up(i int) {
	if i == 0 {
		return
	}

	parentIndex := (i - 1) / 2
	if isMinLevel(i) {
		if h.less(h.entries[i].item, h.entries[parentIndex].item) {
			h.swap(i, parentIndex)
			h.upMax(parentIndex)
		} else {
			h.upMin(i)
		}
	} else {
		if h.less(h.entries[parentIndex].item, h.entries[i].item) {
			h.swap(i, parentIndex)
			h.upMin(parentIndex)
		} else {
			h.upMax(i)
		}
	}
}

func (h *MaxMinHeap[T]) upMin(i int) {
	for {
		parentIndex := (i - 1) / 2
		if parentIndex == 0 {
			break
		}
		grandparentIndex := (parentIndex - 1) / 2
		if h.less(h.entries[grandparentIndex].item, h.entries[i].item) {
			h.swap(i, grandparentIndex)
			i = grandparentIndex
		} else {
			break
		}
	}
}

func (h *MaxMinHeap[T]) upMax(i int) {
	for {
		parentIndex := (i - 1) / 2
		if parentIndex == 0 {
			break
		}
		grandparentIndex := (parentIndex - 1) / 2
		if h.less(h.entries[i].item, h.entries[grandparentIndex].item) {
			h.swap(i, grandparentIndex)
			i = grandparentIndex
		} else {
			break
		}
	}
}

func (h *MaxMinHeap[T]) down(i int) {
	if isMinLevel(i) {
		h.downMin(i)
	} else {
		h.downMax(i)
	}
}

func (h *MaxMinHeap[T]) downMin(i int) {
	for {
		m := h.findSmallestChildOrGrandchild(i)
		if m == -1 {
			break
		}

		if h.less(h.entries[i].item, h.entries[m].item) {
			h.swap(i, m)
			parentOfM := (m - 1) / 2
			if parentOfM != i {
				if h.less(h.entries[m].item, h.entries[parentOfM].item) {
					h.swap(m, parentOfM)
				}
			}
			i = m
		} else {
			break
		}
	}
}

func (h *MaxMinHeap[T]) downMax(i int) {
	for {
		m := h.findLargestChildOrGrandchild(i)
		if m == -1 {
			break
		}

		if h.less(h.entries[m].item, h.entries[i].item) {
			h.swap(i, m)
			parentOfM := (m - 1) / 2
			if parentOfM != i {
				if h.less(h.entries[parentOfM].item, h.entries[m].item) {
					h.swap(m, parentOfM)
				}
			}
			i = m
		} else {
			break
		}
	}
}

func (h *MaxMinHeap[T]) findSmallestChildOrGrandchild(i int) int {
	leftChild := 2*i + 1
	if leftChild >= len(h.entries) {
		return -1
	}

	m := leftChild

	rightChild := 2*i + 2
	if rightChild < len(h.entries) && h.less(h.entries[m].item, h.entries[rightChild].item) {
		m = rightChild
	}

	grandchildStart := 2*leftChild + 1
	grandchildEnd := grandchildStart + 4
	for j := grandchildStart; j < grandchildEnd && j < len(h.entries); j++ {
		if h.less(h.entries[m].item, h.entries[j].item) {
			m = j
		}
	}
	return m
}

func (h *MaxMinHeap[T]) findLargestChildOrGrandchild(i int) int {
	leftChild := 2*i + 1
	if leftChild >= len(h.entries) {
		return -1
	}

	m := leftChild

	rightChild := 2*i + 2
	if rightChild < len(h.entries) && h.less(h.entries[rightChild].item, h.entries[m].item) {
		m = rightChild
	}

	grandchildStart := 2*leftChild + 1
	grandchildEnd := grandchildStart + 4
	for j := grandchildStart; j < grandchildEnd && j < len(h.entries); j++ {
		if h.less(h.entries[j].item, h.entries[m].item) {
			m = j
		}
	}
	return m
}

// isMinLevel checks if the given index is on a min level of the heap.
func isMinLevel(i int) bool {
	level := int(math.Log2(float64(i + 1)))
	return level%2 != 0
}
