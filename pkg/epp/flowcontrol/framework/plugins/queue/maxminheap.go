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

// Package maxminheap provides a concurrent-safe, priority queue implementation using a max-min heap.
//
// A max-min heap is a binary tree structure that maintains a specific ordering property: for any node, if it is at an
// even level (e.g., 0, 2, ...), its value is greater than all values in its subtree (max level). If it is at an odd
// level (e.g., 1, 3, ...), its value is smaller than all values in its subtree (min level). This structure allows for
// efficient O(1) retrieval of both the maximum and minimum priority items.
//
// The core heap maintenance logic (up, down, and grandchild finding) is adapted from the public domain implementation
// at https://github.com/esote/minmaxheap, which is licensed under CC0-1.0.
package queue

import (
	"math"
	"sync"
	"sync/atomic"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// MaxMinHeapName is the name of the max-min heap queue implementation.
const MaxMinHeapName = "MaxMinHeap"

func init() {
	MustRegisterQueue(RegisteredQueueName(MaxMinHeapName),
		func(comparator framework.ItemComparator) (framework.SafeQueue, error) {
			return newMaxMinHeap(comparator), nil
		})
}

// maxMinHeap implements the `framework.SafeQueue` interface using a max-min heap.
// The heap is ordered by the provided comparator, with higher values considered higher priority.
// This implementation is concurrent-safe.
type maxMinHeap struct {
	items      []types.QueueItemAccessor
	handles    map[types.QueueItemHandle]*heapItem
	byteSize   atomic.Uint64
	mu         sync.RWMutex
	comparator framework.ItemComparator
}

// heapItem is an internal struct to hold an item and its index in the heap.
// This allows for O(log n) removal of items from the queue.
type heapItem struct {
	item          types.QueueItemAccessor
	index         int
	isInvalidated bool
}

// Handle returns the heap item itself, which is used as the handle.
func (h *heapItem) Handle() any {
	return h
}

// Invalidate marks the handle as invalid.
func (h *heapItem) Invalidate() {
	h.isInvalidated = true
}

// IsInvalidated returns true if the handle has been invalidated.
func (h *heapItem) IsInvalidated() bool {
	return h.isInvalidated
}

var _ types.QueueItemHandle = &heapItem{}

// newMaxMinHeap creates a new max-min heap with the given comparator.
func newMaxMinHeap(comparator framework.ItemComparator) *maxMinHeap {
	return &maxMinHeap{
		items:      make([]types.QueueItemAccessor, 0),
		handles:    make(map[types.QueueItemHandle]*heapItem),
		comparator: comparator,
	}
}

// --- `framework.SafeQueue` Interface Implementation ---

// Name returns the name of the queue.
func (h *maxMinHeap) Name() string {
	return MaxMinHeapName
}

// Capabilities returns the capabilities of the queue.
func (h *maxMinHeap) Capabilities() []framework.QueueCapability {
	return []framework.QueueCapability{framework.CapabilityPriorityConfigurable}
}

// Len returns the number of items in the queue.
func (h *maxMinHeap) Len() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.items)
}

// ByteSize returns the total byte size of all items in the queue.
func (h *maxMinHeap) ByteSize() uint64 {
	return h.byteSize.Load()
}

// PeekHead returns the item with the highest priority (max value) without removing it.
// Time complexity: O(1).
func (h *maxMinHeap) PeekHead() types.QueueItemAccessor {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if len(h.items) == 0 {
		return nil
	}
	// The root of the max-min heap is always the maximum element.
	return h.items[0]
}

// PeekTail returns the item with the lowest priority (min value) without removing it.
// Time complexity: O(1).
func (h *maxMinHeap) PeekTail() types.QueueItemAccessor {
	h.mu.RLock()
	defer h.mu.RUnlock()

	n := len(h.items)
	if n == 0 {
		return nil
	}
	if n == 1 {
		return h.items[0]
	}
	if n == 2 {
		// With two items, the root is max, the second is min.
		return h.items[1]
	}

	// With three or more items, the minimum element is guaranteed to be one of the two children of the root (at indices 1
	// and 2). We must compare them to find the true minimum.
	if h.comparator.Func()(h.items[1], h.items[2]) {
		return h.items[2]
	}
	return h.items[1]
}

// Add adds an item to the queue.
// Time complexity: O(log n).
func (h *maxMinHeap) Add(item types.QueueItemAccessor) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.push(item)
	h.byteSize.Add(item.OriginalRequest().ByteSize())
}

// push adds an item to the heap and restores the heap property.
func (h *maxMinHeap) push(item types.QueueItemAccessor) {
	heapItem := &heapItem{item: item, index: len(h.items)}
	h.items = append(h.items, item)
	item.SetHandle(heapItem)
	h.handles[item.Handle()] = heapItem
	h.up(len(h.items) - 1)
}

// up moves the item at index i up the heap to its correct position.
func (h *maxMinHeap) up(i int) {
	if i == 0 {
		return
	}

	parentIndex := (i - 1) / 2
	if isMinLevel(i) {
		// Current node is on a min level, parent is on a max level.
		// If the current node is greater than its parent, they are in the wrong order.
		if h.comparator.Func()(h.items[i], h.items[parentIndex]) {
			h.swap(i, parentIndex)
			// After swapping, the new parent (originally at i) might be larger than its ancestors.
			h.upMax(parentIndex)
		} else {
			// The order with the parent is correct, but it might be smaller than a grandparent.
			h.upMin(i)
		}
	} else { // On a max level
		// Current node is on a max level, parent is on a min level.
		// If the current node is smaller than its parent, they are in the wrong order.
		if h.comparator.Func()(h.items[parentIndex], h.items[i]) {
			h.swap(i, parentIndex)
			// After swapping, the new parent (originally at i) might be smaller than its ancestors.
			h.upMin(parentIndex)
		} else {
			// The order with the parent is correct, but it might be larger than a grandparent.
			h.upMax(i)
		}
	}
}

// upMin moves an item up the min levels of the heap.
func (h *maxMinHeap) upMin(i int) {
	// Bubble up on min levels by comparing with grandparents.
	for {
		parentIndex := (i - 1) / 2
		if parentIndex == 0 {
			break
		}
		grandparentIndex := (parentIndex - 1) / 2
		// If the item is smaller than its grandparent, swap them.
		if h.comparator.Func()(h.items[grandparentIndex], h.items[i]) {
			h.swap(i, grandparentIndex)
			i = grandparentIndex
		} else {
			break
		}
	}
}

// upMax moves an item up the max levels of the heap.
func (h *maxMinHeap) upMax(i int) {
	// Bubble up on max levels by comparing with grandparents.
	for {
		parentIndex := (i - 1) / 2
		if parentIndex == 0 {
			break
		}
		grandparentIndex := (parentIndex - 1) / 2
		// If the item is larger than its grandparent, swap them.
		if h.comparator.Func()(h.items[i], h.items[grandparentIndex]) {
			h.swap(i, grandparentIndex)
			i = grandparentIndex
		} else {
			break
		}
	}
}

// swap swaps two items in the heap and updates their handles.
func (h *maxMinHeap) swap(i, j int) {
	h.items[i], h.items[j] = h.items[j], h.items[i]
	h.handles[h.items[i].Handle()].index = i
	h.handles[h.items[j].Handle()].index = j
}

// Remove removes an item from the queue.
// Time complexity: O(log n).
func (h *maxMinHeap) Remove(handle types.QueueItemHandle) (types.QueueItemAccessor, error) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if handle == nil {
		return nil, framework.ErrInvalidQueueItemHandle
	}

	if handle.IsInvalidated() {
		return nil, framework.ErrInvalidQueueItemHandle
	}

	heapItem, ok := handle.(*heapItem)
	if !ok {
		return nil, framework.ErrInvalidQueueItemHandle
	}

	// Now we can check if the handle is in the map
	_, ok = h.handles[handle]
	if !ok {
		return nil, framework.ErrQueueItemNotFound
	}

	i := heapItem.index
	item := h.items[i]
	n := len(h.items) - 1

	if i < n {
		// Swap the item to be removed with the last item.
		h.swap(i, n)
		// Remove the last item (which is the one we wanted to remove).
		h.items = h.items[:n]
		delete(h.handles, handle)
		h.byteSize.Add(^item.OriginalRequest().ByteSize() + 1) // Atomic subtraction

		// The swapped item at index i might violate the heap property, so we must restore it
		// by bubbling the item down the heap.
		h.down(i)
	} else {
		// It's the last item, just remove it.
		h.items = h.items[:n]
		delete(h.handles, handle)
		h.byteSize.Add(^item.OriginalRequest().ByteSize() + 1) // Atomic subtraction
	}

	handle.Invalidate()
	return item, nil
}

// down moves the item at index i down the heap to its correct position.
func (h *maxMinHeap) down(i int) {
	if isMinLevel(i) {
		h.downMin(i)
	} else {
		h.downMax(i)
	}
}

// downMin moves an item down the min levels of the heap.
func (h *maxMinHeap) downMin(i int) {
	for {
		m := h.findSmallestChildOrGrandchild(i)
		if m == -1 {
			break
		}

		// If the smallest descendant is smaller than the current item, swap them.
		if h.comparator.Func()(h.items[i], h.items[m]) {
			h.swap(i, m)
			parentOfM := (m - 1) / 2
			// If m was a grandchild, it might be larger than its new parent.
			if parentOfM != i {
				if h.comparator.Func()(h.items[m], h.items[parentOfM]) {
					h.swap(m, parentOfM)
				}
			}
			i = m
		} else {
			break
		}
	}
}

// downMax moves an item down the max levels of the heap.
func (h *maxMinHeap) downMax(i int) {
	for {
		m := h.findLargestChildOrGrandchild(i)
		if m == -1 {
			break
		}

		// If the largest descendant is larger than the current item, swap them.
		if h.comparator.Func()(h.items[m], h.items[i]) {
			h.swap(i, m)
			parentOfM := (m - 1) / 2
			// If m was a grandchild, it might be smaller than its new parent.
			if parentOfM != i {
				if h.comparator.Func()(h.items[parentOfM], h.items[m]) {
					h.swap(m, parentOfM)
				}
			}
			i = m
		} else {
			break
		}
	}
}

// findSmallestChildOrGrandchild finds the index of the smallest child or grandchild of i.
func (h *maxMinHeap) findSmallestChildOrGrandchild(i int) int {
	leftChild := 2*i + 1
	if leftChild >= len(h.items) {
		return -1 // No descendants
	}

	m := leftChild // Start with the left child as the smallest.

	// Compare with right child.
	rightChild := 2*i + 2
	if rightChild < len(h.items) && h.comparator.Func()(h.items[m], h.items[rightChild]) {
		m = rightChild
	}

	// Compare with grandchildren.
	grandchildStart := 2*leftChild + 1
	grandchildEnd := grandchildStart + 4
	for j := grandchildStart; j < grandchildEnd && j < len(h.items); j++ {
		if h.comparator.Func()(h.items[m], h.items[j]) {
			m = j
		}
	}
	return m
}

// findLargestChildOrGrandchild finds the index of the largest child or grandchild of i.
func (h *maxMinHeap) findLargestChildOrGrandchild(i int) int {
	leftChild := 2*i + 1
	if leftChild >= len(h.items) {
		return -1 // No descendants
	}

	m := leftChild // Start with the left child as the largest.

	// Compare with right child.
	rightChild := 2*i + 2
	if rightChild < len(h.items) && h.comparator.Func()(h.items[rightChild], h.items[m]) {
		m = rightChild
	}

	// Compare with grandchildren.
	grandchildStart := 2*leftChild + 1
	grandchildEnd := grandchildStart + 4
	for j := grandchildStart; j < grandchildEnd && j < len(h.items); j++ {
		if h.comparator.Func()(h.items[j], h.items[m]) {
			m = j
		}
	}
	return m
}

// isMinLevel checks if the given index is on a min level of the heap.
func isMinLevel(i int) bool {
	// The level is the floor of log2(i+1).
	// Levels are 0-indexed. 0, 2, 4... are max levels. 1, 3, 5... are min levels.
	// An integer is on a min level if its level number is odd.
	level := int(math.Log2(float64(i + 1)))
	return level%2 != 0
}

// Cleanup removes items from the queue that satisfy the predicate.
func (h *maxMinHeap) Cleanup(predicate framework.PredicateFunc) []types.QueueItemAccessor {
	h.mu.Lock()
	defer h.mu.Unlock()

	var removedItems []types.QueueItemAccessor
	var itemsToKeep []types.QueueItemAccessor

	for _, item := range h.items {
		if predicate(item) {
			removedItems = append(removedItems, item)
			handle := item.Handle()
			if handle != nil {
				handle.Invalidate()
				delete(h.handles, handle)
			}
			h.byteSize.Add(^item.OriginalRequest().ByteSize() + 1) // Atomic subtraction
		} else {
			itemsToKeep = append(itemsToKeep, item)
		}
	}

	if len(removedItems) > 0 {
		h.items = itemsToKeep
		// Re-establish the heap property on the remaining items.
		// First, update all the indices in the handles map.
		for i, item := range h.items {
			h.handles[item.Handle()].index = i
		}
		// Then, starting from the last non-leaf node, trickle down to fix the heap.
		for i := len(h.items)/2 - 1; i >= 0; i-- {
			h.down(i)
		}
	}

	return removedItems
}

// Drain removes all items from the queue.
func (h *maxMinHeap) Drain() []types.QueueItemAccessor {
	h.mu.Lock()
	defer h.mu.Unlock()

	drainedItems := make([]types.QueueItemAccessor, len(h.items))
	copy(drainedItems, h.items)

	// Invalidate all handles.
	for _, item := range h.items {
		if handle := item.Handle(); handle != nil {
			handle.Invalidate()
		}
	}

	// Clear the internal state.
	h.items = make([]types.QueueItemAccessor, 0)
	h.handles = make(map[types.QueueItemHandle]*heapItem)
	h.byteSize.Store(0)

	return drainedItems
}
