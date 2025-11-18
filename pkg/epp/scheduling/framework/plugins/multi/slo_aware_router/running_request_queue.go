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

package slo_aware_router

import (
	"container/heap"
	"fmt"
	"sort"
	"strings"
	"sync"
)

// request represents an element in the priority queue.
// The index is needed by heap.Remove and is maintained by the heap.Interface methods.
type request struct {
	id    string  // Unique identifier
	tpot  float64 // The priority value (lower is higher priority)
	index int
}

// requestPriorityQueue implements a priority queue with item removal by ID.
type requestPriorityQueue struct {
	items  []*request
	lookup map[string]*request
	mutex  sync.RWMutex
}

// newRequestPriorityQueue initializes and returns a new PriorityQueue.
func newRequestPriorityQueue() *requestPriorityQueue {
	return &requestPriorityQueue{
		lookup: make(map[string]*request),
		items:  []*request{},
	}
}

// Clone creates a deep copy of the priority queue.
// The new queue is completely independent of the original.
func (pq *requestPriorityQueue) Clone() *requestPriorityQueue {
	pq.mutex.RLock()
	defer pq.mutex.RUnlock()

	// Initialize a new priority queue with pre-allocated capacity.
	clonedPq := &requestPriorityQueue{
		items:  make([]*request, len(pq.items)),
		lookup: make(map[string]*request, len(pq.lookup)),
	}

	// Iterate through the original items to create deep copies.
	for i, oldItem := range pq.items {
		// Create a new Request struct, copying all values.
		newItem := &request{
			id:    oldItem.id,
			tpot:  oldItem.tpot,
			index: oldItem.index,
		}

		// Assign the new item to the cloned queue's items slice.
		clonedPq.items[i] = newItem
		// Update the lookup map in the cloned queue to point to the new item.
		clonedPq.lookup[newItem.id] = newItem
	}

	return clonedPq
}

// Len is the number of items in the queue.
func (pq *requestPriorityQueue) Len() int { return len(pq.items) }

// Less reports whether the item with index i should sort before the item with index j.
func (pq *requestPriorityQueue) Less(i, j int) bool {
	return pq.items[i].tpot < pq.items[j].tpot
}

// Swap swaps the items with indexes i and j.
func (pq *requestPriorityQueue) Swap(i, j int) {
	pq.items[i], pq.items[j] = pq.items[j], pq.items[i]
	pq.items[i].index = i
	pq.items[j].index = j
}

// Push adds an item to the heap.
func (pq *requestPriorityQueue) Push(x any) {
	item := x.(*request)
	item.index = len(pq.items)
	pq.items = append(pq.items, item)
}

// Pop removes and returns the minimum item from the heap.
func (pq *requestPriorityQueue) Pop() any {
	n := len(pq.items)
	item := pq.items[n-1]
	pq.items[n-1] = nil // avoid memory leak
	item.index = -1     // for safety
	pq.items = pq.items[0 : n-1]
	return item
}

// Add adds a new item to the queue.
// Returns true if the item was added, false if an item with the same ID already exists.
func (pq *requestPriorityQueue) Add(id string, tpot float64) bool {
	pq.mutex.Lock()
	defer pq.mutex.Unlock()

	// Validate input
	if id == "" {
		return false
	}
	if tpot < 0 {
		return false
	}

	// If item already exists, do not add
	if _, exists := pq.lookup[id]; exists {
		return false
	}

	item := &request{
		id:   id,
		tpot: tpot,
	}
	pq.lookup[id] = item
	heap.Push(pq, item)
	return true
}

// Update modifies the TPOT value of an existing item in the queue.
// If the item doesn't exist, this method does nothing.
func (pq *requestPriorityQueue) Update(id string, tpot float64) bool {
	pq.mutex.Lock()
	defer pq.mutex.Unlock()

	// Validate input
	if tpot < 0 {
		return false
	}

	item, exists := pq.lookup[id]
	if !exists {
		return false
	}

	item.tpot = tpot
	heap.Fix(pq, item.index)
	return true
}

// Remove removes an item from the queue by its ID.
func (pq *requestPriorityQueue) Remove(id string) (*request, bool) {
	pq.mutex.Lock()
	defer pq.mutex.Unlock()

	item, ok := pq.lookup[id]
	if !ok {
		return nil, false
	}
	removed := heap.Remove(pq, item.index).(*request)
	delete(pq.lookup, id)
	return removed, true
}

// Peek returns the item with the lowest value without removing it.
func (pq *requestPriorityQueue) Peek() *request {
	pq.mutex.RLock()
	defer pq.mutex.RUnlock()

	if len(pq.items) == 0 {
		return nil
	}
	return pq.items[0]
}

// GetSize returns the current number of items in the queue.
func (pq *requestPriorityQueue) GetSize() int {
	pq.mutex.RLock()
	defer pq.mutex.RUnlock()
	return len(pq.items)
}

// Contains checks if an item with the given ID exists in the queue.
func (pq *requestPriorityQueue) Contains(id string) bool {
	pq.mutex.RLock()
	defer pq.mutex.RUnlock()
	_, exists := pq.lookup[id]
	return exists
}

// ToSlice returns a copy of all items in the queue, sorted by ID for stable comparison.
// This is primarily intended for testing and validation.
func (pq *requestPriorityQueue) ToSlice() []*request {
	pq.mutex.RLock()
	defer pq.mutex.RUnlock()

	// Create a copy to avoid returning a reference to the internal slice.
	itemsCopy := make([]*request, len(pq.items))
	copy(itemsCopy, pq.items)

	// Sort by ID to have a deterministic order for comparison in tests.
	sort.Slice(itemsCopy, func(i, j int) bool {
		return itemsCopy[i].id < itemsCopy[j].id
	})

	return itemsCopy
}

// String returns a string representation of the queue for debugging.
func (pq *requestPriorityQueue) String() string {
	pq.mutex.RLock()
	defer pq.mutex.RUnlock()

	if len(pq.items) == 0 {
		return "RequestPriorityQueue: []"
	}

	var builder strings.Builder
	builder.WriteString("RequestPriorityQueue: [")

	for i, item := range pq.items {
		if i > 0 {
			builder.WriteString(", ")
		}
		builder.WriteString(item.id)
		builder.WriteString("(")
		builder.WriteString(fmt.Sprintf("%.2f", item.tpot))
		builder.WriteString(")")
	}

	builder.WriteString("]")
	return builder.String()
}
