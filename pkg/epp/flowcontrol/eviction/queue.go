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
	"container/heap"
	"sync"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
)

// heapEntry is the internal wrapper that pairs an EvictionItem with its heap index.
// This keeps the index as a queue implementation detail, not exposed on the domain type.
type heapEntry struct {
	item  *flowcontrol.EvictionItem
	index int
}

// EvictionQueue tracks all in-flight requests and maintains a min-heap of evictable ones.
// The ordering and filter policies are pluggable via the EvictionOrderingPolicy and EvictionFilterPolicy interfaces.
//
// All exported methods are goroutine-safe.
type EvictionQueue struct {
	mu          sync.Mutex
	h           evictionHeap
	handles     map[string]*heapEntry                // requestID → heap entry (for O(log n) removal)
	allInFlight map[string]*flowcontrol.EvictionItem // all tracked requests
	filter      flowcontrol.EvictionFilterPolicy
}

// NewEvictionQueue creates a queue with the given ordering and filter policies.
func NewEvictionQueue(
	ordering flowcontrol.EvictionOrderingPolicy,
	filter flowcontrol.EvictionFilterPolicy,
) *EvictionQueue {
	q := &EvictionQueue{
		handles:     make(map[string]*heapEntry),
		allInFlight: make(map[string]*flowcontrol.EvictionItem),
		filter:      filter,
	}
	q.h = evictionHeap{
		entries: make([]*heapEntry, 0),
		less:    ordering.Less,
	}
	heap.Init(&q.h)
	return q
}

// Track registers an in-flight request. If the filter policy accepts it, the item enters the eviction heap.
// If a request with the same ID is already tracked, it is replaced.
func (q *EvictionQueue) Track(item *flowcontrol.EvictionItem) {
	q.mu.Lock()
	defer q.mu.Unlock()

	// Remove existing entry if re-tracked with the same ID.
	if existing, ok := q.handles[item.RequestID]; ok {
		heap.Remove(&q.h, existing.index)
		delete(q.handles, item.RequestID)
	}

	q.allInFlight[item.RequestID] = item

	if q.filter.Accept(item) {
		e := &heapEntry{item: item}
		heap.Push(&q.h, e)
		q.handles[item.RequestID] = e
	}
}

// Untrack removes a request from tracking. Called when a request completes.
func (q *EvictionQueue) Untrack(requestID string) {
	q.mu.Lock()
	defer q.mu.Unlock()

	if e, ok := q.handles[requestID]; ok {
		heap.Remove(&q.h, e.index)
		delete(q.handles, requestID)
	}

	delete(q.allInFlight, requestID)
}

// PopN removes and returns up to n most-evictable items from the heap.
func (q *EvictionQueue) PopN(n int) []*flowcontrol.EvictionItem {
	q.mu.Lock()
	defer q.mu.Unlock()

	result := make([]*flowcontrol.EvictionItem, 0, n)
	for i := 0; i < n && q.h.Len() > 0; i++ {
		e := heap.Pop(&q.h).(*heapEntry)
		delete(q.handles, e.item.RequestID)
		delete(q.allInFlight, e.item.RequestID)
		result = append(result, e.item)
	}
	return result
}

// Peek returns a shallow copy of the most-evictable item without removing it, or nil if the heap
// is empty. The returned copy is safe to read without holding any lock and modifications to it do
// not affect the heap ordering.
func (q *EvictionQueue) Peek() *flowcontrol.EvictionItem {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.h.Len() == 0 {
		return nil
	}
	item := *q.h.entries[0].item
	return &item
}

// EvictableLen returns the number of items in the eviction heap.
func (q *EvictionQueue) EvictableLen() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.h.Len()
}

// InFlightLen returns the total number of tracked in-flight requests.
func (q *EvictionQueue) InFlightLen() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.allInFlight)
}

// --- container/heap.Interface implementation ---

type evictionHeap struct {
	entries []*heapEntry
	less    func(a, b *flowcontrol.EvictionItem) bool
}

func (h evictionHeap) Len() int           { return len(h.entries) }
func (h evictionHeap) Less(i, j int) bool { return h.less(h.entries[i].item, h.entries[j].item) }

func (h evictionHeap) Swap(i, j int) {
	h.entries[i], h.entries[j] = h.entries[j], h.entries[i]
	h.entries[i].index = i
	h.entries[j].index = j
}

func (h *evictionHeap) Push(x any) {
	e := x.(*heapEntry)
	e.index = len(h.entries)
	h.entries = append(h.entries, e)
}

func (h *evictionHeap) Pop() any {
	old := h.entries
	n := len(old)
	e := old[n-1]
	old[n-1] = nil // avoid memory leak
	h.entries = old[:n-1]
	e.index = -1
	return e
}
