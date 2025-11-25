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

// Package listqueue provides a high-performance, concurrent-safe FIFO (First-In, First-Out) implementation of
// implementation of the `framework.SafeQueue` based on the standard library's `container/list`.
package queue

import (
	"container/list"
	"sync"
	"sync/atomic"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// ListQueueName is the name of the list-based queue implementation.
//
// This queue provides a high-performance, low-overhead implementation based on a standard `container/list`.
// It advertises the `CapabilityFIFO`.
//
// # Behavioral Guarantees
//
// The core guarantee of this queue is strict physical First-In, First-Out (FIFO) ordering. It processes items in the
// exact order they are added to the queue on a specific shard.
//
// # Performance and Trade-offs
//
// Because the physical insertion order may not match a request's logical arrival time (due to the
// `controller.FlowController`'s internal "bounce-and-retry" mechanic), this queue provides an*approximate FCFS behavior
// from a system-wide perspective.
//
// Given that true end-to-end ordering is non-deterministic in a distributed system, this high-performance queue is the
// recommended default for most FCFS-like policies. It prioritizes throughput and efficiency, which aligns with the
// primary goal of the Flow Control system.
//
// For workloads that require the strictest possible logical-time ordering this layer can provide, explicitly using a
// queue that supports `CapabilityPriorityConfigurable` is the appropriate choice.
const ListQueueName = "ListQueue"

func init() {
	MustRegisterQueue(RegisteredQueueName(ListQueueName),
		func(_ framework.ItemComparator) (framework.SafeQueue, error) {
			// The list queue is a simple FIFO queue and does not use a comparator.
			return newListQueue(), nil
		})
}

// listQueue is the internal implementation of the ListQueue.
// See the documentation for the exported `ListQueueName` constant for detailed user-facing information.
type listQueue struct {
	requests *list.List
	byteSize atomic.Uint64
	mu       sync.RWMutex
}

// listItemHandle is the concrete type for `types.QueueItemHandle` used by `listQueue`.
// It wraps the `list.Element` and includes a pointer to the owning `listQueue` for validation.
type listItemHandle struct {
	element       *list.Element
	owner         *listQueue
	isInvalidated bool
}

// Handle returns the underlying queue-specific raw handle.
func (lh *listItemHandle) Handle() any {
	return lh.element
}

// Invalidate marks this handle instance as no longer valid for future operations.
func (lh *listItemHandle) Invalidate() {
	lh.isInvalidated = true
}

// IsInvalidated returns true if this handle instance has been marked as invalid.
func (lh *listItemHandle) IsInvalidated() bool {
	return lh.isInvalidated
}

var _ types.QueueItemHandle = &listItemHandle{}

// newListQueue creates a new `listQueue` instance.
func newListQueue() *listQueue {
	return &listQueue{
		requests: list.New(),
	}
}

// --- `framework.SafeQueue` Interface Implementation ---

// Add enqueues an item to the back of the list.
func (lq *listQueue) Add(item types.QueueItemAccessor) {
	lq.mu.Lock()
	defer lq.mu.Unlock()

	element := lq.requests.PushBack(item)
	lq.byteSize.Add(item.OriginalRequest().ByteSize())
	item.SetHandle(&listItemHandle{element: element, owner: lq})
}

// Remove removes an item identified by the given handle from the queue.
func (lq *listQueue) Remove(handle types.QueueItemHandle) (types.QueueItemAccessor, error) {
	lq.mu.Lock()
	defer lq.mu.Unlock()

	if handle == nil || handle.IsInvalidated() {
		return nil, framework.ErrInvalidQueueItemHandle
	}

	lh, ok := handle.(*listItemHandle)
	if !ok {
		return nil, framework.ErrInvalidQueueItemHandle
	}

	if lh.owner != lq {
		return nil, framework.ErrQueueItemNotFound
	}

	item := lh.element.Value.(types.QueueItemAccessor)
	lq.requests.Remove(lh.element)
	lq.byteSize.Add(^item.OriginalRequest().ByteSize() + 1) // Atomic subtraction
	handle.Invalidate()
	return item, nil
}

// Cleanup removes items from the queue that satisfy the predicate.
func (lq *listQueue) Cleanup(predicate framework.PredicateFunc) (cleanedItems []types.QueueItemAccessor) {
	lq.mu.Lock()
	defer lq.mu.Unlock()

	var removedItems []types.QueueItemAccessor
	var next *list.Element

	for e := lq.requests.Front(); e != nil; e = next {
		next = e.Next() // Get next before potentially removing e

		item := e.Value.(types.QueueItemAccessor)
		if predicate(item) {
			lq.requests.Remove(e)
			lq.byteSize.Add(^item.OriginalRequest().ByteSize() + 1) // Atomic subtraction
			if itemHandle := item.Handle(); itemHandle != nil {
				itemHandle.Invalidate()
			}
			removedItems = append(removedItems, item)
		}
	}
	return removedItems
}

// Drain removes all items from the queue and returns them.
func (lq *listQueue) Drain() (removedItems []types.QueueItemAccessor) {
	lq.mu.Lock()
	defer lq.mu.Unlock()

	removedItems = make([]types.QueueItemAccessor, 0, lq.requests.Len())

	for e := lq.requests.Front(); e != nil; e = e.Next() {
		item := e.Value.(types.QueueItemAccessor)
		removedItems = append(removedItems, item)
		if handle := item.Handle(); handle != nil {
			handle.Invalidate()
		}
	}

	lq.requests.Init()
	lq.byteSize.Store(0)
	return removedItems
}

// Name returns the name of the queue.
func (lq *listQueue) Name() string {
	return ListQueueName
}

// Capabilities returns the capabilities of the queue.
func (lq *listQueue) Capabilities() []framework.QueueCapability {
	return []framework.QueueCapability{framework.CapabilityFIFO}
}

// Len returns the number of items in the queue.
func (lq *listQueue) Len() int {
	lq.mu.RLock()
	defer lq.mu.RUnlock()
	return lq.requests.Len()
}

// ByteSize returns the total byte size of all items in the queue.
func (lq *listQueue) ByteSize() uint64 {
	return lq.byteSize.Load()
}

// PeekHead returns the item at the front of the queue without removing it.
func (lq *listQueue) PeekHead() types.QueueItemAccessor {
	lq.mu.RLock()
	defer lq.mu.RUnlock()

	if lq.requests.Len() == 0 {
		return nil
	}
	element := lq.requests.Front()
	return element.Value.(types.QueueItemAccessor)
}

// PeekTail returns the item at the back of the queue without removing it.
func (lq *listQueue) PeekTail() types.QueueItemAccessor {
	lq.mu.RLock()
	defer lq.mu.RUnlock()

	if lq.requests.Len() == 0 {
		return nil
	}
	element := lq.requests.Back()
	return element.Value.(types.QueueItemAccessor)
}
