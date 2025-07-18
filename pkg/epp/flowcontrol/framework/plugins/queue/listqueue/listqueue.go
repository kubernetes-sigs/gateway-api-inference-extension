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

// Package listqueue provides a simple, concurrent-safe queue implementation using a standard library
// `container/list.List` as the underlying data structure for FIFO (First-In, First-Out) behavior.
package listqueue

import (
	"container/list"
	"sync"
	"sync/atomic"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// ListQueueName is the name of the list queue implementation.
const ListQueueName = "ListQueue"

func init() {
	queue.MustRegisterQueue(queue.RegisteredQueueName(ListQueueName),
		func(_ framework.ItemComparator) (framework.SafeQueue, error) {
			// The list queue is a simple FIFO queue and does not use a comparator.
			return newListQueue(), nil
		})
}

// listQueue implements the `framework.SafeQueue` interface using a standard `container/list.List` for FIFO behavior.
// This implementation is concurrent-safe.
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
func (lq *listQueue) Add(item types.QueueItemAccessor) error {
	lq.mu.Lock()
	defer lq.mu.Unlock()

	if item == nil {
		return framework.ErrNilQueueItem
	}

	element := lq.requests.PushBack(item)
	lq.byteSize.Add(item.OriginalRequest().ByteSize())
	item.SetHandle(&listItemHandle{element: element, owner: lq})
	return nil
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
func (lq *listQueue) Cleanup(predicate framework.PredicateFunc) (cleanedItems []types.QueueItemAccessor, err error) {
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
	return removedItems, nil
}

// Drain removes all items from the queue and returns them.
func (lq *listQueue) Drain() (removedItems []types.QueueItemAccessor, err error) {
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
	return removedItems, nil
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
func (lq *listQueue) PeekHead() (head types.QueueItemAccessor, err error) {
	lq.mu.RLock()
	defer lq.mu.RUnlock()

	if lq.requests.Len() == 0 {
		return nil, framework.ErrQueueEmpty
	}
	element := lq.requests.Front()
	return element.Value.(types.QueueItemAccessor), nil
}

// PeekTail returns the item at the back of the queue without removing it.
func (lq *listQueue) PeekTail() (tail types.QueueItemAccessor, err error) {
	lq.mu.RLock()
	defer lq.mu.RUnlock()

	if lq.requests.Len() == 0 {
		return nil, framework.ErrQueueEmpty
	}
	element := lq.requests.Back()
	return element.Value.(types.QueueItemAccessor), nil
}
