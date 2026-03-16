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
	"sync/atomic"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/heap"
)

// MaxMinHeapName is the name of the max-min heap queue implementation.
const MaxMinHeapName = "MaxMinHeap"

func init() {
	MustRegisterQueue(RegisteredQueueName(MaxMinHeapName),
		func(policy flowcontrol.OrderingPolicy) (contracts.SafeQueue, error) {
			return newMaxMinHeap(policy), nil
		})
}

// maxMinHeap implements the SafeQueue interface by wrapping the generic heap.MaxMinHeap.
// It bridges between the domain-specific flowcontrol types (QueueItemAccessor, QueueItemHandle)
// and the generic, type-parameterized heap.
type maxMinHeap struct {
	inner    *heap.MaxMinHeap[flowcontrol.QueueItemAccessor]
	handles  map[flowcontrol.QueueItemHandle]*heap.Handle // QueueItemHandle → generic Handle
	byteSize atomic.Uint64
	policy   flowcontrol.OrderingPolicy
}

// newMaxMinHeap creates a new max-min heap with the given policy.
func newMaxMinHeap(policy flowcontrol.OrderingPolicy) *maxMinHeap {
	return &maxMinHeap{
		inner: heap.New(func(a, b flowcontrol.QueueItemAccessor) bool {
			return policy.Less(a, b)
		}),
		handles: make(map[flowcontrol.QueueItemHandle]*heap.Handle),
		policy:  policy,
	}
}

// handleBridge implements flowcontrol.QueueItemHandle by wrapping a generic heap.Handle.
type handleBridge struct {
	inner *heap.Handle
}

func (h *handleBridge) Handle() any         { return h.inner }
func (h *handleBridge) Invalidate()         { h.inner.Invalidate() }
func (h *handleBridge) IsInvalidated() bool { return h.inner.IsInvalidated() }

var _ flowcontrol.QueueItemHandle = &handleBridge{}

// --- SafeQueue Interface Implementation ---

// Name returns the name of the queue.
func (h *maxMinHeap) Name() string {
	return MaxMinHeapName
}

// Capabilities returns the capabilities of the queue.
func (h *maxMinHeap) Capabilities() []flowcontrol.QueueCapability {
	return []flowcontrol.QueueCapability{flowcontrol.CapabilityPriorityConfigurable}
}

// Len returns the number of items in the queue.
func (h *maxMinHeap) Len() int {
	return h.inner.Len()
}

// ByteSize returns the total byte size of all items in the queue.
func (h *maxMinHeap) ByteSize() uint64 {
	return h.byteSize.Load()
}

// PeekHead returns the item with the highest priority (max value) without removing it.
// Time complexity: O(1).
func (h *maxMinHeap) PeekHead() flowcontrol.QueueItemAccessor {
	item, ok := h.inner.PeekMax()
	if !ok {
		return nil
	}
	return item
}

// PeekTail returns the item with the lowest priority (min value) without removing it.
// Time complexity: O(1).
func (h *maxMinHeap) PeekTail() flowcontrol.QueueItemAccessor {
	item, ok := h.inner.PeekMin()
	if !ok {
		return nil
	}
	return item
}

// Add adds an item to the queue.
// Time complexity: O(log n).
func (h *maxMinHeap) Add(item flowcontrol.QueueItemAccessor) {
	genericHandle := h.inner.Add(item)
	bridge := &handleBridge{inner: genericHandle}
	item.SetHandle(bridge)
	h.handles[bridge] = genericHandle
	h.byteSize.Add(item.OriginalRequest().ByteSize())
}

// Remove removes an item from the queue.
// Time complexity: O(log n).
func (h *maxMinHeap) Remove(handle flowcontrol.QueueItemHandle) (flowcontrol.QueueItemAccessor, error) {
	if handle == nil {
		return nil, contracts.ErrInvalidQueueItemHandle
	}

	if handle.IsInvalidated() {
		return nil, contracts.ErrInvalidQueueItemHandle
	}

	bridge, ok := handle.(*handleBridge)
	if !ok {
		return nil, contracts.ErrInvalidQueueItemHandle
	}

	genericHandle, ok := h.handles[handle]
	if !ok {
		return nil, contracts.ErrQueueItemNotFound
	}

	item, ok := h.inner.Remove(genericHandle)
	if !ok {
		delete(h.handles, handle)
		return nil, contracts.ErrQueueItemNotFound
	}

	delete(h.handles, handle)
	bridge.Invalidate()
	h.byteSize.Add(^item.OriginalRequest().ByteSize() + 1) // Atomic subtraction
	return item, nil
}

// Cleanup removes items from the queue that satisfy the predicate.
func (h *maxMinHeap) Cleanup(predicate contracts.PredicateFunc) []flowcontrol.QueueItemAccessor {
	removed := h.inner.Cleanup(func(item flowcontrol.QueueItemAccessor) bool {
		return predicate(item)
	})

	for _, item := range removed {
		if qHandle := item.Handle(); qHandle != nil {
			delete(h.handles, qHandle)
			// The generic heap already invalidated its internal handle, but we also need to invalidate the bridge.
			qHandle.Invalidate()
		}
		h.byteSize.Add(^item.OriginalRequest().ByteSize() + 1)
	}

	return removed
}

// Drain removes all items from the queue.
func (h *maxMinHeap) Drain() []flowcontrol.QueueItemAccessor {
	drained := h.inner.Drain()

	for _, item := range drained {
		if qHandle := item.Handle(); qHandle != nil {
			delete(h.handles, qHandle)
			qHandle.Invalidate()
		}
	}

	h.byteSize.Store(0)
	return drained
}
