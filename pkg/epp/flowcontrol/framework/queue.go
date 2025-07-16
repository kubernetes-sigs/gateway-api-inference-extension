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

package framework

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// QueueCapability defines a functional capability that a `SafeQueue` implementation can provide.
// These capabilities allow policies (like `IntraFlowDispatchPolicy`) to declare their operational requirements,
// ensuring that a policy is always paired with a compatible queue.
//
// For example, a policy that requires a priority-ordered queue would declare `CapabilityPriorityConfigurable`, and the
// `contracts.FlowRegistry` would ensure it is paired with a queue implementation (like a heap) that provides this
// capability.
//
// While a simpler boolean method (e.g., `IsPriorityConfigurable()`) could satisfy current needs, this slice-based
// approach is intentionally chosen for future extensibility. It allows a queue to advertise multiple features (e.g.,
// dynamic priority reshuffling, size bounds, etc.) without future breaking changes to the `SafeQueue` interface.
type QueueCapability string

const (
	// CapabilityFIFO indicates that the queue operates in a First-In, First-Out manner.
	// `PeekHead()` will return the oldest item (by logical enqueue time).
	CapabilityFIFO QueueCapability = "FIFO"

	// CapabilityPriorityConfigurable indicates that the queue's ordering is determined by an `ItemComparator`.
	// `PeekHead()` will return the highest priority item according to this comparator.
	CapabilityPriorityConfigurable QueueCapability = "PriorityConfigurable"
)

// QueueInspectionMethods defines `SafeQueue`'s read-only methods.
type QueueInspectionMethods interface {
	// Name returns a string identifier for the concrete queue implementation type (e.g., "ListQueue").
	Name() string

	// Capabilities returns the set of functional capabilities this queue instance provides.
	Capabilities() []QueueCapability

	// Len returns the current number of items in the queue.
	Len() int

	// ByteSize returns the current total byte size of all items in the queue.
	ByteSize() uint64

	// PeekHead returns the item at the "head" of the queue (the item with the highest priority according to the queue's
	// ordering) without removing it.
	//
	// Returns `ErrQueueEmpty` if the queue is empty.
	PeekHead() (peekedItem types.QueueItemAccessor, err error)

	// PeekTail returns the item at the "tail" of the queue (the item with the lowest priority according to the queue's
	// ordering) without removing it.
	//
	// Returns `ErrQueueEmpty` if the queue is empty.
	PeekTail() (peekedItem types.QueueItemAccessor, err error)
}

// PredicateFunc defines a function that returns true if a given item matches a certain condition.
// It is used by `SafeQueue.Cleanup` to filter items.
type PredicateFunc func(item types.QueueItemAccessor) bool

// SafeQueue defines the contract for a single, concurrent-safe queue implementation.
//
// All implementations MUST be goroutine-safe.
type SafeQueue interface {
	QueueInspectionMethods

	// Add attempts to enqueue an item. On success, it must associate a new, unique `types.QueueItemHandle` with the item
	// by calling `item.SetHandle()`.
	Add(item types.QueueItemAccessor) error

	// Remove atomically finds and removes the item identified by the given handle.
	//
	// On success, implementations MUST invalidate the provided handle by calling `handle.Invalidate()`.
	//
	// Returns the removed item.
	// Returns `ErrInvalidQueueItemHandle` if the handle is invalid (e.g., nil, wrong type, already invalidated).
	// Returns `ErrQueueItemNotFound` if the handle is valid but the item is not in the queue.
	Remove(handle types.QueueItemHandle) (removedItem types.QueueItemAccessor, err error)

	// Cleanup iterates through the queue and atomically removes all items for which the predicate returns true, returning
	// them in a slice.
	//
	// The handle for each removed item MUST be invalidated.
	Cleanup(predicate PredicateFunc) (cleanedItems []types.QueueItemAccessor, err error)

	// Drain atomically removes all items from the queue and returns them in a slice.
	//
	// The handle for all removed items MUST be invalidated. The queue MUST be empty after this operation.
	Drain() (drainedItems []types.QueueItemAccessor, err error)
}
