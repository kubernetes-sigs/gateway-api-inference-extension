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

package types

import (
	"time"
)

// FlowControlRequest is the contract for an incoming request submitted to the `controller.FlowController`. It
// represents the "raw" user-provided data and context for a single unit of work.
//
// An object implementing this interface is the primary input to `FlowController.EnqueueAndWait()`. The controller then
// wraps this object with its own internal structures (which implement `QueueItemAccessor`) to manage the request's
// lifecycle without modifying the original.
type FlowControlRequest interface {
	// FlowKey returns the composite key that uniquely identifies the flow instance this request belongs to.
	// The `controller.FlowController` uses this key as the primary identifier to look up the correct
	// `contracts.ManagedQueue` and configured `framework.IntraFlowDispatchPolicy` from a `contracts.RegistryShard`.
	// The returned key is treated as an immutable value.
	FlowKey() FlowKey

	// ByteSize returns the request's size in bytes (e.g., prompt size). This is used by the `controller.FlowController`
	// for managing byte-based capacity limits and for `contracts.FlowRegistry` statistics.
	ByteSize() uint64

	// InitialEffectiveTTL returns the suggested Time-To-Live for this request.
	// This value is treated as a hint; the `controller.FlowController` may override it based on its own configuration or
	// policies. A zero value indicates the request has no specific TTL preference, and a system-wide default should be
	// applied.
	InitialEffectiveTTL() time.Duration

	// ID returns an optional, user-facing unique identifier for this specific request. It is intended for logging,
	// tracing, and observability. The `controller.FlowController` does not use this ID for dispatching decisions; it uses
	// the internal, opaque `QueueItemHandle`.
	ID() string

	// GetMetadata returns the opaque metadata associated with the request (e.g., header-derived context, subset filters).
	// This data is passed transparently to components like the contracts.PodLocator to resolve resources (candidate pods)
	// lazily during the dispatch cycle.
	GetMetadata() map[string]any
}

// QueueItemHandle is an opaque handle to an item that has been successfully added to a `framework.SafeQueue`. It acts
// as a key, allowing the `controller.FlowController` to perform targeted operations (like removal) on a specific item
// without needing to know the queue's internal structure.
//
// A handle is created by and bound to the specific `framework.SafeQueue` instance that stores the item.
type QueueItemHandle interface {
	// Handle returns the underlying, queue-specific raw handle (e.g., `*list.Element`).
	// This method is intended for internal use by the `framework.SafeQueue` implementation that created it.
	// Callers outside the queue implementation should treat the returned value as opaque.
	Handle() any

	// Invalidate marks this handle as no longer valid for future operations.
	// This method MUST be called by the `framework.SafeQueue` implementation itself after the item associated with this
	// handle has been removed.
	//
	// Conformance: Implementations of this method MUST be idempotent.
	Invalidate()

	// IsInvalidated returns true if this handle has been marked as invalid (e.g., by a call to `Invalidate`).
	// A `framework.SafeQueue` MUST reject any operation that attempts to use an invalidated handle, typically by
	// returning `framework.ErrInvalidQueueItemHandle`.
	IsInvalidated() bool
}

// QueueItemAccessor provides the internal, enriched, read-only view of a request being managed within the
// controller.FlowController`'s queues. It is the primary interface through which `framework.SafeQueue` implementations
// and policy plugins interact with request data and its associated flow control metadata.
//
// The `controller.FlowController` creates an object that implements this interface by wrapping an incoming
// `FlowControlRequest`.
type QueueItemAccessor interface {
	// OriginalRequest returns the underlying `FlowControlRequest` that this accessor provides a view of.
	// This method serves as an escape hatch, allowing policies or components that are aware of specific
	// `FlowControlRequest` implementations to perform type assertions and access richer, application-specific data.
	OriginalRequest() FlowControlRequest

	// EnqueueTime is the timestamp when the item was logically accepted by the `controller.FlowController` for queuing
	// (i.e., when `controller.FlowController.EnqueueAndWait()` was called). It does not reflect the time the request
	// landed in a `framework.SafeQueue` instance.
	EnqueueTime() time.Time

	// EffectiveTTL is the actual Time-To-Live assigned to this item by the `controller.FlowController`, taking into
	// account the request's preference (`FlowControlRequest.InitialEffectiveTTL()`) and any `controller.FlowController`
	// or per-flow defaults/policies.
	EffectiveTTL() time.Duration

	// Handle returns the `QueueItemHandle` associated with this item once it has been successfully added to a
	// `framework.SafeQueue`. It returns nil if the item is not yet in a queue.
	Handle() QueueItemHandle

	// SetHandle associates a `QueueItemHandle` with this item.
	//
	// Conformance: This method MUST be called by a `framework.SafeQueue` implementation within its `Add` method,
	// immediately after a new `QueueItemHandle` is created for the item. This ensures that the item always carries a
	// valid handle while it is in a queue. This method is not intended for use outside of `framework.SafeQueue`
	// implementations.
	SetHandle(handle QueueItemHandle)
}
