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

package contracts

import "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"

// PredicateFunc defines a function that returns true if a given item matches a certain condition.
// It is used by SafeQueue.Cleanup to filter items.
type PredicateFunc func(item flowcontrol.QueueItemAccessor) bool

// SafeQueue defines the contract for a single, concurrent-safe queue implementation.
// This interface is designed for in-memory, synchronous flow control. Implementations are expected to be unbounded;
// capacity management occurs outside the queue implementation.
// All implementations MUST be goroutine-safe.
type SafeQueue interface {
	flowcontrol.QueueInspectionMethods

	// Add enqueues an item. It must associate a new, unique QueueItemHandle with the item by calling
	// item.SetHandle().
	// Contract: The caller MUST NOT provide a nil item.
	Add(item flowcontrol.QueueItemAccessor)

	// Remove atomically finds and removes the item identified by the given handle.
	// On success, implementations MUST invalidate the provided handle by calling handle.Invalidate().
	// Returns the removed item.
	// Returns ErrInvalidQueueItemHandle if the handle is invalid.
	// Returns ErrQueueItemNotFound if the handle is valid but the item is not in the queue.
	Remove(handle flowcontrol.QueueItemHandle) (removedItem flowcontrol.QueueItemAccessor, err error)

	// Cleanup iterates through the queue and atomically removes all items for which the predicate returns true,
	// returning them in a slice.
	// The handle for each removed item MUST be invalidated.
	Cleanup(predicate PredicateFunc) (cleanedItems []flowcontrol.QueueItemAccessor)

	// Drain atomically removes all items from the queue and returns them in a slice.
	// The handle for all removed items MUST be invalidated. The queue MUST be empty after this operation.
	Drain() (drainedItems []flowcontrol.QueueItemAccessor)
}
