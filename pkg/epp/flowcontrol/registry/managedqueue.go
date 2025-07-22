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

package registry

import (
	"fmt"
	"sync/atomic"

	"github.com/go-logr/logr"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// parentStatsReconciler defines the callback function that a `managedQueue` uses to propagate its statistics changes up
// to its parent `registryShard`.
type parentStatsReconciler func(lenDelta, byteSizeDelta int64)

// managedQueue implements `contracts.ManagedQueue`. It is a stateful decorator that wraps a `framework.SafeQueue`,
// augmenting it with two critical, registry-level responsibilities:
//  1. Atomic Statistics: It maintains its own `len` and `byteSize` counters, which are updated atomically. This allows
//     the parent `registryShard` to aggregate statistics across many queues without locks.
//  2. Lifecycle Enforcement: It tracks the queue's lifecycle state (active vs. draining) via an `isActive` flag. This
//     is crucial for graceful flow updates, as it allows the registry to stop new requests from being enqueued while
//     allowing existing items to be drained.
//
// # Statistical Integrity
//
// For performance, `managedQueue` maintains its own `len` and `byteSize` fields using atomic operations. This provides
// O(1) access for the parent `registryShard`'s aggregated statistics without needing to lock the underlying queue.
//
// This design is predicated on two critical assumptions:
//  1. Exclusive Access: All mutating operations on the underlying `framework.SafeQueue` MUST be performed exclusively
//     through this `managedQueue` wrapper. Direct access to the underlying queue will cause statistical drift.
//  2. In-Process Queue: The `framework.SafeQueue` implementation is an in-process data structure (e.g., a list or
//     heap). Its state MUST NOT change through external mechanisms. For example, a queue implementation backed by a
//     distributed cache like Redis with its own TTL-based eviction policy would violate this assumption and lead to
//     state inconsistency, as items could be removed without notifying the `managedQueue`.
//
// This approach avoids the need for the `framework.SafeQueue` interface to return state deltas on each operation,
// keeping its contract simpler.
type managedQueue struct {
	// Note: There is no mutex here. Concurrency control is delegated to the underlying `framework.SafeQueue` and the
	// atomic operations on the stats fields.
	queue               framework.SafeQueue
	dispatchPolicy      framework.IntraFlowDispatchPolicy
	flowSpec            types.FlowSpecification
	byteSize            atomic.Uint64
	len                 atomic.Uint64
	isActive            atomic.Bool
	reconcileShardStats parentStatsReconciler
	logger              logr.Logger
}

// newManagedQueue creates a new instance of a `managedQueue`.
func newManagedQueue(
	queue framework.SafeQueue,
	dispatchPolicy framework.IntraFlowDispatchPolicy,
	flowSpec types.FlowSpecification,
	logger logr.Logger,
	reconcileShardStats parentStatsReconciler,
) *managedQueue {
	mqLogger := logger.WithName("managed-queue").WithValues(
		"flowID", flowSpec.ID,
		"priority", flowSpec.Priority,
		"queueType", queue.Name(),
	)
	mq := &managedQueue{
		queue:               queue,
		dispatchPolicy:      dispatchPolicy,
		flowSpec:            flowSpec,
		reconcileShardStats: reconcileShardStats,
		logger:              mqLogger,
	}
	mq.isActive.Store(true)
	return mq
}

// markAsDraining is an internal method called by the parent shard to transition this queue to a draining state.
// Once a queue is marked as draining, it will no longer accept new items via `Add`.
func (mq *managedQueue) markAsDraining() {
	// Use CompareAndSwap to ensure we only log the transition once.
	if mq.isActive.CompareAndSwap(true, false) {
		mq.logger.V(logging.DEFAULT).Info("Queue marked as draining")
	}
}

// FlowQueueAccessor returns a new `flowQueueAccessor` instance, which provides a read-only, policy-facing view of the
// queue.
func (mq *managedQueue) FlowQueueAccessor() framework.FlowQueueAccessor {
	return &flowQueueAccessor{mq: mq}
}

// Add first checks if the queue is active. If it is, it wraps the underlying `framework.SafeQueue.Add` call and
// atomically updates the queue's and the parent shard's statistics.
func (mq *managedQueue) Add(item types.QueueItemAccessor) error {
	if !mq.isActive.Load() {
		return fmt.Errorf("flow instance %q is not active and cannot accept new requests: %w",
			mq.flowSpec.ID, contracts.ErrFlowInstanceNotFound)
	}
	if err := mq.queue.Add(item); err != nil {
		return err
	}
	mq.reconcileStats(1, int64(item.OriginalRequest().ByteSize()))
	mq.logger.V(logging.TRACE).Info("Request added to queue", "requestID", item.OriginalRequest().ID())
	return nil
}

// Remove wraps the underlying `framework.SafeQueue.Remove` call and atomically updates statistics upon successful
// removal.
func (mq *managedQueue) Remove(handle types.QueueItemHandle) (types.QueueItemAccessor, error) {
	removedItem, err := mq.queue.Remove(handle)
	if err != nil {
		return nil, err
	}
	mq.reconcileStats(-1, -int64(removedItem.OriginalRequest().ByteSize()))
	mq.logger.V(logging.TRACE).Info("Request removed from queue", "requestID", removedItem.OriginalRequest().ID())
	return removedItem, nil
}

// Cleanup wraps the underlying `framework.SafeQueue.Cleanup` call and atomically updates statistics for all removed
// items.
func (mq *managedQueue) Cleanup(predicate framework.PredicateFunc) (cleanedItems []types.QueueItemAccessor, err error) {
	cleanedItems, err = mq.queue.Cleanup(predicate)
	if err != nil || len(cleanedItems) == 0 {
		return cleanedItems, err
	}

	var lenDelta int64
	var byteSizeDelta int64
	for _, item := range cleanedItems {
		lenDelta--
		byteSizeDelta -= int64(item.OriginalRequest().ByteSize())
	}
	mq.reconcileStats(lenDelta, byteSizeDelta)
	mq.logger.V(logging.DEBUG).Info("Cleaned up queue", "removedItemCount", len(cleanedItems),
		"lenDelta", lenDelta, "byteSizeDelta", byteSizeDelta)
	return cleanedItems, nil
}

// Drain wraps the underlying `framework.SafeQueue.Drain` call and atomically updates statistics for all removed items.
func (mq *managedQueue) Drain() ([]types.QueueItemAccessor, error) {
	drainedItems, err := mq.queue.Drain()
	if err != nil || len(drainedItems) == 0 {
		return drainedItems, err
	}

	var lenDelta int64
	var byteSizeDelta int64
	for _, item := range drainedItems {
		lenDelta--
		byteSizeDelta -= int64(item.OriginalRequest().ByteSize())
	}
	mq.reconcileStats(lenDelta, byteSizeDelta)
	mq.logger.V(logging.DEBUG).Info("Drained queue", "itemCount", len(drainedItems),
		"lenDelta", lenDelta, "byteSizeDelta", byteSizeDelta)
	return drainedItems, nil
}

// reconcileStats atomically updates the queue's own statistics and calls the parent shard's reconciler to ensure
// aggregated stats remain consistent.
func (mq *managedQueue) reconcileStats(lenDelta, byteSizeDelta int64) {
	// The use of Add with a negative number on a Uint64 is the standard Go atomic way to perform subtraction, leveraging
	// two's complement arithmetic.
	mq.len.Add(uint64(lenDelta))
	mq.byteSize.Add(uint64(byteSizeDelta))
	if mq.reconcileShardStats != nil {
		mq.reconcileShardStats(lenDelta, byteSizeDelta)
	}
}

// --- Pass-through and accessor methods ---

// Name returns the name of the underlying queue implementation.
func (mq *managedQueue) Name() string { return mq.queue.Name() }

// Capabilities returns the capabilities of the underlying queue implementation.
func (mq *managedQueue) Capabilities() []framework.QueueCapability { return mq.queue.Capabilities() }

// Len returns the number of items in the queue.
func (mq *managedQueue) Len() int { return int(mq.len.Load()) }

// ByteSize returns the total byte size of all items in the queue.
func (mq *managedQueue) ByteSize() uint64 { return mq.byteSize.Load() }

// PeekHead returns the item at the front of the queue without removing it.
func (mq *managedQueue) PeekHead() (types.QueueItemAccessor, error) { return mq.queue.PeekHead() }

// PeekTail returns the item at the back of the queue without removing it.
func (mq *managedQueue) PeekTail() (types.QueueItemAccessor, error) { return mq.queue.PeekTail() }

// Comparator returns the `framework.ItemComparator` that defines this queue's item ordering logic, as dictated by its
// configured dispatch policy.
func (mq *managedQueue) Comparator() framework.ItemComparator { return mq.dispatchPolicy.Comparator() }

var _ contracts.ManagedQueue = &managedQueue{}

// --- flowQueueAccessor ---

// flowQueueAccessor implements `framework.FlowQueueAccessor`. It provides a read-only, policy-facing view of a
// `managedQueue`.
type flowQueueAccessor struct {
	mq *managedQueue
}

// Name returns the name of the queue.
func (a *flowQueueAccessor) Name() string { return a.mq.Name() }

// Capabilities returns the capabilities of the queue.
func (a *flowQueueAccessor) Capabilities() []framework.QueueCapability { return a.mq.Capabilities() }

// Len returns the number of items in the queue.
func (a *flowQueueAccessor) Len() int { return a.mq.Len() }

// ByteSize returns the total byte size of all items in the queue.
func (a *flowQueueAccessor) ByteSize() uint64 { return a.mq.ByteSize() }

// PeekHead returns the item at the front of the queue without removing it.
func (a *flowQueueAccessor) PeekHead() (types.QueueItemAccessor, error) { return a.mq.PeekHead() }

// PeekTail returns the item at the back of the queue without removing it.
func (a *flowQueueAccessor) PeekTail() (types.QueueItemAccessor, error) { return a.mq.PeekTail() }

// Comparator returns the `framework.ItemComparator` that defines this queue's item ordering logic.
func (a *flowQueueAccessor) Comparator() framework.ItemComparator { return a.mq.Comparator() }

// FlowSpec returns the `types.FlowSpecification` of the flow this queue accessor is associated with.
func (a *flowQueueAccessor) FlowSpec() types.FlowSpecification { return a.mq.flowSpec }

var _ framework.FlowQueueAccessor = &flowQueueAccessor{}
