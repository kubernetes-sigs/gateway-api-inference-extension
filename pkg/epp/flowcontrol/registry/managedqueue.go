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

// managedQueueCallbacks groups the callback functions that a `managedQueue` uses to communicate with its parent shard.
type managedQueueCallbacks struct {
	// propagateStatsDelta is called to propagate statistics changes (e.g., queue length, byte size) up to the parent.
	propagateStatsDelta propagateStatsDeltaFunc
	// signalQueueState is called to signal important lifecycle events, such as becoming empty or fully drained, which are
	// used by the garbage collector.
	signalQueueState signalQueueStateFunc
}

// managedQueue implements `contracts.ManagedQueue`. It is a stateful decorator that wraps a generic
// `framework.SafeQueue` to enrich it with the state and behaviors required by the registry's control plane.
//
// # Role: The Stateful Decorator
//
// Its responsibilities are centered around augmenting a generic queue implementation with critical registry features.
// It is designed for high-performance, concurrent access on the hot path (Enqueue/Dequeue).
//
//  1. Atomic Statistics (Lock-Free): It maintains its own `len` and `byteSize` counters, which are updated atomically
//     using sophisticated lock-free patterns (see `propagateStatsDelta`). This provides O(1), lock-free access for the
//     parent `registryShard` to aggregate statistics across many queues.
//
//  2. Lifecycle Enforcement (Active/Draining/Drained): It tracks the queue's lifecycle state via an atomic `status`
//     enum (`componentStatus`). This allows for robust, atomic state transitions and is crucial for graceful flow
//     updates (e.g., stopping new requests while allowing existing items to drain).
//
//  3. Exactly-Once Edge Signaling: It uses atomic operations to detect state transitions. Crucially, it ensures that
//     signals (like `queueStateSignalBecameEmpty` or `queueStateSignalBecameDrained`) are sent exactly once when the
//     transition occurs. The transition from `componentStatusDraining` to `componentStatusDrained` acts as an atomic
//     latch for the Drained signal.
//
// # Statistical Integrity and Assumptions
//
//  1. Exclusive Access: All mutating operations on the underlying `framework.SafeQueue` MUST be performed exclusively
//     through this `managedQueue` wrapper. Direct access to the underlying queue will cause statistical drift.
//
//  2. In-Process, Stable State: The `framework.SafeQueue` implementation must be an in-process data structure (e.g., a
//     list or heap). Its state MUST NOT change through external mechanisms. For example, a queue implementation backed
//     by a distributed cache (like Redis) with its own TTL-based eviction policy would violate this assumption and lead
//     to state inconsistency, as items could be removed without notifying the `managedQueue`.
type managedQueue struct {
	// Note: There is no mutex here. Concurrency control is delegated to the underlying `framework.SafeQueue` for queue
	// operations, and atomic operations are used for stats and lifecycle status.
	queue          framework.SafeQueue
	dispatchPolicy framework.IntraFlowDispatchPolicy
	flowSpec       types.FlowSpecification
	byteSize       atomic.Uint64
	len            atomic.Uint64

	// status tracks the lifecycle state of the queue (Active, Draining, Drained).
	// It is stored as an `int32` for atomic operations.
	status atomic.Int32 // `componentStatus`

	parentCallbacks managedQueueCallbacks
	logger          logr.Logger
}

var _ contracts.ManagedQueue = &managedQueue{}

// newManagedQueue creates a new instance of a `managedQueue`.
func newManagedQueue(
	queue framework.SafeQueue,
	dispatchPolicy framework.IntraFlowDispatchPolicy,
	flowSpec types.FlowSpecification,
	logger logr.Logger,
	parentCallbacks managedQueueCallbacks,
) *managedQueue {
	mqLogger := logger.WithName("managed-queue").WithValues(
		"flowID", flowSpec.ID,
		"priority", flowSpec.Priority,
		"queueType", queue.Name(),
	)
	mq := &managedQueue{
		queue:           queue,
		dispatchPolicy:  dispatchPolicy,
		flowSpec:        flowSpec,
		parentCallbacks: parentCallbacks,
		logger:          mqLogger,
	}
	// Initialize the queue in the Active state.
	mq.status.Store(int32(componentStatusActive))
	return mq
}

// FlowQueueAccessor returns a read-only, flow-aware view of this queue.
// This accessor is primarily used by policy plugins to inspect the queue's state in a structured way.
func (mq *managedQueue) FlowQueueAccessor() framework.FlowQueueAccessor {
	return &flowQueueAccessor{mq: mq}
}

// Add wraps the underlying `framework.SafeQueue.Add` call. It first checks if the queue is active. If so, it proceeds
// with the addition and atomically updates the queue's and the parent shard's statistics.
func (mq *managedQueue) Add(item types.QueueItemAccessor) error {
	// Only StatusActive queues can accept new requests.
	status := componentStatus(mq.status.Load())
	if status != componentStatusActive {
		return fmt.Errorf("flow instance %q is not active (status: %s) and cannot accept new requests: %w",
			mq.flowSpec.ID, status, contracts.ErrFlowInstanceNotFound)
	}
	if err := mq.queue.Add(item); err != nil {
		return err
	}
	mq.propagateStatsDelta(1, int64(item.OriginalRequest().ByteSize()))
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
	mq.propagateStatsDelta(-1, -int64(removedItem.OriginalRequest().ByteSize()))
	mq.logger.V(logging.TRACE).Info("Request removed from queue", "requestID", removedItem.OriginalRequest().ID())
	return removedItem, nil
}

// Cleanup wraps the underlying `framework.SafeQueue.Cleanup` call and atomically updates statistics for all removed
// items.
func (mq *managedQueue) Cleanup(predicate framework.PredicateFunc) (cleanedItems []types.QueueItemAccessor, err error) {
	cleanedItems, err = mq.queue.Cleanup(predicate)
	if err != nil {
		return nil, err
	}
	if len(cleanedItems) == 0 {
		return cleanedItems, nil
	}
	mq.propagateStatsDeltaForRemovedItems(cleanedItems)
	mq.logger.V(logging.DEBUG).Info("Cleaned up queue", "removedItemCount", len(cleanedItems))
	return cleanedItems, nil
}

// Drain wraps the underlying `framework.SafeQueue.Drain` call and atomically updates statistics for all removed items.
func (mq *managedQueue) Drain() ([]types.QueueItemAccessor, error) {
	drainedItems, err := mq.queue.Drain()
	if err != nil {
		return nil, err
	}
	if len(drainedItems) == 0 {
		return drainedItems, nil
	}
	mq.propagateStatsDeltaForRemovedItems(drainedItems)
	mq.logger.V(logging.DEBUG).Info("Drained queue", "itemCount", len(drainedItems))
	return drainedItems, nil
}

// --- Pass-through and Accessor Methods ---

// Name returns the name of the queue.
func (mq *managedQueue) Name() string { return mq.queue.Name() }

// Capabilities returns the capabilities of the queue.
func (mq *managedQueue) Capabilities() []framework.QueueCapability { return mq.queue.Capabilities() }

// Len returns the number of items in the queue.
func (mq *managedQueue) Len() int { return int(mq.len.Load()) }

// ByteSize returns the total byte size of all items in the queue.
func (mq *managedQueue) ByteSize() uint64 { return mq.byteSize.Load() }

// PeekHead returns the item at the front of the queue without removing it.
func (mq *managedQueue) PeekHead() (types.QueueItemAccessor, error) { return mq.queue.PeekHead() }

// PeekTail returns the item at the back of the queue without removing it.
func (mq *managedQueue) PeekTail() (types.QueueItemAccessor, error) { return mq.queue.PeekTail() }

// Comparator returns the `framework.ItemComparator` that defines this queue's item ordering logic.
func (mq *managedQueue) Comparator() framework.ItemComparator { return mq.dispatchPolicy.Comparator() }

// --- Internal Methods (Called by `registryShard`) ---

// reactivate is an internal method called by the parent shard to transition this queue from a non-active state
// (Draining or Drained) back to Active.
func (mq *managedQueue) reactivate() {
	// Atomically transition the state back to Active.
	// We use `Swap` to get the old status for observability. Since this is called under the `FlowRegistry`'s control
	// plane lock (via the shard lock), we don't strictly need CompareAndSwap for correctness.
	oldStatus := componentStatus(mq.status.Swap(int32(componentStatusActive)))
	if oldStatus != componentStatusActive {
		// We rely on the `FlowRegistry` to re-evaluate the GC state immediately after the synchronization that caused this
		// reactivation.
		mq.logger.V(logging.DEFAULT).Info("Queue reactivated", "previousStatus", oldStatus)
	}
}

// markAsDraining is an internal method called by the parent shard to transition this queue to a draining state.
// Once draining, it will no longer accept new items via `Add`.
func (mq *managedQueue) markAsDraining() {
	// Attempt to transition from Active to Draining atomically.
	if mq.status.CompareAndSwap(int32(componentStatusActive), int32(componentStatusDraining)) {
		mq.logger.V(logging.DEFAULT).Info("Queue marked as draining")
	}

	// CRITICAL: Check if the queue is *already* empty at the moment it's marked as draining (or if it was already
	// draining and empty). If so, we must immediately attempt the transition to Drained to ensure timely GC.
	// This handles the race where the queue becomes empty just before or during being marked draining.
	if mq.Len() == 0 {
		// Attempt the transition from Draining to Drained atomically.
		if mq.status.CompareAndSwap(int32(componentStatusDraining), int32(componentStatusDrained)) {
			mq.parentCallbacks.signalQueueState(mq.flowSpec, queueStateSignalBecameDrained)
		}
	}
}

// propagateStatsDelta atomically updates the queue's own statistics and calls the parent shard's propagator.
// It also implements the core state machine logic for signaling lifecycle events to the control plane, which is
// critical for driving garbage collection.
func (mq *managedQueue) propagateStatsDelta(lenDelta, byteSizeDelta int64) {
	// The use of Add with a negative number on a `uint64` is the standard Go atomic way to perform subtraction,
	// leveraging two's complement arithmetic.
	newLen := mq.len.Add(uint64(lenDelta))

	// CRITICAL: oldLen is derived *after* the atomic operation. This is a deliberate and non-obvious pattern to prevent a
	// race condition. If we were to read the value *before* the `Add` operation (e.g., `oldLen := mq.len.Load()`), two
	// concurrent goroutines could both read the same `oldLen` value (e.g., 1) before either of them performs the `Add`.
	// If both were decrementing the length, they would both calculate `newLen` as 0 and `oldLen` as 1, causing them both
	// to incorrectly signal a transition from non-empty to empty. By deriving `oldLen` from `newLen`, we ensure that only
	// the goroutine that actually causes the transition to 0 will see the correct `oldLen` of 1.
	oldLen := newLen - uint64(lenDelta)
	mq.byteSize.Add(uint64(byteSizeDelta))

	mq.parentCallbacks.propagateStatsDelta(mq.flowSpec.Priority, lenDelta, byteSizeDelta)

	// --- State Machine Signaling Logic ---

	// Case 1: Check for Draining -> Drained transition (Exactly-Once).
	// This must happen if the length just hit zero.
	if newLen == 0 {
		// Attempt to transition from Draining to Drained atomically.
		// This acts as the exactly-once latch. If it succeeds, we are the single goroutine responsible for signaling.
		if mq.status.CompareAndSwap(int32(componentStatusDraining), int32(componentStatusDrained)) {
			mq.parentCallbacks.signalQueueState(mq.flowSpec, queueStateSignalBecameDrained)
			return // Drained is a terminal state for signaling until reactivation.
		}
	}

	// Case 2: Standard Active Queue Empty/Non-Empty transitions.
	// We only signal these if the queue is currently Active.
	// If it's Draining or Drained, these signals are irrelevant (we are waiting for the Drained signal or reactivation).
	// We must check the status again here, as it might have changed concurrently (e.g., if it was reactivated).
	if componentStatus(mq.status.Load()) == componentStatusActive {
		if oldLen > 0 && newLen == 0 {
			mq.parentCallbacks.signalQueueState(mq.flowSpec, queueStateSignalBecameEmpty)
		} else if oldLen == 0 && newLen > 0 {
			mq.parentCallbacks.signalQueueState(mq.flowSpec, queueStateSignalBecameNonEmpty)
		}
	}
}

// propagateStatsDeltaForRemovedItems calculates the total stat changes for a slice of removed items and applies them.
func (mq *managedQueue) propagateStatsDeltaForRemovedItems(items []types.QueueItemAccessor) {
	var lenDelta int64
	var byteSizeDelta int64
	for _, item := range items {
		lenDelta--
		byteSizeDelta -= int64(item.OriginalRequest().ByteSize())
	}
	mq.propagateStatsDelta(lenDelta, byteSizeDelta)
}

// --- `flowQueueAccessor` ---

// flowQueueAccessor implements `framework.FlowQueueAccessor`. It provides a read-only, policy-facing view of a
// `managedQueue`.
type flowQueueAccessor struct {
	mq *managedQueue
}

var _ framework.FlowQueueAccessor = &flowQueueAccessor{}

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
