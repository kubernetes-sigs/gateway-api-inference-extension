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
	"sync"
	"sync/atomic"

	"github.com/go-logr/logr"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// managedQueue implements `contracts.ManagedQueue`. It acts as a stateful decorator around a `framework.SafeQueue`.
//
// # Role: The Stateful Statistics Decorator
//
// Its sole responsibility is to augment a generic queue with strictly consistent, atomic statistics tracking.
// All mutating operations (`Add`, `Remove`, etc.) are wrapped to ensure that the queue's internal state and its
// externally visible statistics (`Len`, `ByteSize`) are always perfectly synchronized.
//
// # Concurrency Model: Mutex for Writes, Atomics for Reads
//
// The `managedQueue` employs a hybrid locking strategy to guarantee strict consistency while maintaining high read
// performance.
//
//  1. Mutex-Protected Writes (`sync.Mutex`): A single mutex protects all mutating operations.
//     This ensures the update to the underlying queue and the update to the internal counters occur as a single, atomic
//     transaction.
//  2. Synchronous Propagation: Statistics deltas are propagated synchronously within this critical section,
//     guaranteeing a non-negativity invariant across the entire system (Shard/Registry aggregates).
//  3. Lock-Free Reads (Atomics): The counters use `atomic.Int64`, allowing high-frequency accessors (`Len()`,
//     `ByteSize()`) to read statistics without acquiring the mutex.
//
// # Invariant Protection
//
// To guarantee statistical integrity, the following invariants must be upheld:
//  1. Exclusive Access: All mutations on the underlying `SafeQueue` MUST be performed exclusively through this wrapper.
//  2. Non-Autonomous State: The underlying queue must not change state autonomously (e.g., no internal TTL eviction).
type managedQueue struct {
	// --- Immutable Identity & Dependencies (set at construction) ---
	key            types.FlowKey
	dispatchPolicy framework.IntraFlowDispatchPolicy
	logger         logr.Logger

	// onStatsDelta is the callback used to propagate statistics changes up to the parent shard.
	onStatsDelta propagateStatsDeltaFunc
	// isDraining is a callback that checks the lifecycle state of the parent shard, allowing this queue to reject new
	// work when the shard is being decommissioned.
	isDraining func() bool

	// --- State Protected by `mu` ---

	// mu protects all mutating operations. It ensures that any changes to the underlying `queue` and the updates to the
	// atomic counters occur as a single, atomic transaction.
	mu sync.Mutex
	// queue is the underlying, concurrency-safe queue implementation that this `managedQueue` decorates.
	// Its state must only be modified while holding `mu`.
	queue framework.SafeQueue

	// --- Concurrent-Safe State (Atomics) ---

	// Queue-level statistics.
	// These are written under the protection of `mu` but can be read lock-free at any time using atomic operations.
	// They are guaranteed to be non-negative.
	byteSize atomic.Int64
	len      atomic.Int64
}

var _ contracts.ManagedQueue = &managedQueue{}

// newManagedQueue creates a new instance of a `managedQueue`.
func newManagedQueue(
	queue framework.SafeQueue,
	dispatchPolicy framework.IntraFlowDispatchPolicy,
	key types.FlowKey,
	logger logr.Logger,
	onStatsDelta propagateStatsDeltaFunc,
	isDraining func() bool,
) *managedQueue {
	mqLogger := logger.WithName("managed-queue").WithValues(
		"flowKey", key,
		"queueType", queue.Name(),
	)
	return &managedQueue{
		queue:          queue,
		dispatchPolicy: dispatchPolicy,
		key:            key,
		onStatsDelta:   onStatsDelta,
		logger:         mqLogger,
		isDraining:     isDraining,
	}
}

// FlowQueueAccessor returns a read-only, flow-aware view of this queue.
func (mq *managedQueue) FlowQueueAccessor() framework.FlowQueueAccessor {
	return &flowQueueAccessor{mq: mq}
}

// Add performs an atomic check on the parent shard's lifecycle state before adding the item to the underlying queue.
// This is the critical enforcement point that prevents new requests from entering a draining shard.
func (mq *managedQueue) Add(item types.QueueItemAccessor) error {
	mq.mu.Lock()
	defer mq.mu.Unlock()

	if mq.isDraining() {
		return contracts.ErrShardDraining
	}
	mq.queue.Add(item)

	mq.propagateStatsDeltaLocked(1, int64(item.OriginalRequest().ByteSize()))
	mq.logger.V(logging.TRACE).Info("Request added to queue", "requestID", item.OriginalRequest().ID())
	return nil
}

// Remove wraps the underlying framework.SafeQueue.Remove and updates statistics.
func (mq *managedQueue) Remove(handle types.QueueItemHandle) (types.QueueItemAccessor, error) {
	mq.mu.Lock()
	defer mq.mu.Unlock()

	removedItem, err := mq.queue.Remove(handle)
	if err != nil {
		return nil, err
	}
	mq.propagateStatsDeltaLocked(-1, -int64(removedItem.OriginalRequest().ByteSize()))
	mq.logger.V(logging.TRACE).Info("Request removed from queue", "requestID", removedItem.OriginalRequest().ID())
	return removedItem, nil
}

// Cleanup wraps the underlying framework.SafeQueue.Cleanup and updates statistics.
func (mq *managedQueue) Cleanup(predicate framework.PredicateFunc) []types.QueueItemAccessor {
	mq.mu.Lock()
	defer mq.mu.Unlock()

	cleanedItems := mq.queue.Cleanup(predicate)
	if len(cleanedItems) == 0 {
		return nil
	}
	mq.propagateStatsDeltaForRemovedItemsLocked(cleanedItems)
	mq.logger.V(logging.DEBUG).Info("Cleaned up queue", "removedItemCount", len(cleanedItems))
	return cleanedItems
}

// Drain wraps the underlying framework.SafeQueue.Drain and updates statistics.
func (mq *managedQueue) Drain() []types.QueueItemAccessor {
	mq.mu.Lock()
	defer mq.mu.Unlock()

	drainedItems := mq.queue.Drain()
	if len(drainedItems) == 0 {
		return nil
	}
	mq.propagateStatsDeltaForRemovedItemsLocked(drainedItems)
	mq.logger.V(logging.DEBUG).Info("Drained queue", "itemCount", len(drainedItems))
	return drainedItems
}

// Len returns the current number of items in the queue.
func (mq *managedQueue) Len() int {
	return int(mq.len.Load())
}

// ByteSize returns the current total byte size of all items in the queue.
func (mq *managedQueue) ByteSize() uint64 {
	return uint64(mq.byteSize.Load())
}

// propagateStatsDeltaLocked updates the queue's statistics and propagates the delta to the parent shard.
// It must be called while holding the `managedQueue.mu` lock.
//
// Invariant Check: This function panics if a statistic becomes negative. This enforces the non-negative invariant
// locally, which mathematically guarantees that the aggregated statistics (Shard/Registry level) also remain
// non-negative.
func (mq *managedQueue) propagateStatsDeltaLocked(lenDelta, byteSizeDelta int64) {
	newLen := mq.len.Add(lenDelta)
	if newLen < 0 {
		panic(fmt.Sprintf("invariant violation: managedQueue length for flow %s became negative (%d)", mq.key, newLen))
	}
	mq.byteSize.Add(byteSizeDelta)

	// Propagate the delta up to the parent shard. This propagation is lock-free and eventually consistent.
	mq.onStatsDelta(mq.key.Priority, lenDelta, byteSizeDelta)
}

// propagateStatsDeltaForRemovedItemsLocked calculates the total stat changes for a slice of removed items and applies
// them. It must be called while holding the `managedQueue.mu` lock.
func (mq *managedQueue) propagateStatsDeltaForRemovedItemsLocked(items []types.QueueItemAccessor) {
	var lenDelta int64
	var byteSizeDelta int64
	for _, item := range items {
		lenDelta--
		byteSizeDelta -= int64(item.OriginalRequest().ByteSize())
	}
	mq.propagateStatsDeltaLocked(lenDelta, byteSizeDelta)
}

// --- `flowQueueAccessor` ---

// flowQueueAccessor implements `framework.FlowQueueAccessor`. It provides a read-only, policy-facing view.
//
// # Role: The Read-Only Proxy
//
// This wrapper protects system invariants. It acts as a proxy that exposes only the read-only methods.
// This prevents policy plugins from using type assertions to access the concrete `*managedQueue` and calling mutation
// methods, which would bypass statistics tracking.
type flowQueueAccessor struct {
	mq *managedQueue
}

var _ framework.FlowQueueAccessor = &flowQueueAccessor{}

// --- Read-only pass-through methods to the underlying SafeQueue ---
func (a *flowQueueAccessor) Name() string { return a.mq.queue.Name() }
func (a *flowQueueAccessor) Capabilities() []framework.QueueCapability {
	return a.mq.queue.Capabilities()
}
func (a *flowQueueAccessor) PeekHead() types.QueueItemAccessor { return a.mq.queue.PeekHead() }
func (a *flowQueueAccessor) PeekTail() types.QueueItemAccessor { return a.mq.queue.PeekTail() }

// --- Read-only methods from the managedQueue wrapper ---
func (a *flowQueueAccessor) Len() int         { return a.mq.Len() }
func (a *flowQueueAccessor) ByteSize() uint64 { return a.mq.ByteSize() }
func (a *flowQueueAccessor) Comparator() framework.ItemComparator {
	return a.mq.dispatchPolicy.Comparator()
}
func (a *flowQueueAccessor) FlowKey() types.FlowKey { return a.mq.key }
