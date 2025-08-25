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

// managedQueueCallbacks groups the callback functions that a `managedQueue` uses to communicate with its parent shard.
type managedQueueCallbacks struct {
	// propagateStatsDelta is called to propagate statistics changes (e.g., queue length, byte size) up to the parent.
	propagateStatsDelta propagateStatsDeltaFunc
	// signalQueueState is called to signal important lifecycle events, such as becoming empty, which are used by the
	// garbage collector.
	signalQueueState signalQueueStateFunc
}

// managedQueue implements `contracts.ManagedQueue`. It acts as a stateful decorator around a `framework.SafeQueue`.
//
// # Role: The Stateful Decorator
//
// Its primary responsibility is to augment a generic queue implementation with critical registry features: strictly
// consistent statistics tracking and exactly-once, edge-triggered signaling of state changes (e.g., becoming empty) to
// the control plane for garbage collection.
//
// # Concurrency Model: Hybrid Locking (Mutex + Atomics + `SafeQueue`)
//
// The `managedQueue` employs a hybrid locking strategy to guarantee strict consistency between the queue contents and
// its statistics, while maintaining high performance.
//
//  1. Mutex-Protected Writes (`sync.Mutex`): A single mutex protects all mutating operations (Add, Remove, etc.).
//     This ensures that the update to the underlying queue and the update to the internal counters occur as a single,
//     atomic transaction. This strict consistency is required for GC correctness.
//
//  2. Synchronous Propagation and Ordering: The propagation of statistics deltas (both `Len` and `ByteSize`) occurs
//     synchronously within this critical section. This strict ordering (`Add` propagation before `Remove` propagation
//     for the same item) guarantees the non-negative invariant across the entire system (Shard/Registry aggregates).
//
//  3. Lock-Free Statistics Reads (Atomics): The counters use `atomic.Int64`. This allows high-frequency accessors
//     (`Len()`, `ByteSize()`) to read the statistics without acquiring the Mutex.
//
//  4. Concurrent Reads (`SafeQueue`): Read operations (e.g., `PeekHead`/`PeekTail` used by policies) rely on the
//     underlying `framework.SafeQueue`'s concurrency control and do not acquire the `managedQueue`'s write lock.
//
// # Statistical Integrity and Invariant Protection
//
//  1. Exclusive Access: All mutations on the underlying `SafeQueue` MUST be performed exclusively through this wrapper.
//
//  2. Non-Autonomous State: The underlying queue implementation must not change state autonomously (e.g., no internal
//     TTL eviction).
//
//  3. Read-Only Proxy: To prevent policy plugins from bypassing statistics tracking, the read-only view is provided via
//     the `flowQueueAccessor` wrapper. This enforces the read-only contract at the type system level.
type managedQueue struct {
	// mu protects all mutating operations (writes) on the queue. It ensures that the underlying queue's state and the
	// atomic counters are updated atomically. Read operations (like `Peek`) do not acquire this lock.
	mu sync.Mutex

	// queue is the underlying, concurrency-safe queue implementation that this `managedQueue` decorates.
	queue framework.SafeQueue

	// dispatchPolicy is the intra-flow policy used to select items from this specific queue.
	dispatchPolicy framework.IntraFlowDispatchPolicy

	// key uniquely identifies the flow instance this queue belongs to.
	key types.FlowKey

	// Queue-level statistics. Updated under the protection of the `mu` lock, but read lock-free.
	// Guaranteed to be non-negative.
	byteSize atomic.Int64
	len      atomic.Int64

	// parentCallbacks provides the communication channels back to the parent shard.
	parentCallbacks managedQueueCallbacks
	logger          logr.Logger
}

var _ contracts.ManagedQueue = &managedQueue{}

// newManagedQueue creates a new instance of a `managedQueue`.
func newManagedQueue(
	queue framework.SafeQueue,
	dispatchPolicy framework.IntraFlowDispatchPolicy,
	key types.FlowKey,
	logger logr.Logger,
	parentCallbacks managedQueueCallbacks,
) *managedQueue {
	mqLogger := logger.WithName("managed-queue").WithValues(
		"flowKey", key,
		"queueType", queue.Name(),
	)
	mq := &managedQueue{
		queue:           queue,
		dispatchPolicy:  dispatchPolicy,
		key:             key,
		parentCallbacks: parentCallbacks,
		logger:          mqLogger,
	}
	return mq
}

// FlowQueueAccessor returns a read-only, flow-aware view of this queue.
// This accessor is primarily used by policy plugins to inspect the queue's state in a structured way.
func (mq *managedQueue) FlowQueueAccessor() framework.FlowQueueAccessor {
	return &flowQueueAccessor{mq: mq}
}

// Add wraps the underlying `framework.SafeQueue.Add` call and atomically updates the queue's and the parent shard's
// statistics.
func (mq *managedQueue) Add(item types.QueueItemAccessor) error {
	mq.mu.Lock()
	defer mq.mu.Unlock()

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
	mq.mu.Lock()
	defer mq.mu.Unlock()

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
	mq.mu.Lock()
	defer mq.mu.Unlock()

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
	mq.mu.Lock()
	defer mq.mu.Unlock()

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

func (mq *managedQueue) Name() string                               { return mq.queue.Name() }
func (mq *managedQueue) Capabilities() []framework.QueueCapability  { return mq.queue.Capabilities() }
func (mq *managedQueue) Len() int                                   { return int(mq.len.Load()) }
func (mq *managedQueue) ByteSize() uint64                           { return uint64(mq.byteSize.Load()) }
func (mq *managedQueue) PeekHead() (types.QueueItemAccessor, error) { return mq.queue.PeekHead() }
func (mq *managedQueue) PeekTail() (types.QueueItemAccessor, error) { return mq.queue.PeekTail() }
func (mq *managedQueue) Comparator() framework.ItemComparator       { return mq.dispatchPolicy.Comparator() }

// --- Internal Methods ---

// propagateStatsDelta updates the queue's statistics, signals emptiness events, and propagates the delta.
//
// Invariant Check: This function panics if a statistic becomes negative. Because all updates are protected by the
// `managedQueue.mu` lock, a negative value indicates a critical logic error (e.g., double-counting a removal).
// This check enforces the non-negative invariant locally, which mathematically guarantees that the aggregated
// statistics (Shard/Registry level) also remain non-negative.
func (mq *managedQueue) propagateStatsDelta(lenDelta, byteSizeDelta int64) {
	// Note: We rely on the caller (Add/Remove/etc.) to hold the `managedQueue.mu` lock.

	newLen := mq.len.Add(lenDelta)
	if newLen < 0 {
		panic(fmt.Sprintf("invariant violation: managedQueue length for flow %s became negative (%d)", mq.key, newLen))
	}
	mq.byteSize.Add(byteSizeDelta)

	// Evaluate and signal our own state change *before* propagating the statistics to the parent shard. This ensures a
	// strict, bottom-up event ordering, preventing race conditions where the parent might process its own state change
	// before the child's signal is handled.
	// 1. Evaluate GC signals based on the strictly consistent local state.
	// 2. Propagate the delta up to the Shard/Registry. This propagation is lock-free and eventually consistent.
	mq.evaluateEmptinessState(newLen-lenDelta, newLen)
	mq.parentCallbacks.propagateStatsDelta(mq.key.Priority, lenDelta, byteSizeDelta)
}

// evaluateEmptinessState checks if the queue has transitioned between non-empty <-> empty and signals the parent if so.
func (mq *managedQueue) evaluateEmptinessState(oldLen, newLen int64) {
	if oldLen > 0 && newLen == 0 {
		mq.parentCallbacks.signalQueueState(mq.key, queueStateSignalBecameEmpty)
	} else if oldLen == 0 && newLen > 0 {
		mq.parentCallbacks.signalQueueState(mq.key, queueStateSignalBecameNonEmpty)
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

// flowQueueAccessor implements `framework.FlowQueueAccessor`. It provides a read-only, policy-facing view.
//
// # Role: The Read-Only Proxy
//
// This wrapper protects system invariants. It acts as a proxy that exposes only the read-only methods.
// This prevents policy plugins from using type assertions to access the concrete `*managedQueue` and calling mutation
// methods, which would bypass statistics tracking and signaling.
type flowQueueAccessor struct {
	mq *managedQueue
}

var _ framework.FlowQueueAccessor = &flowQueueAccessor{}

func (a *flowQueueAccessor) Name() string                               { return a.mq.Name() }
func (a *flowQueueAccessor) Capabilities() []framework.QueueCapability  { return a.mq.Capabilities() }
func (a *flowQueueAccessor) Len() int                                   { return a.mq.Len() }
func (a *flowQueueAccessor) ByteSize() uint64                           { return a.mq.ByteSize() }
func (a *flowQueueAccessor) PeekHead() (types.QueueItemAccessor, error) { return a.mq.PeekHead() }
func (a *flowQueueAccessor) PeekTail() (types.QueueItemAccessor, error) { return a.mq.PeekTail() }
func (a *flowQueueAccessor) Comparator() framework.ItemComparator       { return a.mq.Comparator() }
func (a *flowQueueAccessor) FlowKey() types.FlowKey                     { return a.mq.key }
