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

// Package registry provides the concrete implementation of the Flow Registry, which is the stateful control plane for
// the Flow Control system. It implements the service interfaces defined in the `contracts` package.
//
// This initial version includes the implementation of the `contracts.ManagedQueue`, a stateful wrapper that adds atomic
// statistics tracking to a `framework.SafeQueue`.
package registry

import (
	"sync/atomic"

	"github.com/go-logr/logr"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

type parentStatsReconciler func(lenDelta, byteSizeDelta int64)

// managedQueue implements `contracts.ManagedQueue`. It wraps a `framework.SafeQueue` and is responsible for maintaining
// accurate, atomically-updated statistics that are aggregated at the shard level.
//
// # Statistical Integrity
//
// For performance, `managedQueue` maintains its own `len` and `byteSize` fields using atomic operations. This provides
// O(1) access for the parent `contracts.RegistryShard`'s aggregated statistics without needing to lock the underlying
// queue.
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
	reconcileShardStats parentStatsReconciler
	logger              logr.Logger
}

// newManagedQueue creates a new instance of a `managedQueue`.
func newManagedQueue(
	queue framework.SafeQueue,
	dispatchPolicy framework.IntraFlowDispatchPolicy,
	flowSpec types.FlowSpecification,
	logger logr.Logger,
	reconcileShardStats func(lenDelta, byteSizeDelta int64),
) *managedQueue {
	mqLogger := logger.WithName("managed-queue").WithValues(
		"flowID", flowSpec.ID,
		"priority", flowSpec.Priority,
		"queueType", queue.Name(),
	)
	return &managedQueue{
		queue:               queue,
		dispatchPolicy:      dispatchPolicy,
		flowSpec:            flowSpec,
		reconcileShardStats: reconcileShardStats,
		logger:              mqLogger,
	}
}

// FlowQueueAccessor returns a new `flowQueueAccessor` instance.
func (mq *managedQueue) FlowQueueAccessor() framework.FlowQueueAccessor {
	return &flowQueueAccessor{mq: mq}
}

func (mq *managedQueue) Add(item types.QueueItemAccessor) error {
	if err := mq.queue.Add(item); err != nil {
		return err
	}
	mq.reconcileStats(1, int64(item.OriginalRequest().ByteSize()))
	return nil
}

func (mq *managedQueue) Remove(handle types.QueueItemHandle) (types.QueueItemAccessor, error) {
	removedItem, err := mq.queue.Remove(handle)
	if err != nil {
		return nil, err
	}
	mq.reconcileStats(-1, -int64(removedItem.OriginalRequest().ByteSize()))
	// TODO: If mq.len.Load() == 0, signal shard for optimistic instance cleanup.
	return removedItem, nil
}

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
	// TODO: If mq.len.Load() == 0, signal shard for optimistic instance cleanup.
	return cleanedItems, nil
}

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
	// TODO: If mq.len.Load() == 0, signal shard for optimistic instance cleanup.
	return drainedItems, nil
}

// reconcileStats atomically updates the queue's own statistics and calls the parent shard's reconciler to ensure
// aggregated stats remain consistent.
func (mq *managedQueue) reconcileStats(lenDelta, byteSizeDelta int64) {
	// The use of Add with a negative number on a Uint64 is the standard Go atomic way to perform subtraction, leveraging
	// two's complement arithmetic.
	mq.len.Add(uint64(lenDelta))
	mq.byteSize.Add(uint64(byteSizeDelta))
	mq.reconcileShardStats(lenDelta, byteSizeDelta)
}

// --- Pass-through and accessor methods ---

func (mq *managedQueue) Name() string                               { return mq.queue.Name() }
func (mq *managedQueue) Capabilities() []framework.QueueCapability  { return mq.queue.Capabilities() }
func (mq *managedQueue) Len() int                                   { return int(mq.len.Load()) }
func (mq *managedQueue) ByteSize() uint64                           { return mq.byteSize.Load() }
func (mq *managedQueue) PeekHead() (types.QueueItemAccessor, error) { return mq.queue.PeekHead() }
func (mq *managedQueue) PeekTail() (types.QueueItemAccessor, error) { return mq.queue.PeekTail() }
func (mq *managedQueue) Comparator() framework.ItemComparator       { return mq.dispatchPolicy.Comparator() }

var _ contracts.ManagedQueue = &managedQueue{}

// --- flowQueueAccessor ---

// flowQueueAccessor implements `framework.FlowQueueAccessor`. It provides a read-only, policy-facing view of a
// `managedQueue`.
type flowQueueAccessor struct {
	mq *managedQueue
}

func (a *flowQueueAccessor) Name() string                               { return a.mq.Name() }
func (a *flowQueueAccessor) Capabilities() []framework.QueueCapability  { return a.mq.Capabilities() }
func (a *flowQueueAccessor) Len() int                                   { return a.mq.Len() }
func (a *flowQueueAccessor) ByteSize() uint64                           { return a.mq.ByteSize() }
func (a *flowQueueAccessor) PeekHead() (types.QueueItemAccessor, error) { return a.mq.PeekHead() }
func (a *flowQueueAccessor) PeekTail() (types.QueueItemAccessor, error) { return a.mq.PeekTail() }
func (a *flowQueueAccessor) Comparator() framework.ItemComparator       { return a.mq.Comparator() }
func (a *flowQueueAccessor) FlowSpec() types.FlowSpecification          { return a.mq.flowSpec }

var _ framework.FlowQueueAccessor = &flowQueueAccessor{}
