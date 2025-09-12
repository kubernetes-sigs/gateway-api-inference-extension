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

package internal

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/utils/clock"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// maxCleanupWorkers caps the number of concurrent workers for background cleanup tasks. This prevents a single shard
// from overwhelming the Go scheduler with too many goroutines.
const maxCleanupWorkers = 4

// ErrProcessorBusy is a sentinel error returned by a the processor's `Submit` method.
// It indicates that the processor's internal buffer is momentarily full and cannot accept new work.
// This is used as a signal for the `controller.FlowController`'s "fast failover" logic.
var ErrProcessorBusy = errors.New("shard processor is busy")

// ShardProcessor is the core worker of the `controller.FlowController`.
// It is paired one-to-one with a `contracts.RegistryShard` instance and is responsible for all request lifecycle
// operations on that shard. It acts as the "data plane" worker that executes against the concurrent-safe state provided
// by its shard.
//
// # Concurrency Model: The Single-Writer Actor
//
// To ensure correctness and high performance, the processor uses a single-goroutine, actor-based model. The main `Run`
// loop is the sole "writer" for all state-mutating operations, particularly enqueueing. This makes complex transactions
// inherently atomic without coarse-grained locks.
//
// # Concurrency Guarantees
//
//  1. Safe Enqueueing: The "check-then-act" sequence for capacity is safe because it is only ever performed by the
//     single `Run` goroutine.
//  2. Idempotent Finalization: The primary internal race condition is between the main `dispatchCycle` and the
//     background `runExpiryCleanup` goroutine, both of which might try to finalize an item. This is resolved by the
//     `FlowItem.Finalize` method, which uses `sync.Once` to guarantee that only the first attempt to finalize an item
//     succeeds.
type ShardProcessor struct {
	shard                 contracts.RegistryShard
	dispatchFilter        BandFilter
	clock                 clock.Clock
	expiryCleanupInterval time.Duration
	logger                logr.Logger

	// enqueueChan is the entry point for new requests to be processed by this shard's `Run` loop.
	enqueueChan chan *FlowItem
	// wg is used to wait for background tasks like expiry cleanup to complete on shutdown.
	wg             sync.WaitGroup
	isShuttingDown atomic.Bool
	shutdownOnce   sync.Once
}

// NewShardProcessor creates a new `ShardProcessor` instance.
func NewShardProcessor(
	shard contracts.RegistryShard,
	dispatchFilter BandFilter,
	clock clock.Clock,
	expiryCleanupInterval time.Duration,
	enqueueChannelBufferSize int,
	logger logr.Logger,
) *ShardProcessor {
	return &ShardProcessor{
		shard:                 shard,
		dispatchFilter:        dispatchFilter,
		clock:                 clock,
		expiryCleanupInterval: expiryCleanupInterval,
		logger:                logger,
		// A buffered channel decouples the processor from the distributor, allowing for a fast, asynchronous handoff of new
		// requests.
		enqueueChan: make(chan *FlowItem, enqueueChannelBufferSize),
	}
}

// Submit attempts a non-blocking handoff of an item to the processor's internal channel for asynchronous processing.
//
// It returns nil if the item was accepted by the processor, or if the processor is shutting down (in which case the
// item is immediately finalized with a shutdown error). In both cases, a nil return means the item's lifecycle has been
// handled by this processor and the caller should not retry.
// It returns `ErrProcessorBusy` if the processor's channel is momentarily full, signaling that the caller should try
// another processor.
func (sp *ShardProcessor) Submit(item *FlowItem) error {
	if sp.isShuttingDown.Load() {
		item.Finalize(types.QueueOutcomeRejectedOther,
			fmt.Errorf("%w: %w", types.ErrRejected, types.ErrFlowControllerNotRunning))
		return nil // Success from the caller's perspective; the item is terminal.
	}

	select {
	case sp.enqueueChan <- item:
		return nil // Success
	default:
		// The channel buffer is full, signaling transient backpressure.
		return ErrProcessorBusy
	}
}

// SubmitOrBlock performs a blocking submission of an item to the processor's internal channel.
// It will wait until either the submission succeeds or the provided context is cancelled.
//
// This method is the fallback used by the distributor when all processors are busy, providing graceful backpressure
// instead of immediate rejection.
//
// It returns the `ctx.Err()` if the context is cancelled during the wait.
func (sp *ShardProcessor) SubmitOrBlock(ctx context.Context, item *FlowItem) error {
	if sp.isShuttingDown.Load() {
		// Here, we return an error because the caller, expecting to block, was prevented from doing so by the shutdown.
		// This is a failure of the operation.
		item.Finalize(types.QueueOutcomeRejectedOther,
			fmt.Errorf("%w: %w", types.ErrRejected, types.ErrFlowControllerNotRunning))
		return types.ErrFlowControllerNotRunning
	}

	select {
	case sp.enqueueChan <- item:
		return nil // Success
	case <-ctx.Done():
		// The caller's context was cancelled while we were blocked.
		return ctx.Err()
	}
}

// Run is the main operational loop for the shard processor. It must be run as a goroutine.
// It uses a `select` statement to interleave accepting new requests with dispatching existing ones, balancing
// responsiveness with throughput.
func (sp *ShardProcessor) Run(ctx context.Context) {
	sp.logger.V(logutil.DEFAULT).Info("Shard processor run loop starting.")
	defer sp.logger.V(logutil.DEFAULT).Info("Shard processor run loop stopped.")

	sp.wg.Add(1)
	go sp.runExpiryCleanup(ctx)

	// This is the main worker loop. It continuously processes incoming requests and dispatches queued requests until the
	// context is cancelled. The `select` statement has three cases:
	//
	//  1. Context Cancellation: The highest priority is shutting down. If the context's `Done` channel is closed, the
	//     loop will drain all queues and exit. This is the primary exit condition.
	//  2. New Item Arrival: If an item is available on `enqueueChan`, it will be processed. This ensures that the
	//     processor is responsive to new work.
	//  3. Default (Dispatch): If neither of the above cases is ready, the `default` case executes, ensuring the loop is
	//     non-blocking. It continuously attempts to dispatch items from the existing backlog, preventing starvation and
	//     ensuring queues are drained.
	for {
		select {
		case <-ctx.Done():
			sp.shutdown()
			sp.wg.Wait()
			return
		case item, ok := <-sp.enqueueChan:
			if !ok { // Should not happen in practice, but is a clean shutdown signal.
				sp.shutdown()
				sp.wg.Wait()
				return
			}
			// This is a safeguard against logic errors in the distributor.
			if item == nil {
				sp.logger.Error(nil, "Logic error: nil item received on shard processor enqueue channel, ignoring.")
				continue
			}
			sp.enqueue(item)
			sp.dispatchCycle(ctx)
		default:
			// If no new items are arriving, continuously try to dispatch from the backlog.
			if !sp.dispatchCycle(ctx) {
				// If no work was done, yield to the scheduler to prevent a tight, busy-loop when idle, while still allowing for
				// immediate rescheduling.
				runtime.Gosched()
			}
		}
	}
}

// enqueue is responsible for adding a new item to its designated queue. It is always run from the single main `Run`
// goroutine, which makes its multi-step "check-then-act" logic for capacity management inherently atomic and safe from
// race conditions.
func (sp *ShardProcessor) enqueue(item *FlowItem) {
	req := item.OriginalRequest()
	key := req.FlowKey()

	logger := log.FromContext(req.Context()).WithName("enqueue").WithValues(
		"flowKey", key,
		"flowID", key.ID,
		"priority", key.Priority,
		"reqID", req.ID(),
		"reqByteSize", req.ByteSize(),
	)

	managedQ, err := sp.shard.ManagedQueue(key)
	if err != nil {
		finalErr := fmt.Errorf("configuration error: failed to get queue for flow key %s: %w", key, err)
		logger.Error(finalErr, "Rejecting item.")
		item.Finalize(types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, finalErr))
		return
	}

	band, err := sp.shard.PriorityBandAccessor(key.Priority)
	if err != nil {
		finalErr := fmt.Errorf("configuration error: failed to get priority band for priority %d: %w", key.Priority, err)
		logger.Error(finalErr, "Rejecting item.")
		item.Finalize(types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, finalErr))
		return
	}
	logger = logger.WithValues("priorityName", band.PriorityName())

	if !sp.hasCapacity(key.Priority, req.ByteSize()) {
		// This is an expected outcome, not a system error. Log at the default level with rich context.
		stats := sp.shard.Stats()
		bandStats := stats.PerPriorityBandStats[key.Priority]
		logger.V(logutil.DEFAULT).Info("Rejecting request, queue at capacity",
			"outcome", types.QueueOutcomeRejectedCapacity,
			"shardTotalBytes", stats.TotalByteSize,
			"shardCapacityBytes", stats.TotalCapacityBytes,
			"bandTotalBytes", bandStats.ByteSize,
			"bandCapacityBytes", bandStats.CapacityBytes,
		)
		item.Finalize(types.QueueOutcomeRejectedCapacity, fmt.Errorf("%w: %w", types.ErrRejected, types.ErrQueueAtCapacity))
		return
	}

	// This is an optimistic check to prevent a needless add/remove cycle for an item that was finalized (e.g., context
	// cancelled) during the handoff to this processor. A race condition still exists where an item can be finalized
	// after this check but before the `Add` call completes.
	//
	// This is considered acceptable because:
	//  1. The race window is extremely small.
	//  2. The background `runExpiryCleanup` goroutine acts as the ultimate guarantor of correctness, as it will
	//     eventually find and evict any finalized item that slips through this check and is added to a queue.
	if item.isFinalized() {
		finalState := item.finalState
		outcome, err := finalState.Outcome, finalState.Err
		logger.V(logutil.VERBOSE).Info("Item finalized before adding to queue, ignoring.", "outcome", outcome, "err", err)
		return
	}

	// This is the point of commitment. After this call, the item is officially in the queue and is the responsibility of
	// the dispatch or cleanup loops to finalize.
	if err := managedQ.Add(item); err != nil {
		finalErr := fmt.Errorf("failed to add item to queue for flow key %s: %w", key, err)
		logger.Error(finalErr, "Rejecting item.")
		item.Finalize(types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, finalErr))
		return
	}
	logger.V(logutil.TRACE).Info("Item enqueued.")
}

// hasCapacity checks if the shard and the specific priority band have enough capacity to accommodate an item of a given
// size. This check is only safe because it is called from the single-writer `enqueue` method.
func (sp *ShardProcessor) hasCapacity(priority int, itemByteSize uint64) bool {
	if itemByteSize == 0 {
		return true
	}
	stats := sp.shard.Stats()
	if stats.TotalCapacityBytes > 0 && stats.TotalByteSize+itemByteSize > stats.TotalCapacityBytes {
		return false
	}
	bandStats, ok := stats.PerPriorityBandStats[priority]
	if !ok {
		// This should not happen if the registry is consistent, but we fail closed just in case.
		return false
	}
	return bandStats.ByteSize+itemByteSize <= bandStats.CapacityBytes
}

// dispatchCycle attempts to dispatch a single item by iterating through all priority bands from highest to lowest.
// It applies the configured policies for each band to select an item and then attempts to dispatch it.
// It returns true if an item was successfully dispatched, and false otherwise, indicating that no work was done in this
// cycle.
//
// # Error Handling Philosophy: Failure Isolation & Work Conservation
//
// The engine is designed to be resilient to failures in individual policies or transient errors within a specific flow.
// The core principle is failure isolation. A problem in one priority band must not be allowed to halt processing for
// other, healthy bands.
//
// To achieve this, any error encountered during the selection or dispatch process for a given priority band is treated
// as a non-critical failure for that band, in this cycle. The processor will log the error for observability and then
// immediately continue its attempt to find work in the next-lower priority band. This promotes work conservation and
// maximizes system throughput even in the presence of partial failures.
func (sp *ShardProcessor) dispatchCycle(ctx context.Context) bool {
	baseLogger := sp.logger.WithName("dispatchCycle")

	// FUTURE EXTENSION POINT: The iteration over priority bands is currently a simple, strict-priority loop.
	// This could be abstracted into a third policy tier (e.g., an `InterBandDispatchPolicy`) if more complex scheduling
	// between bands, such as Weighted Fair Queuing (WFQ), is ever required. For now, strict priority is sufficient.
	for _, priority := range sp.shard.AllOrderedPriorityLevels() {
		originalBand, err := sp.shard.PriorityBandAccessor(priority)
		if err != nil {
			baseLogger.Error(err, "Failed to get PriorityBandAccessor, skipping band", "priority", priority)
			continue
		}
		logger := baseLogger.WithValues("priority", priority, "priorityName", originalBand.PriorityName())

		// Apply the configured filter to get a view of only the dispatchable flows.
		dispatchableBand, shouldPause := sp.dispatchFilter(ctx, originalBand, logger)
		if shouldPause {
			return false // A global gate told us to stop the entire cycle.
		}
		if dispatchableBand == nil {
			// A nil return from the filter indicates the fast path: no filtering was needed.
			dispatchableBand = originalBand
		}

		// Pass the (potentially filtered) band to the policies.
		item, err := sp.selectItem(dispatchableBand, logger)
		if err != nil {
			logger.Error(err, "Failed to select item, skipping priority band for this cycle")
			continue
		}
		if item == nil {
			// This is the common case where a priority band has no items to dispatch.
			logger.V(logutil.TRACE).Info("No item selected by dispatch policies, skipping band")
			continue
		}
		logger = logger.WithValues(
			"flowKey", item.OriginalRequest().FlowKey(),
			"flowID", item.OriginalRequest().FlowKey().ID,
			"flowPriority", item.OriginalRequest().FlowKey().Priority,
			"reqID", item.OriginalRequest().ID(),
			"reqByteSize", item.OriginalRequest().ByteSize())

		if err := sp.dispatchItem(item, logger); err != nil {
			logger.Error(err, "Failed to dispatch item, skipping priority band for this cycle")
			continue
		}
		// A successful dispatch occurred, so we return true to signal that work was done.
		return true
	}
	// No items were dispatched in this cycle across all priority bands.
	return false
}

// selectItem applies the configured inter- and intra-flow dispatch policies to select a single item from a priority
// band.
func (sp *ShardProcessor) selectItem(
	band framework.PriorityBandAccessor,
	logger logr.Logger,
) (types.QueueItemAccessor, error) {
	interP, err := sp.shard.InterFlowDispatchPolicy(band.Priority())
	if err != nil {
		return nil, fmt.Errorf("could not get InterFlowDispatchPolicy: %w", err)
	}
	queue, err := interP.SelectQueue(band)
	if err != nil {
		return nil, fmt.Errorf("InterFlowDispatchPolicy %q failed to select queue: %w", interP.Name(), err)
	}
	if queue == nil {
		logger.V(logutil.TRACE).Info("No queue selected by InterFlowDispatchPolicy")
		return nil, nil
	}
	key := queue.FlowKey()
	logger = logger.WithValues(
		"selectedFlowKey", key,
		"selectedFlowID", key.ID,
		"selectedFlowPriority", key.Priority)
	intraP, err := sp.shard.IntraFlowDispatchPolicy(key)
	if err != nil {
		return nil, fmt.Errorf("could not get IntraFlowDispatchPolicy for flow %s: %w", key, err)
	}
	item, err := intraP.SelectItem(queue)
	if err != nil {
		return nil, fmt.Errorf("IntraFlowDispatchPolicy %q failed to select item for flow %s: %w", intraP.Name(), key, err)
	}
	if item == nil {
		logger.V(logutil.TRACE).Info("No item selected by IntraFlowDispatchPolicy")
		return nil, nil
	}
	return item, nil
}

// dispatchItem handles the final steps of dispatching an item after it has been selected by policies. This includes
// removing it from its queue and finalizing its outcome.
func (sp *ShardProcessor) dispatchItem(itemAcc types.QueueItemAccessor, logger logr.Logger) error {
	logger = logger.WithName("dispatchItem")

	req := itemAcc.OriginalRequest()
	// We must look up the queue by its specific priority, as a flow might have draining queues at other levels.
	managedQ, err := sp.shard.ManagedQueue(req.FlowKey())
	if err != nil {
		return fmt.Errorf("failed to get ManagedQueue for flow %s: %w", req.FlowKey(), err)
	}

	// The core mutation: remove the item from the queue.
	removedItemAcc, err := managedQ.Remove(itemAcc.Handle())
	if err != nil {
		// This can happen benignly if the item was already removed by the expiry cleanup loop between the time it was
		// selected by the policy and the time this function is called.
		logger.V(logutil.VERBOSE).Info("Item already removed from queue, likely by expiry cleanup", "err", err)
		return fmt.Errorf("failed to remove item %q from queue for flow %s: %w", req.ID(), req.FlowKey(), err)
	}

	// Final check for expiry/cancellation right before dispatch.
	removedItem := removedItemAcc.(*FlowItem)
	isExpired, outcome, expiryErr := checkItemExpiry(removedItem, sp.clock.Now())
	if isExpired {
		// Ensure we always have a non-nil error to wrap for consistent logging and error handling.
		finalErr := expiryErr
		if finalErr == nil {
			finalErr = errors.New("item finalized before dispatch")
		}
		logger.V(logutil.VERBOSE).Info("Item expired at time of dispatch, evicting", "outcome", outcome,
			"err", finalErr)
		removedItem.Finalize(outcome, fmt.Errorf("%w: %w", types.ErrEvicted, finalErr))
		// Return an error to signal that the dispatch did not succeed.
		return fmt.Errorf("item %q expired before dispatch: %w", req.ID(), finalErr)
	}

	// Finalize the item as dispatched.
	removedItem.Finalize(types.QueueOutcomeDispatched, nil)
	logger.V(logutil.TRACE).Info("Item dispatched.")
	return nil
}

// checkItemExpiry provides the authoritative check to determine if an item should be evicted due to TTL expiry or
// context cancellation.
//
// It serves as a safeguard against race conditions. Its first action is to check if the item has already been finalized
// by a competing goroutine (e.g., the cleanup loop finalizing an item the dispatch loop is trying to process).=
// This ensures that the final outcome is decided exactly once.
func checkItemExpiry(
	itemAcc types.QueueItemAccessor,
	now time.Time,
) (isExpired bool, outcome types.QueueOutcome, err error) {
	item := itemAcc.(*FlowItem)

	// This check is a critical defense against race conditions. If another goroutine (e.g., the cleanup loop) has
	// already finalized this item, we must respect that outcome.
	if item.isFinalized() {
		finalState := item.finalState
		return true, finalState.Outcome, finalState.Err
	}

	// Check if the request's context has been cancelled.
	if ctxErr := item.OriginalRequest().Context().Err(); ctxErr != nil {
		return true, types.QueueOutcomeEvictedContextCancelled, fmt.Errorf("%w: %w", types.ErrContextCancelled, ctxErr)
	}

	// Check if the item has outlived its TTL.
	if item.EffectiveTTL() > 0 && now.Sub(item.EnqueueTime()) > item.EffectiveTTL() {
		return true, types.QueueOutcomeEvictedTTL, types.ErrTTLExpired
	}

	return false, types.QueueOutcomeNotYetFinalized, nil
}

// runExpiryCleanup starts a background goroutine that periodically scans all queues on the shard for expired items.
func (sp *ShardProcessor) runExpiryCleanup(ctx context.Context) {
	defer sp.wg.Done()
	logger := sp.logger.WithName("runExpiryCleanup")
	logger.V(logutil.DEFAULT).Info("Shard expiry cleanup goroutine starting.")
	defer logger.V(logutil.DEFAULT).Info("Shard expiry cleanup goroutine stopped.")

	ticker := time.NewTicker(sp.expiryCleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case now := <-ticker.C:
			sp.cleanupExpired(now)
		}
	}
}

// cleanupExpired performs a single scan of all queues on the shard, removing and finalizing any items that have
// expired.
func (sp *ShardProcessor) cleanupExpired(now time.Time) {
	processFn := func(managedQ contracts.ManagedQueue, queueLogger logr.Logger) {
		// This predicate identifies items to be removed by the Cleanup call.
		predicate := func(item types.QueueItemAccessor) bool {
			isExpired, _, _ := checkItemExpiry(item, now)
			return isExpired
		}

		removedItems, err := managedQ.Cleanup(predicate)
		if err != nil {
			queueLogger.Error(err, "Error during ManagedQueue Cleanup")
		}

		// Finalize all the items that were removed.
		sp.finalizeExpiredItems(removedItems, now, queueLogger)
	}
	sp.processAllQueuesConcurrently("cleanupExpired", processFn)
}

// shutdown handles the graceful termination of the processor, ensuring any pending items in the enqueue channel or in
// the queues are finalized correctly.
func (sp *ShardProcessor) shutdown() {
	sp.shutdownOnce.Do(func() {
		// Set the atomic bool so that any new calls to Enqueue will fail fast.
		sp.isShuttingDown.Store(true)
		sp.logger.V(logutil.DEFAULT).Info("Shard processor shutting down.")

		// Drain the channel BEFORE closing it. This prevents a panic from any goroutine that is currently blocked trying to
		// send to the channel. We read until it's empty.
	DrainLoop:
		for {
			select {
			case item := <-sp.enqueueChan:
				if item == nil { // This is a safeguard against logic errors in the distributor.
					continue
				}
				item.Finalize(types.QueueOutcomeRejectedOther,
					fmt.Errorf("%w: %w", types.ErrRejected, types.ErrFlowControllerNotRunning))
			default:
				// The channel is empty, we can now safely close it.
				break DrainLoop
			}
		}
		close(sp.enqueueChan)

		// Evict all remaining items from the queues.
		sp.evictAll()
	})
}

// evictAll drains all queues on the shard and finalizes every item with a shutdown error.
func (sp *ShardProcessor) evictAll() {
	processFn := func(managedQ contracts.ManagedQueue, queueLogger logr.Logger) {
		removedItems, err := managedQ.Drain()
		if err != nil {
			queueLogger.Error(err, "Error during ManagedQueue Drain")
		}

		// Finalize all the items that were removed.
		getOutcome := func(_ types.QueueItemAccessor) (types.QueueOutcome, error) {
			return types.QueueOutcomeEvictedOther, fmt.Errorf("%w: %w", types.ErrEvicted, types.ErrFlowControllerNotRunning)
		}
		sp.finalizeItems(removedItems, queueLogger, getOutcome)
	}
	sp.processAllQueuesConcurrently("evictAll", processFn)
}

// processAllQueuesConcurrently iterates over all queues in all priority bands on the shard and executes the given
// `processFn` for each queue using a dynamically sized worker pool.
func (sp *ShardProcessor) processAllQueuesConcurrently(
	ctxName string,
	processFn func(mq contracts.ManagedQueue, logger logr.Logger),
) {
	logger := sp.logger.WithName(ctxName)

	// Phase 1: Collect all queues to be processed into a single slice.
	// This avoids holding locks on the shard while processing, and allows us to determine the optimal number of workers.
	var queuesToProcess []framework.FlowQueueAccessor
	for _, priority := range sp.shard.AllOrderedPriorityLevels() {
		band, err := sp.shard.PriorityBandAccessor(priority)
		if err != nil {
			logger.Error(err, "Failed to get PriorityBandAccessor", "priority", priority)
			continue
		}
		band.IterateQueues(func(queue framework.FlowQueueAccessor) bool {
			queuesToProcess = append(queuesToProcess, queue)
			return true // Continue iterating.
		})
	}

	if len(queuesToProcess) == 0 {
		return
	}

	// Phase 2: Determine the optimal number of workers.
	// We cap the number of workers to a reasonable fixed number to avoid overwhelming the scheduler when many shards are
	// running. We also don't need more workers than there are queues.
	numWorkers := min(maxCleanupWorkers, len(queuesToProcess))

	// Phase 3: Create a worker pool to process the queues.
	tasks := make(chan framework.FlowQueueAccessor)

	var wg sync.WaitGroup
	for range numWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for q := range tasks {
				key := q.FlowKey()
				queueLogger := logger.WithValues(
					"flowKey", key,
					"flowID", key.ID,
					"flowPriority", key.Priority)
				managedQ, err := sp.shard.ManagedQueue(key)
				if err != nil {
					queueLogger.Error(err, "Failed to get ManagedQueue")
					continue
				}
				processFn(managedQ, queueLogger)
			}
		}()
	}

	// Feed the channel with all the queues to be processed.
	for _, q := range queuesToProcess {
		tasks <- q
	}
	close(tasks) // Close the channel to signal workers to exit.
	wg.Wait()    // Wait for all workers to finish.
}

// finalizeItems is a helper to iterate over a slice of items, safely cast them, and finalize them with an outcome
// determined by the `getOutcome` function.
func (sp *ShardProcessor) finalizeItems(
	items []types.QueueItemAccessor,
	logger logr.Logger,
	getOutcome func(item types.QueueItemAccessor) (types.QueueOutcome, error),
) {
	for _, i := range items {
		item, ok := i.(*FlowItem)
		if !ok {
			unexpectedItemErr := fmt.Errorf("internal error: item %q of type %T is not a *FlowItem",
				i.OriginalRequest().ID(), i)
			logger.Error(unexpectedItemErr, "Panic condition detected during finalization", "item", i)
			continue
		}

		outcome, err := getOutcome(i)
		item.Finalize(outcome, err)
		logger.V(logutil.TRACE).Info("Item finalized", "reqID", item.OriginalRequest().ID(),
			"outcome", outcome, "err", err)
	}
}

// finalizeExpiredItems is a specialized version of finalizeItems for items that are known to be expired.
// It determines the precise reason for expiry and finalizes the item accordingly.
func (sp *ShardProcessor) finalizeExpiredItems(items []types.QueueItemAccessor, now time.Time, logger logr.Logger) {
	getOutcome := func(item types.QueueItemAccessor) (types.QueueOutcome, error) {
		// We don't need the `isExpired` boolean here because we know it's true, but this function conveniently returns the
		// precise outcome and error.
		_, outcome, expiryErr := checkItemExpiry(item, now)
		return outcome, fmt.Errorf("%w: %w", types.ErrEvicted, expiryErr)
	}
	sp.finalizeItems(items, logger, getOutcome)
}
