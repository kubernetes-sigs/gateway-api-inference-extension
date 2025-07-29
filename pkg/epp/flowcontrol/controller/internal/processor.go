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

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	// enqueueChannelBufferSize sets the size of the buffered channel that accepts incoming requests for the shard
	// processor. This buffer acts as a "shock absorber," decoupling the upstream distributor from the processor's main
	// loop and allowing the system to handle short, intense bursts of traffic without blocking the distributor.
	enqueueChannelBufferSize = 100

	// maxCleanupWorkers caps the number of concurrent workers for background cleanup tasks. This prevents a single shard
	// from overwhelming the Go scheduler with too many goroutines.
	maxCleanupWorkers = 4
)

var (
	// errInterFlow is a sentinel error for failures during the inter-flow dispatch phase (e.g., a
	// `framework.InterFlowDispatchPolicy` fails to select a queue).
	//
	// Strategy: When this error is encountered, the dispatch cycle aborts processing for the current priority band and
	// immediately moves to the next, promoting work conservation. A failure in one band should not halt progress in
	// others.
	errInterFlow = errors.New("inter-flow policy failure")

	// errIntraFlow is a sentinel error for failures *after* a specific flow's queue has been selected (e.g., a
	// `framework.IntraFlowDispatchPolicy` fails or a queue `Remove` fails).
	//
	// Strategy: When this error is encountered, the dispatch cycle aborts processing for the entire priority band for the
	// current cycle. This acts as a critical circuit breaker. A stateless inter-flow policy could otherwise repeatedly
	// select the same problematic queue in a tight loop of failures. Halting the band for one cycle prevents this.
	errIntraFlow = errors.New("intra-flow operation failure")
)

// clock defines an interface for getting the current time, allowing for dependency injection in tests.
type clock interface {
	Now() time.Time
}

// ShardProcessor is the core worker of the `controller.FlowController`. It is paired one-to-one with a
// `contracts.RegistryShard` instance and is responsible for all request lifecycle operations on that shard, including
// enqueueing, dispatching, and expiry cleanup. It acts as the "data plane" worker that executes against the
// concurrent-safe state provided by its shard.
//
// For a full rationale on the single-writer concurrency model, see the package-level documentation in `doc.go`.
//
// # Concurrency Guarantees and Race Conditions
//
// This model provides two key guarantees:
//
//  1. **Safe Enqueueing**: The `Run` method's goroutine has exclusive ownership of all operations that *add* items to
//     queues. This makes the "check-then-act" sequence in `enqueue` (calling `hasCapacity` then `managedQ.Add`)
//     inherently atomic from a writer's perspective, preventing capacity breaches. While the background
//     `runExpiryCleanup` goroutine can concurrently *remove* items, this is a benign race; a concurrent removal only
//     creates more available capacity, ensuring the `hasCapacity` check remains valid.
//
//  2. **Idempotent Finalization**: The primary internal race is between the main `dispatchCycle` and the background
//     `runExpiryCleanup` goroutine, which might try to finalize the same `flowItem` simultaneously. This race is
//     resolved by the `flowItem.finalize` method, which uses `sync.Once` to guarantee that only one of these goroutines
//     can set the item's final state.
type ShardProcessor struct {
	shard                 contracts.RegistryShard
	dispatchFilter        BandFilter
	clock                 clock
	expiryCleanupInterval time.Duration
	logger                logr.Logger

	// enqueueChan is the entry point for new requests to be processed by this shard's `Run` loop.
	enqueueChan chan *flowItem
	// wg is used to wait for background tasks like expiry cleanup to complete on shutdown.
	wg             sync.WaitGroup
	isShuttingDown atomic.Bool
	shutdownOnce   sync.Once
}

// NewShardProcessor creates a new `ShardProcessor` instance.
func NewShardProcessor(
	shard contracts.RegistryShard,
	dispatchFilter BandFilter,
	clock clock,
	expiryCleanupInterval time.Duration,
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
		enqueueChan: make(chan *flowItem, enqueueChannelBufferSize),
	}
}

// Run is the main operational loop for the shard processor. It must be run as a goroutine.
//
// # Loop Strategy: Interleaving Enqueue and Dispatch
//
// The loop uses a `select` statement to interleave two primary tasks:
//  1. Accepting new requests from the `enqueueChan`.
//  2. Attempting to dispatch existing requests from queues via `dispatchCycle`.
//
// This strategy is crucial for balancing responsiveness and throughput. When a new item arrives, it is immediately
// enqueued, and a dispatch cycle is triggered. This gives high-priority new arrivals a chance to be dispatched quickly.
// When no new items are arriving, the loop's `default` case continuously calls `dispatchCycle` to drain the existing
// backlog, ensuring work continues.
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
	//
	//  2. New Item Arrival: If an item is available on `enqueueChan`, it will be processed. This ensures that the
	//     processor is responsive to new work.
	//
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
			if !sp.dispatchCycle(ctx) {
				// If no work was done, yield to other goroutines to prevent a tight, busy-loop when idle, but allow for
				// immediate rescheduling.
				runtime.Gosched()
			}
		}
	}
}

// Enqueue sends a new flow item to the processor's internal channel for asynchronous processing by its main `Run` loop.
// If the processor is shutting down, it immediately finalizes the item with a shutdown error.
func (sp *ShardProcessor) Enqueue(item *flowItem) {
	if sp.isShuttingDown.Load() {
		item.finalize(types.QueueOutcomeRejectedOther,
			fmt.Errorf("%w: %w", types.ErrRejected, types.ErrFlowControllerShutdown))
		return
	}
	sp.enqueueChan <- item
}

// enqueue is the internal implementation for adding a new item to a managed queue. It is always run from the single
// main `Run` goroutine, making its "check-then-act" logic for capacity safe.
func (sp *ShardProcessor) enqueue(item *flowItem) {
	req := item.OriginalRequest()
	logger := log.FromContext(req.Context()).WithName("enqueue").WithValues(
		"flowID", req.FlowID(),
		"reqID", req.ID(),
		"reqByteSize", req.ByteSize(),
	)

	managedQ, err := sp.shard.ActiveManagedQueue(req.FlowID())
	if err != nil {
		// This is a significant configuration error; an active queue should exist for a valid flow.
		finalErr := fmt.Errorf("configuration error: failed to get active queue for flow %q: %w", req.FlowID(), err)
		logger.Error(finalErr, "Rejecting item.")
		item.finalize(types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, finalErr))
		return
	}
	priority := managedQ.FlowQueueAccessor().FlowSpec().Priority
	logger = logger.WithValues("priority", priority)

	band, err := sp.shard.PriorityBandAccessor(priority)
	if err != nil {
		finalErr := fmt.Errorf("configuration error: failed to get priority band for priority %d: %w", priority, err)
		logger.Error(finalErr, "Rejecting item.")
		item.finalize(types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, finalErr))
		return
	}
	logger = logger.WithValues("priorityName", band.PriorityName())

	if !sp.hasCapacity(priority, req.ByteSize()) {
		// This is an expected outcome, not a system error. Log at the default level with rich context.
		stats := sp.shard.Stats()
		bandStats := stats.PerPriorityBandStats[priority]
		logger.V(logutil.DEFAULT).Info("Rejecting request, queue at capacity",
			"outcome", types.QueueOutcomeRejectedCapacity,
			"shardTotalBytes", stats.TotalByteSize,
			"shardCapacityBytes", stats.TotalCapacityBytes,
			"bandTotalBytes", bandStats.ByteSize,
			"bandCapacityBytes", bandStats.CapacityBytes,
		)
		item.finalize(types.QueueOutcomeRejectedCapacity, fmt.Errorf("%w: %w", types.ErrRejected, types.ErrQueueAtCapacity))
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
		outcome, err := item.FinalState()
		logger.V(logutil.VERBOSE).Info("Item finalized before adding to queue, ignoring.",
			"outcome", outcome, "err", err)
		return
	}

	// This is the point of commitment. After this call, the item is officially in the queue and is the responsibility of
	// the dispatch or cleanup loops to finalize.
	if err := managedQ.Add(item); err != nil {
		finalErr := fmt.Errorf("failed to add item to queue for flow %q: %w", req.FlowID(), err)
		logger.Error(finalErr, "Rejecting item.")
		item.finalize(types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, finalErr))
		return
	}
	logger.V(logutil.TRACE).Info("Item enqueued.")
}

// hasCapacity checks if the shard and the specific priority band have enough capacity to accommodate an item of a given
// size.
func (sp *ShardProcessor) hasCapacity(priority uint, itemByteSize uint64) bool {
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
// It returns true if an item was successfully dispatched, and false otherwise.
//
// # Error Handling Philosophy
//
// The engine employs a robust, two-tiered error handling strategy to isolate failures and maximize system availability.
// This is managed via the `errInterFlow` and `errIntraFlow` sentinel errors.
//
//   - Inter-Flow Failures: If a failure occurs while selecting a flow (e.g., the `InterFlowDispatchPolicy` fails), the
//     processor aborts the *current priority band* and immediately moves to the next one. This promotes work
//     conservation, ensuring a single misconfigured band does not halt progress for the entire system.
//
//   - Intra-Flow Failures: If a failure occurs *after* a flow has been selected (e.g., the `IntraFlowDispatchPolicy`
//     fails), the processor aborts the *entire priority band* for the current cycle. This is a critical circuit
//     breaker. An inter-flow policy that is not stateful with respect to past failures could otherwise repeatedly
//     select the same problematic queue, causing a tight loop of failures. Halting the band for one cycle prevents
//     this.
func (sp *ShardProcessor) dispatchCycle(ctx context.Context) bool {
	baseLogger := sp.logger.WithName("dispatchCycle")

	// FUTURE EXTENSION POINT: The iteration over priority bands is currently a simple, strict-priority loop.
	// This could be abstracted into a third policy tier (e.g., an `InterBandDispatchPolicy`) if more complex scheduling
	// between bands, such as Weighted Fair Queuing (WFQ), is ever required. For now, strict priority is sufficient.
	for _, priority := range sp.shard.AllOrderedPriorityLevels() {
		band, err := sp.shard.PriorityBandAccessor(priority)
		if err != nil {
			baseLogger.Error(err, "Failed to get PriorityBandAccessor, skipping band", "priority", priority)
			continue
		}
		logger := baseLogger.WithValues("priority", priority, "priorityName", band.PriorityName())

		// Apply the configured filter to get a view of only the dispatchable flows.
		allowedFlows, shouldPause := sp.dispatchFilter(ctx, band, logger)
		if shouldPause {
			return false // A global gate told us to stop the entire cycle.
		}

		dispatchableBand := band
		if allowedFlows != nil {
			// An explicit subset of flows is allowed; create a filtered view.
			dispatchableBand = newSubsetPriorityBandAccessor(band, allowedFlows)
		}

		// Pass the (potentially filtered) band to the policies.
		item, dispatchPriority, err := sp.selectItem(dispatchableBand, logger)
		if err != nil {
			// The error handling strategy depends on the type of failure (inter- vs. intra-flow).
			if errors.Is(err, errIntraFlow) {
				logger.Error(err, "Intra-flow policy failure, skipping priority band for this cycle")
			} else {
				logger.Error(err, "Inter-flow policy or configuration failure, skipping priority band for this cycle")
			}
			continue
		}
		if item == nil {
			// This is the common case where a priority band has no items to dispatch.
			logger.V(logutil.TRACE).Info("No item selected by dispatch policies, skipping band")
			continue
		}
		logger = logger.WithValues("flowID", item.OriginalRequest().FlowID(), "reqID", item.OriginalRequest().ID())

		if err := sp.dispatchItem(item, dispatchPriority, logger); err != nil {
			// All errors from dispatchItem are considered intra-flow and unrecoverable for this band in this cycle.
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
) (types.QueueItemAccessor, uint, error) {
	interP, err := sp.shard.InterFlowDispatchPolicy(band.Priority())
	if err != nil {
		return nil, 0, fmt.Errorf("%w: could not get InterFlowDispatchPolicy: %w", errInterFlow, err)
	}
	queue, err := interP.SelectQueue(band)
	if err != nil {
		return nil, 0, fmt.Errorf("%w: InterFlowDispatchPolicy %q failed to select queue: %w",
			errInterFlow, interP.Name(), err)
	}
	if queue == nil {
		logger.V(logutil.TRACE).Info("No queue selected by InterFlowDispatchPolicy")
		return nil, 0, nil
	}
	logger = logger.WithValues("selectedFlowID", queue.FlowSpec().ID)

	priority := queue.FlowSpec().Priority
	intraP, err := sp.shard.IntraFlowDispatchPolicy(queue.FlowSpec().ID, priority)
	if err != nil {
		// This is an intra-flow failure because we have already successfully selected a queue.
		return nil, 0, fmt.Errorf("%w: could not get IntraFlowDispatchPolicy for flow %q: %w",
			errIntraFlow, queue.FlowSpec().ID, err)
	}
	item, err := intraP.SelectItem(queue)
	if err != nil {
		return nil, 0, fmt.Errorf("%w: IntraFlowDispatchPolicy %q failed to select item for flow %q: %w",
			errIntraFlow, intraP.Name(), queue.FlowSpec().ID, err)
	}
	if item == nil {
		logger.V(logutil.TRACE).Info("No item selected by IntraFlowDispatchPolicy")
		return nil, 0, nil
	}
	return item, priority, nil
}

// dispatchItem handles the final steps of dispatching an item after it has been selected by policies. This includes
// removing it from its queue, checking for last-minute expiry, and finalizing its outcome.
func (sp *ShardProcessor) dispatchItem(itemAcc types.QueueItemAccessor, priority uint, logger logr.Logger) error {
	logger = logger.WithName("dispatchItem")

	req := itemAcc.OriginalRequest()
	// We must look up the queue by its specific priority, as a flow might have draining queues at other levels.
	managedQ, err := sp.shard.ManagedQueue(req.FlowID(), priority)
	if err != nil {
		return fmt.Errorf("%w: failed to get ManagedQueue for flow %q at priority %d: %w",
			errIntraFlow, req.FlowID(), priority, err)
	}

	// The core mutation: remove the item from the queue.
	removedItemAcc, err := managedQ.Remove(itemAcc.Handle())
	if err != nil {
		// This can happen benignly if the item was already removed by the expiry cleanup loop between the time it was
		// selected by the policy and the time this function is called.
		logger.V(logutil.VERBOSE).Info("Item already removed from queue, likely by expiry cleanup", "err", err)
		return fmt.Errorf("%w: failed to remove item %q from queue for flow %q: %w",
			errIntraFlow, req.ID(), req.FlowID(), err)
	}

	removedItem, ok := removedItemAcc.(*flowItem)
	if !ok {
		// This indicates a severe logic error where a queue returns an item of an unexpected type. This violates a
		// core system invariant: all items managed by the processor must be of type *flowItem. This is an unrecoverable
		// state for this shard.
		unexpectedItemErr := fmt.Errorf("%w: internal error: item %q of type %T is not a *flowItem",
			errIntraFlow, removedItemAcc.OriginalRequest().ID(), removedItemAcc)
		panic(unexpectedItemErr)
	}

	// Final check for expiry/cancellation right before dispatch.
	isExpired, outcome, expiryErr := checkItemExpiry(removedItem, sp.clock.Now())
	if isExpired {
		// Ensure we always have a non-nil error to wrap for consistent logging and error handling.
		finalErr := expiryErr
		if finalErr == nil {
			finalErr = errors.New("item finalized before dispatch")
		}
		logger.V(logutil.VERBOSE).Info("Item expired at time of dispatch, evicting", "outcome", outcome,
			"err", finalErr)
		removedItem.finalize(outcome, fmt.Errorf("%w: %w", types.ErrEvicted, finalErr))
		// Return an error to signal that the dispatch did not succeed.
		return fmt.Errorf("%w: item %q expired before dispatch: %w", errIntraFlow, req.ID(), finalErr)
	}

	// Finalize the item as dispatched.
	removedItem.finalize(types.QueueOutcomeDispatched, nil)
	logger.V(logutil.TRACE).Info("Item dispatched.")
	return nil
}

// checkItemExpiry checks if an item has been cancelled (via its context) or has exceeded its TTL. It returns true if
// the item is expired, along with the corresponding outcome and error.
//
// This function provides "defense in depth" against race conditions. It is the authoritative check that is called from
// multiple locations (the dispatch loop and the cleanup loop) to determine if an item should be evicted. Its first
// action is to check if the item has *already* been finalized by a competing goroutine, ensuring that the final outcome
// is decided exactly once.
func checkItemExpiry(
	itemAcc types.QueueItemAccessor,
	now time.Time,
) (isExpired bool, outcome types.QueueOutcome, err error) {
	item, ok := itemAcc.(*flowItem)
	if !ok {
		// This indicates a severe logic error where a queue returns an item of an unexpected type. This violates a
		// core system invariant: all items managed by the processor must be of type *flowItem. This is an unrecoverable
		// state for this shard.
		unexpectedItemErr := fmt.Errorf("internal error: item %q of type %T is not a *flowItem",
			itemAcc.OriginalRequest().ID(), itemAcc)
		panic(unexpectedItemErr)
	}

	// This check is a critical defense against race conditions. If another goroutine (e.g., the cleanup loop) has
	// already finalized this item, we must respect that outcome.
	if item.isFinalized() {
		outcome, err := item.FinalState()
		return true, outcome, err
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
// expired due to TTL or context cancellation.
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

// shutdown handles the graceful termination of the processor. It uses sync.Once to guarantee that the shutdown logic is
// executed exactly once, regardless of whether it's triggered by context cancellation or the closing of the enqueue
// channel.
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
				item.finalize(types.QueueOutcomeRejectedOther,
					fmt.Errorf("%w: %w", types.ErrRejected, types.ErrFlowControllerShutdown))
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

// evictAll drains all queues on the shard and finalizes every item with a shutdown error. This is called when the
// processor is shutting down to ensure no requests are left in a pending state.
func (sp *ShardProcessor) evictAll() {
	processFn := func(managedQ contracts.ManagedQueue, queueLogger logr.Logger) {
		removedItems, err := managedQ.Drain()
		if err != nil {
			queueLogger.Error(err, "Error during ManagedQueue Drain")
		}

		// Finalize all the items that were removed.
		getOutcome := func(_ types.QueueItemAccessor) (types.QueueOutcome, error) {
			return types.QueueOutcomeEvictedOther, fmt.Errorf("%w: %w", types.ErrEvicted, types.ErrFlowControllerShutdown)
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
				queueLogger := logger.WithValues("flowID", q.FlowSpec().ID, "priority", q.FlowSpec().Priority)
				managedQ, err := sp.shard.ManagedQueue(q.FlowSpec().ID, q.FlowSpec().Priority)
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
		item, ok := i.(*flowItem)
		if !ok {
			unexpectedItemErr := fmt.Errorf("internal error: item %q of type %T is not a *flowItem",
				i.OriginalRequest().ID(), i)
			logger.Error(unexpectedItemErr, "Panic condition detected during finalization", "item", i)
			continue
		}

		outcome, err := getOutcome(i)
		item.finalize(outcome, err)
		logger.V(logutil.TRACE).Info("Item finalized", "reqID", item.OriginalRequest().ID(),
			"outcome", outcome, "err", err)
	}
}

// finalizeExpiredItems is a specialized version of finalizeItems for items that are known to be expired. It determines
// the precise reason for expiry and finalizes the item accordingly.
func (sp *ShardProcessor) finalizeExpiredItems(items []types.QueueItemAccessor, now time.Time, logger logr.Logger) {
	getOutcome := func(item types.QueueItemAccessor) (types.QueueOutcome, error) {
		// We don't need the `isExpired` boolean here because we know it's true, but this function conveniently returns the
		// precise outcome and error.
		_, outcome, expiryErr := checkItemExpiry(item, now)
		return outcome, fmt.Errorf("%w: %w", types.ErrEvicted, expiryErr)
	}
	sp.finalizeItems(items, logger, getOutcome)
}
