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

// ShardProcessor is the core worker of the FlowController.
//
// It is paired one-to-one with a RegistryShard instance and is responsible for all request lifecycle operations on that
// shard, from the point an item is successfully submitted to it.
//
// # Request Lifecycle Management & Ownership
//
// The ShardProcessor takes ownership of a FlowItem only after it has been successfully sent to its internal enqueueChan
// via Submit or SubmitOrBlock (i.e., when these methods return nil).
// Once the Processor takes ownership, it is solely responsible for ensuring that item.Finalize() or
// item.FinalizeWithOutcome() is called exactly once for that item, under all circumstances (dispatch, rejection, sweep,
// or shutdown).
//
// If Submit or SubmitOrBlock return an error, ownership remains with the caller (the Controller), which must then
// handle the finalization.
//
// # Concurrency Model
//
// To ensure correctness and high performance, the processor uses a single-goroutine, actor-based model. The main run
// loop is the sole writer for all state-mutating operations. This makes complex transactions (like capacity checks)
// inherently atomic without coarse-grained locks.
type ShardProcessor struct {
	shard                contracts.RegistryShard
	saturationDetector   contracts.SaturationDetector
	clock                clock.WithTicker
	cleanupSweepInterval time.Duration
	logger               logr.Logger

	// lifecycleCtx controls the processor's lifetime. Monitored by Submit* methods for safe shutdown.
	lifecycleCtx context.Context

	// enqueueChan is the entry point for new requests.
	enqueueChan chan *FlowItem

	// wg is used to wait for background tasks (cleanup sweep) to complete on shutdown.
	wg             sync.WaitGroup
	isShuttingDown atomic.Bool
	shutdownOnce   sync.Once
}

// NewShardProcessor creates a new ShardProcessor instance.
func NewShardProcessor(
	ctx context.Context,
	shard contracts.RegistryShard,
	saturationDetector contracts.SaturationDetector,
	clock clock.WithTicker,
	cleanupSweepInterval time.Duration,
	enqueueChannelBufferSize int,
	logger logr.Logger,
) *ShardProcessor {
	return &ShardProcessor{
		shard:                shard,
		saturationDetector:   saturationDetector,
		clock:                clock,
		cleanupSweepInterval: cleanupSweepInterval,
		logger:               logger,
		lifecycleCtx:         ctx,
		enqueueChan:          make(chan *FlowItem, enqueueChannelBufferSize),
	}
}

// Submit attempts a non-blocking handoff of an item to the processor's internal enqueue channel.
//
// Ownership Contract:
//   - Returns nil: The item was successfully handed off.
//     The ShardProcessor takes responsibility for calling Finalize on the item.
//   - Returns error: The item was not handed off.
//     Ownership of the FlowItem remains with the caller, who is responsible for calling Finalize.
//
// Possible errors:
//   - ErrProcessorBusy: The processor's input channel is full.
//   - types.ErrFlowControllerNotRunning: The processor is shutting down.
func (sp *ShardProcessor) Submit(item *FlowItem) error {
	if sp.isShuttingDown.Load() {
		return types.ErrFlowControllerNotRunning
	}
	select {
	case sp.enqueueChan <- item:
		return nil // Ownership transferred.
	case <-sp.lifecycleCtx.Done():
		return types.ErrFlowControllerNotRunning
	default:
		return ErrProcessorBusy
	}
}

// SubmitOrBlock performs a blocking handoff of an item to the processor's internal enqueue channel.
// It waits until the item is handed off, the caller's context is cancelled, or the processor shuts down.
//
// Ownership Contract:
//   - Returns nil: The item was successfully handed off.
//     The ShardProcessor takes responsibility for calling Finalize on the item.
//   - Returns error: The item was not handed off.
//     Ownership of the FlowItem remains with the caller, who is responsible for calling Finalize.
//
// Possible errors:
//   - ctx.Err(): The provided context was cancelled or its deadline exceeded.
//   - types.ErrFlowControllerNotRunning: The processor is shutting down.
func (sp *ShardProcessor) SubmitOrBlock(ctx context.Context, item *FlowItem) error {
	if sp.isShuttingDown.Load() {
		return types.ErrFlowControllerNotRunning
	}

	select {
	case sp.enqueueChan <- item:
		return nil // Ownership transferred.
	case <-ctx.Done():
		return ctx.Err()
	case <-sp.lifecycleCtx.Done():
		return types.ErrFlowControllerNotRunning
	}
}

// Run is the main operational loop for the shard processor. It must be run as a goroutine.
// It uses a `select` statement to interleave accepting new requests with dispatching existing ones, balancing
// responsiveness with throughput.
func (sp *ShardProcessor) Run(ctx context.Context) {
	sp.logger.V(logutil.DEFAULT).Info("Shard processor run loop starting.")
	defer sp.logger.V(logutil.DEFAULT).Info("Shard processor run loop stopped.")

	sp.wg.Add(1)
	go sp.runCleanupSweep(ctx)

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

	// --- External Finalization Check ---
	// Check if the item was finalized by the Controller while buffered in enqueueChan.
	if finalState := item.FinalState(); finalState != nil {
		sp.logger.V(logutil.TRACE).Info("Item finalized externally before processing, discarding.",
			"outcome", finalState.Outcome, "err", finalState.Err, "flowKey", key, "reqID", req.ID())
		return
	}

	// --- Configuration Validation ---
	managedQ, err := sp.shard.ManagedQueue(key)
	if err != nil {
		finalErr := fmt.Errorf("configuration error: failed to get queue for flow key %s: %w", key, err)
		sp.logger.Error(finalErr, "Rejecting item.", "flowKey", key, "reqID", req.ID())
		item.FinalizeWithOutcome(types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, finalErr))
		return
	}

	band, err := sp.shard.PriorityBandAccessor(key.Priority)
	if err != nil {
		finalErr := fmt.Errorf("configuration error: failed to get priority band for priority %d: %w", key.Priority, err)
		sp.logger.Error(finalErr, "Rejecting item.", "flowKey", key, "reqID", req.ID())
		item.FinalizeWithOutcome(types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, finalErr))
		return
	}

	// --- Capacity Check ---
	// This check is safe because it is performed by the single-writer Run goroutine.
	if !sp.hasCapacity(key.Priority, req.ByteSize()) {
		sp.logger.V(logutil.DEBUG).Info("Rejecting request, queue at capacity",
			"flowKey", key, "reqID", req.ID(), "priorityName", band.PriorityName(), "reqByteSize", req.ByteSize())
		item.FinalizeWithOutcome(types.QueueOutcomeRejectedCapacity, fmt.Errorf("%w: %w",
			types.ErrRejected, types.ErrQueueAtCapacity))
		return
	}

	// --- Commitment Point ---
	// The item is admitted. The ManagedQueue.Add implementation is responsible for calling item.SetHandle() atomically.
	if err := managedQ.Add(item); err != nil {
		finalErr := fmt.Errorf("failed to add item to queue for flow key %s: %w", key, err)
		sp.logger.Error(finalErr, "Rejecting item post-admission.",
			"flowKey", key, "reqID", req.ID(), "priorityName", band.PriorityName())
		item.FinalizeWithOutcome(types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, finalErr))
		return
	}
	sp.logger.V(logutil.TRACE).Info("Item enqueued.",
		"flowKey", key, "reqID", req.ID(), "priorityName", band.PriorityName())
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
// It returns true if an item was successfully dispatched, and false otherwise.
//
// # Error Handling Philosophy: Failure Isolation & Work Conservation
//
// A problem in one priority band (e.g., a failing policy) must not halt processing for other, healthy bands.
// Therefore, any error during selection or dispatch for a given band is logged, and the processor immediately continues
// to the next-lower priority band to maximize system throughput.
//
// # Strict Policy Adherence vs. Work Conservation
//
// This function's logic strictly adheres to the scheduling decisions made by the configured policies, even at the cost
// of work conservation. After the inter-flow (fairness) and intra-flow (ordering) policies select a request (e.g.,
// `A_1` from flow `A`), a post-selection viability check is performed.
//
// If request `A_1` targets saturated backends, this function will stop the entire dispatch cycle for the current tick.
// It will NOT attempt to find other work (like request `B_1` or `A_2`). Instead, it respects the policy decision that
// `A_1` is next and enforces Head-of-Line blocking on it.
//
// # Future Extension Point
//
// The iteration over priority bands is currently a simple, strict-priority loop. This could be abstracted into a third
// policy tier (e.g., an `InterBandDispatchPolicy`) if more complex scheduling between bands, such as Weighted Fair
// Queuing (WFQ), is ever required.
func (sp *ShardProcessor) dispatchCycle(ctx context.Context) bool {
	baseLogger := sp.logger.WithName("dispatchCycle")
	for _, priority := range sp.shard.AllOrderedPriorityLevels() {
		originalBand, err := sp.shard.PriorityBandAccessor(priority)
		if err != nil {
			baseLogger.Error(err, "Failed to get PriorityBandAccessor, skipping band", "priority", priority)
			continue
		}
		logger := baseLogger.WithValues("priority", priority, "priorityName", originalBand.PriorityName())

		item, err := sp.selectItem(originalBand, logger)
		if err != nil {
			logger.Error(err, "Failed to select item, skipping priority band for this cycle")
			continue
		}
		if item == nil {
			logger.V(logutil.TRACE).Info("No item selected by dispatch policies, skipping band")
			continue
		}

		logger = logger.WithValues(
			"flowKey", item.OriginalRequest().FlowKey(),
			"flowID", item.OriginalRequest().FlowKey().ID,
			"flowPriority", item.OriginalRequest().FlowKey().Priority,
			"reqID", item.OriginalRequest().ID(),
			"reqByteSize", item.OriginalRequest().ByteSize())

		candidatePods := item.OriginalRequest().CandidatePodsForScheduling()
		if sp.saturationDetector.IsSaturated(ctx, candidatePods) {
			logger.V(logutil.VERBOSE).Info("Policy's chosen item is for a saturated flow; pausing dispatch and blocking on HoL")
			return false
		}

		if err := sp.dispatchItem(item, logger); err != nil {
			logger.Error(err, "Failed to dispatch item, skipping priority band for this cycle")
			continue
		}
		return true
	}
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

	// Final check for external finalization right before dispatch.
	if finalState := removedItem.FinalState(); finalState != nil {
		// The item was finalized externally (e.g., by the Controller) between selection and removal.
		// We must respect that outcome.
		sp.logger.V(logutil.DEBUG).Info("Item finalized externally at time of dispatch, discarding.",
			"flowKey", req.FlowKey(), "reqID", req.ID())
		return nil // This is not considered a dispatch error. We effectively discarded a "zombie" item from the HoL.
	}

	sp.logger.V(logutil.TRACE).Info("Item dispatched.", "flowKey", req.FlowKey(), "reqID", req.ID())
	removedItem.FinalizeWithOutcome(types.QueueOutcomeDispatched, nil)
	return nil
}

// runCleanupSweep starts a background goroutine that periodically scans all queues for externally finalized items
// ("zombie" items) and removes them in batches.
func (sp *ShardProcessor) runCleanupSweep(ctx context.Context) {
	defer sp.wg.Done()
	logger := sp.logger.WithName("runCleanupSweep")
	logger.V(logutil.DEFAULT).Info("Shard cleanup sweep goroutine starting.")
	defer logger.V(logutil.DEFAULT).Info("Shard cleanup sweep goroutine stopped.")

	ticker := sp.clock.NewTicker(sp.cleanupSweepInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C():
			sp.sweepFinalizedItems()
		}
	}
}

// sweepFinalizedItems performs a single scan of all queues, removing finalized items in batch and releasing their
// memory.
func (sp *ShardProcessor) sweepFinalizedItems() {
	processFn := func(managedQ contracts.ManagedQueue, logger logr.Logger) {
		key := managedQ.FlowQueueAccessor().FlowKey()
		predicate := func(itemAcc types.QueueItemAccessor) bool {
			return itemAcc.(*FlowItem).FinalState() != nil
		}
		removedItems, err := managedQ.Cleanup(predicate)
		if err != nil {
			logger.Error(err, "Error during ManagedQueue Cleanup", "flowKey", key)
		}
		logger.V(logutil.DEBUG).Info("Swept finalized items and released capacity.",
			"flowKey", key, "count", len(removedItems))
	}
	sp.processAllQueuesConcurrently("sweepFinalizedItems", processFn)
}

// shutdown handles the graceful termination of the processor, ensuring all pending items (in channel and queues) are
// Finalized.
func (sp *ShardProcessor) shutdown() {
	sp.shutdownOnce.Do(func() {
		sp.isShuttingDown.Store(true)
		sp.logger.V(logutil.DEFAULT).Info("Shard processor shutting down.")

	DrainLoop: // Drain the enqueueChan to finalize buffered items.
		for {
			select {
			case item := <-sp.enqueueChan:
				if item == nil {
					continue
				}
				// Finalize buffered items.
				item.FinalizeWithOutcome(types.QueueOutcomeRejectedOther,
					fmt.Errorf("%w: %w", types.ErrRejected, types.ErrFlowControllerNotRunning))
			default:
				break DrainLoop
			}
		}
		// We do not close enqueueChan because external goroutines (Controller) send on it.
		// The channel will be garbage collected when the processor terminates.
		sp.evictAll()
	})
}

// evictAll drains all queues on the shard, finalizes every item, and releases their memory.
func (sp *ShardProcessor) evictAll() {
	processFn := func(managedQ contracts.ManagedQueue, logger logr.Logger) {
		key := managedQ.FlowQueueAccessor().FlowKey()
		removedItems, err := managedQ.Drain()
		if err != nil {
			logger.Error(err, "Error during ManagedQueue Drain", "flowKey", key)
		}

		outcome := types.QueueOutcomeEvictedOther
		errShutdown := fmt.Errorf("%w: %w", types.ErrEvicted, types.ErrFlowControllerNotRunning)
		for _, i := range removedItems {
			item, ok := i.(*FlowItem)
			if !ok {
				logger.Error(fmt.Errorf("internal error: unexpected type %T", i),
					"Panic condition detected during shutdown", "flowKey", key)
				continue
			}

			// Finalization is idempotent; safe to call even if already finalized externally.
			item.FinalizeWithOutcome(outcome, errShutdown)
			logger.V(logutil.TRACE).Info("Item evicted during shutdown.",
				"flowKey", key, "reqID", item.OriginalRequest().ID())
		}
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
