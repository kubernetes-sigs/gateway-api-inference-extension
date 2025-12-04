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

// ErrProcessorBusy is a sentinel error returned by the processor's Submit method indicating that the processor's.
// internal buffer is momentarily full and cannot accept new work.
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
	podLocator           contracts.PodLocator
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
	podLocator contracts.PodLocator,
	clock clock.WithTicker,
	cleanupSweepInterval time.Duration,
	enqueueChannelBufferSize int,
	logger logr.Logger,
) *ShardProcessor {
	return &ShardProcessor{
		shard:                shard,
		saturationDetector:   saturationDetector,
		podLocator:           podLocator,
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
	select { // The default case makes this select non-blocking.
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

	select { // The absence of a default case makes this call blocking.
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

	// Create a ticker for periodic dispatch attempts to avoid tight loops
	dispatchTicker := sp.clock.NewTicker(time.Millisecond)
	defer dispatchTicker.Stop()

	// This is the main worker loop. It continuously processes incoming requests and dispatches queued requests until the
	// context is cancelled. The `select` statement has three cases:
	//
	//  1. Context Cancellation: The highest priority is shutting down. If the context's `Done` channel is closed, the
	//     loop will drain all queues and exit. This is the primary exit condition.
	//  2. New Item Arrival: If an item is available on `enqueueChan`, it will be processed. This ensures that the
	//     processor is responsive to new work.
	//  3. Dispatch Ticker: Periodically triggers a dispatch cycle to attempt to dispatch items from existing queues,
	//     ensuring that queued work is processed even when no new items arrive.
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
			sp.dispatchCycle(ctx) // Process immediately when an item arrives
		case <-dispatchTicker.C():
			sp.dispatchCycle(ctx) // Periodically attempt to dispatch from queues
		}
	}
}

// enqueue processes an item received from the enqueueChan.
// It handles capacity checks, checks for external finalization, and either admits the item to a queue or rejects it.
func (sp *ShardProcessor) enqueue(item *FlowItem) {
	req := item.OriginalRequest()
	key := req.FlowKey()

	// --- Optimistic External Finalization Check ---
	// Check if the item was finalized by the Controller (due to TTL/cancellation) while it was buffered in enqueueChan.
	// This is an optimistic check to avoid unnecessary processing on items already considered dead.
	// The ultimate guarantee of cleanup for any races is the runCleanupSweep mechanism.
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

// hasCapacity checks if the shard and the specific priority band have enough capacity.
// This check reflects actual resource utilization, including "zombie" items (finalized but unswept), to prevent
// physical resource overcommitment.
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
		return false // Fail closed if configuration is inconsistent.
	}
	return bandStats.ByteSize+itemByteSize <= bandStats.CapacityBytes
}

// dispatchCycle attempts to dispatch a single item by iterating through priority bands from highest to lowest.
// It applies the configured policies for each band to select an item and then attempts to dispatch it.
// It returns true if an item was successfully dispatched, and false otherwise.
// It enforces Head-of-Line (HoL) blocking if the selected item is saturated.
//
// # Work Conservation and Head-of-Line (HoL) Blocking
//
// The cycle attempts to be work-conserving by skipping bands where selection fails.
// However, if a selected item is saturated (cannot be scheduled), the cycle stops immediately. This enforces HoL
// blocking to respect the policy's decision and prevent priority inversion, where dispatching lower-priority work might
// exacerbate the saturation affecting the high-priority item.
func (sp *ShardProcessor) dispatchCycle(ctx context.Context) bool {
	for _, priority := range sp.shard.AllOrderedPriorityLevels() {
		originalBand, err := sp.shard.PriorityBandAccessor(priority)
		if err != nil {
			sp.logger.Error(err, "Failed to get PriorityBandAccessor, skipping band", "priority", priority)
			continue
		}

		item, err := sp.selectItem(originalBand)
		if err != nil {
			sp.logger.Error(err, "Failed to select item, skipping priority band for this cycle",
				"priority", priority, "priorityName", originalBand.PriorityName())
			continue // Continue to the next band to maximize work conservation.
		}
		if item == nil {
			continue
		}

		// --- Viability Check (Saturation/HoL Blocking) ---
		req := item.OriginalRequest()
		candidates := sp.podLocator.Locate(ctx, req.GetMetadata())
		if sp.saturationDetector.IsSaturated(ctx, candidates) {
			sp.logger.V(logutil.DEBUG).Info("Policy's chosen item is saturated; enforcing HoL blocking.",
				"flowKey", req.FlowKey(), "reqID", req.ID(), "priorityName", originalBand.PriorityName())
			// Stop the dispatch cycle entirely to respect strict policy decision and prevent priority inversion where
			// lower-priority work might exacerbate the saturation affecting high-priority work.
			return false
		}

		// --- Dispatch ---
		if err := sp.dispatchItem(item); err != nil {
			sp.logger.Error(err, "Failed to dispatch item, skipping priority band for this cycle",
				"flowKey", req.FlowKey(), "reqID", req.ID(), "priorityName", originalBand.PriorityName())
			continue // Continue to the next band to maximize work conservation.
		}
		return true
	}
	return false
}

// selectItem applies the configured inter- and intra-flow dispatch policies to select a single item.
func (sp *ShardProcessor) selectItem(band framework.PriorityBandAccessor) (types.QueueItemAccessor, error) {
	interP, err := sp.shard.InterFlowDispatchPolicy(band.Priority())
	if err != nil {
		return nil, fmt.Errorf("could not get InterFlowDispatchPolicy: %w", err)
	}
	queue, err := interP.SelectQueue(band)
	if err != nil {
		return nil, fmt.Errorf("InterFlowDispatchPolicy %q failed to select queue: %w", interP.Name(), err)
	}
	if queue == nil {
		return nil, nil
	}
	key := queue.FlowKey()
	intraP, err := sp.shard.IntraFlowDispatchPolicy(key)
	if err != nil {
		return nil, fmt.Errorf("could not get IntraFlowDispatchPolicy for flow %s: %w", key, err)
	}
	item, err := intraP.SelectItem(queue)
	if err != nil {
		return nil, fmt.Errorf("IntraFlowDispatchPolicy %q failed to select item for flow %s: %w", intraP.Name(), key, err)
	}
	return item, nil
}

// dispatchItem handles the final steps of dispatching an item: removing it from the queue and finalizing its outcome.
func (sp *ShardProcessor) dispatchItem(itemAcc types.QueueItemAccessor) error {
	req := itemAcc.OriginalRequest()
	key := req.FlowKey()
	managedQ, err := sp.shard.ManagedQueue(key)
	if err != nil {
		return fmt.Errorf("failed to get ManagedQueue for flow %s: %w", key, err)
	}

	removedItemAcc, err := managedQ.Remove(itemAcc.Handle())
	if err != nil {
		// This happens benignly if the item was already removed by the cleanup sweep loop.
		// We log it at a low level for visibility but return nil so the dispatch cycle proceeds.
		sp.logger.V(logutil.DEBUG).Info("Failed to remove item during dispatch (likely already finalized and swept).",
			"flowKey", key, "reqID", req.ID(), "error", err)
		return nil
	}

	removedItem := removedItemAcc.(*FlowItem)
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
		removedItems := managedQ.Cleanup(predicate)
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
		removedItems := managedQ.Drain()

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
