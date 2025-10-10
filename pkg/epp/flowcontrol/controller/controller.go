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

// Package controller contains the implementation of the `FlowController` engine.
//
// The FlowController is the central processing engine of the Flow Control system. It is a sharded, high-throughput
// component responsible for managing the lifecycle of all incoming requests. It achieves this by acting as a stateless
// supervisor that orchestrates a pool of stateful workers (`internal.ShardProcessor`), distributing incoming requests
// among them using a sophisticated load-balancing algorithm.
package controller

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/go-logr/logr"
	k8srand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/utils/clock"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/controller/internal"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// registryClient defines the minimal interface that the `FlowController` needs to interact with the `FlowRegistry`.
type registryClient interface {
	contracts.FlowRegistryObserver
	contracts.FlowRegistryDataPlane
}

// shardProcessor is the minimal internal interface that the `FlowController` requires from its workers.
// This abstraction allows for the injection of mock processors during testing.
type shardProcessor interface {
	Run(ctx context.Context)
	Submit(item *internal.FlowItem) error
	SubmitOrBlock(ctx context.Context, item *internal.FlowItem) error
}

// shardProcessorFactory defines the signature for a function that creates a `shardProcessor`.
// This enables dependency injection for testing.
type shardProcessorFactory func(
	ctx context.Context,
	shard contracts.RegistryShard,
	saturationDetector contracts.SaturationDetector,
	clock clock.WithTicker,
	cleanupSweepInterval time.Duration,
	enqueueChannelBufferSize int,
	logger logr.Logger,
) shardProcessor

var _ shardProcessor = &internal.ShardProcessor{}

// managedWorker holds the state for a single supervised worker.
type managedWorker struct {
	processor shardProcessor
	cancel    context.CancelFunc
}

// FlowController is the central, high-throughput engine of the Flow Control system.
// It is designed as a stateless distributor that orchestrates a pool of stateful workers (`internal.ShardProcessor`),
// following a "supervisor-worker" pattern.
//
// The controller's `Run` loop executes periodically, acting as a garbage collector that keeps the pool of running
// workers synchronized with the dynamic shard topology of the `FlowRegistry`.
//
// Request Lifecycle Management:
//
//  1. Asynchronous Finalization (Controller-Owned): The Controller actively monitors the request Context
//     (TTL/Cancellation) in EnqueueAndWait. If the Context expires, the Controller immediately Finalizes the item and
//     unblocks the caller.
//  2. Synchronous Finalization (Processor-Owned): The Processor handles Dispatch, Capacity Rejection, and Shutdown.
//  3. Cleanup (Processor-Owned): The Processor periodically sweeps externally finalized items to reclaim capacity.
type FlowController struct {
	// --- Immutable dependencies (set at construction) ---

	config                Config
	registry              registryClient
	saturationDetector    contracts.SaturationDetector
	clock                 clock.WithTicker
	logger                logr.Logger
	shardProcessorFactory shardProcessorFactory

	// --- Lifecycle state ---

	// parentCtx is the root context for the controller's lifecycle, established when `Run` is called.
	// It is the parent for all long-lived worker goroutines.
	parentCtx context.Context

	// --- Concurrent state ---

	// workers is a highly concurrent map storing the `managedWorker` for each shard.
	// It is the controller's source of truth for the worker pool.
	// The key is the shard ID (`string`), and the value is a `*managedWorker`.
	workers sync.Map

	wg sync.WaitGroup
}

// flowControllerOption is a function that applies a configuration change to a `FlowController`.
// test-only
type flowControllerOption func(*FlowController)

// NewFlowController creates a new `FlowController` instance.
func NewFlowController(
	ctx context.Context,
	config Config,
	registry contracts.FlowRegistry,
	sd contracts.SaturationDetector,
	logger logr.Logger,
	opts ...flowControllerOption,
) (*FlowController, error) {
	fc := &FlowController{
		config:             *config.deepCopy(),
		registry:           registry,
		saturationDetector: sd,
		clock:              clock.RealClock{},
		logger:             logger.WithName("flow-controller"),
		parentCtx:          ctx,
	}

	// Use the real shard processor implementation by default.
	fc.shardProcessorFactory = func(
		ctx context.Context,
		shard contracts.RegistryShard,
		saturationDetector contracts.SaturationDetector,
		clock clock.WithTicker,
		cleanupSweepInterval time.Duration,
		enqueueChannelBufferSize int,
		logger logr.Logger,
	) shardProcessor {
		return internal.NewShardProcessor(
			ctx,
			shard,
			saturationDetector,
			clock,
			cleanupSweepInterval,
			enqueueChannelBufferSize,
			logger)
	}

	for _, opt := range opts {
		opt(fc)
	}

	go fc.run(ctx)
	return fc, nil
}

// run starts the `FlowController`'s main reconciliation loop.
// This loop is responsible for garbage collecting workers whose shards no longer exist in the registry.
// This method blocks until the provided context is cancelled and ALL worker goroutines have fully terminated.
func (fc *FlowController) run(ctx context.Context) {
	fc.logger.Info("Starting FlowController reconciliation loop.")
	defer fc.logger.Info("FlowController reconciliation loop stopped.")

	ticker := fc.clock.NewTicker(fc.config.ProcessorReconciliationInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			fc.shutdown()
			return
		case <-ticker.C():
			fc.reconcileProcessors()
		}
	}
}

// EnqueueAndWait is the primary, synchronous entry point to the Flow Control system. It submits a request and blocks
// until the request reaches a terminal outcome (dispatched, rejected, or evicted).
//
// # Design Rationale: The Synchronous Model
//
// This blocking model is deliberately chosen for its simplicity and robustness, especially in the context of Envoy
// External Processing (`ext_proc`), which operates on a stream-based protocol.
//
//   - `ext_proc` Alignment: A single goroutine typically manages the stream for a given HTTP request.
//     `EnqueueAndWait` fits this perfectly: the request-handling goroutine calls it, blocks, and upon return, has a
//     definitive outcome to act upon.
//   - Simplified State Management: The state of a "waiting" request is implicitly managed by the blocked goroutine's
//     stack and its `context.Context`. The system only needs to signal this specific goroutine to unblock it.
//   - Direct Backpressure: If queues are full, `EnqueueAndWait` returns an error immediately, providing direct
//     backpressure to the caller.
func (fc *FlowController) EnqueueAndWait(
	ctx context.Context,
	req types.FlowControlRequest,
) (types.QueueOutcome, error) {
	if req == nil {
		return types.QueueOutcomeRejectedOther, errors.New("request cannot be nil")
	}

	flowKey := req.FlowKey()
	fairnessID := flowKey.ID
	priority := strconv.Itoa(flowKey.Priority)
	metrics.IncFlowControlQueueSize(fairnessID, priority)
	defer metrics.DecFlowControlQueueSize(fairnessID, priority)

	// 1. Create the derived context that governs this request's lifecycle (Parent Cancellation + TTL).
	reqCtx, cancel, enqueueTime := fc.createRequestContext(ctx, req)
	defer cancel()

	// 2. Enter the distribution loop to find a home for the request.
	// This loop is responsible for retrying on ErrShardDraining.
	for {
		select { // Non-blocking check on controller lifecycle.
		case <-fc.parentCtx.Done():
			return types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, types.ErrFlowControllerNotRunning)
		case <-reqCtx.Done():
			return types.QueueOutcomeRejectedOther, fmt.Errorf(
				"%w: request context cancelled or TTL expired before distribution: %w",
				types.ErrRejected, context.Cause(reqCtx))
		default:
		}

		// Attempt to distribute the request once.
		item, err := fc.tryDistribution(reqCtx, req, enqueueTime)
		if err != nil {
			// Distribution failed terminally (e.g., no shards, context cancelled during blocking submit).
			// The item has already been finalized by tryDistribution.
			finalState := item.FinalState()
			return finalState.Outcome, finalState.Err
		}

		// Distribution was successful; ownership of the item has been transferred to a processor.
		// Now, we block here in awaitFinalization until the request is finalized by either the processor (e.g., dispatched,
		// rejected) or the controller itself (e.g., caller's context cancelled/TTL expired).
		outcome, err := fc.awaitFinalization(reqCtx, item)
		if errors.Is(err, contracts.ErrShardDraining) {
			// This is a benign race condition where the chosen shard started draining after acceptance.
			fc.logger.V(logutil.DEBUG).Info("Selected shard is Draining, retrying request distribution",
				"flowKey", req.FlowKey(), "requestID", req.ID())
			// Introduce a small, randomized delay (1-10ms) to prevent tight spinning loops and thundering herds during retry
			// scenarios (e.g., shard draining)
			// TODO: Replace this with a more sophisticated backoff strategy when our data parallelism story matures.
			// For now, this is more than sufficient.
			jitterMs := k8srand.Intn(10) + 1
			fc.clock.Sleep(time.Duration(jitterMs) * time.Millisecond)
			continue
		}

		// The outcome is terminal (Dispatched, Evicted, or a non-retriable rejection).
		return outcome, err
	}
}

var errNoShards = errors.New("no viable active shards available")

// tryDistribution handles a single attempt to select a shard and submit a request.
// If this function returns an error, it guarantees that the provided `item` has been finalized.
func (fc *FlowController) tryDistribution(
	reqCtx context.Context,
	req types.FlowControlRequest,
	enqueueTime time.Time,
) (*internal.FlowItem, error) {
	// Calculate effective TTL for item initialization (reqCtx is the enforcement mechanism).
	effectiveTTL := fc.config.DefaultRequestTTL
	if deadline, ok := reqCtx.Deadline(); ok {
		if ttl := deadline.Sub(enqueueTime); ttl > 0 {
			effectiveTTL = ttl
		}
	}

	// We must create a fresh FlowItem on each attempt as finalization is per-lifecycle.
	item := internal.NewItem(req, effectiveTTL, enqueueTime)

	candidates, err := fc.selectDistributionCandidates(item.OriginalRequest().FlowKey())
	if err != nil {
		outcome := types.QueueOutcomeRejectedOther
		if errors.Is(err, errNoShards) {
			outcome = types.QueueOutcomeRejectedCapacity
		}
		finalErr := fmt.Errorf("%w: request not accepted: %w", types.ErrRejected, err)
		item.FinalizeWithOutcome(outcome, finalErr)
		return item, finalErr
	}

	outcome, err := fc.distributeRequest(reqCtx, item, candidates)
	if err == nil {
		// Success: Ownership of the item has been transferred to the processor.
		return item, nil
	}

	// For any distribution error, the controller retains ownership and must finalize the item.
	var finalErr error
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		// We propagate the original context error here, EnqueueAndWait will rely on item.FinalState().Err.
		finalErr = err
		item.Finalize(context.Cause(reqCtx))
	} else { // e.g.,
		finalErr = fmt.Errorf("%w: request not accepted: %w", types.ErrRejected, err)
		item.FinalizeWithOutcome(outcome, finalErr)
	}
	return item, finalErr
}

// awaitFinalization blocks until an item is finalized, either by the processor (synchronously) or by the controller
// itself due to context expiry (asynchronously).
func (fc *FlowController) awaitFinalization(
	reqCtx context.Context,
	item *internal.FlowItem,
) (types.QueueOutcome, error) {
	select {
	case <-reqCtx.Done():
		// Asynchronous Finalization (Controller-initiated):
		// The request Context expired (Cancellation/TTL) while the item was being processed.
		cause := context.Cause(reqCtx)
		item.Finalize(cause)

		// The processor will eventually discard this "zombie" item during its cleanup sweep.
		finalState := item.FinalState()
		return finalState.Outcome, finalState.Err

	case finalState := <-item.Done():
		// Synchronous Finalization (Processor-initiated):
		// The processor finalized the item (Dispatch, Reject, Shutdown).
		return finalState.Outcome, finalState.Err
	}
}

// createRequestContext derives the context that governs a request's lifecycle, enforcing the TTL deadline.
func (fc *FlowController) createRequestContext(
	ctx context.Context,
	req types.FlowControlRequest,
) (context.Context, context.CancelFunc, time.Time) {
	enqueueTime := fc.clock.Now()
	effectiveTTL := req.InitialEffectiveTTL()
	if effectiveTTL <= 0 {
		effectiveTTL = fc.config.DefaultRequestTTL
	}

	if effectiveTTL > 0 {
		reqCtx, cancel := context.WithDeadlineCause(ctx, enqueueTime.Add(effectiveTTL), types.ErrTTLExpired)
		return reqCtx, cancel, enqueueTime
	}
	reqCtx, cancel := context.WithCancel(ctx)
	return reqCtx, cancel, enqueueTime
}

// candidate holds the information needed to evaluate a shard as a potential target for a request.
type candidate struct {
	processor shardProcessor
	shardID   string
	byteSize  uint64
}

// selectDistributionCandidates identifies all Active shards for the item's flow and ranks them by the current byte size
// of that flow's queue, from least to most loaded.
func (fc *FlowController) selectDistributionCandidates(key types.FlowKey) ([]candidate, error) {
	var candidates []candidate
	err := fc.registry.WithConnection(key, func(conn contracts.ActiveFlowConnection) error {
		shards := conn.ActiveShards()
		candidates = make([]candidate, len(shards))
		for i, shard := range shards {
			worker := fc.getOrStartWorker(shard)
			mq, err := shard.ManagedQueue(key)
			if err != nil {
				panic(fmt.Sprintf("invariant violation: ManagedQueue for leased flow %s failed on shard %s: %v",
					key, shard.ID(), err))
			}
			candidates[i] = candidate{worker.processor, shard.ID(), mq.ByteSize()}
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to acquire lease for flow %s: %w", key, err)
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("%w for flow %s", errNoShards, key)
	}

	slices.SortFunc(candidates, func(a, b candidate) int {
		return cmp.Compare(a.byteSize, b.byteSize)
	})

	return candidates, nil
}

// distributeRequest implements a flow-aware, two-phase "Join-Shortest-Queue-by-Bytes" (JSQ-Bytes) distribution strategy
// with graceful backpressure. It attempts to submit an item to the best-ranked candidate from the provided list.
//
// The algorithm operates as follows:
//  1. Phase 1 (Non-blocking Fast Failover): It iterates through the ranked candidates and attempts a non-blocking
//     submission. The first successful submission wins.
//  2. Phase 2 (Blocking Fallback): If all non-blocking attempts fail, it performs a single blocking submission to the
//     least-loaded candidate, providing backpressure.
//
// The provided context (ctx) is used for the blocking submission phase (SubmitOrBlock).
//
// Ownership Contract:
//   - Returns nil: Success. Ownership transferred to Processor.
//   - Returns error: Failure (Context expiry, shutdown,, etc.).
//     Ownership retained by Controller. The Controller MUST finalize the item.
func (fc *FlowController) distributeRequest(
	ctx context.Context,
	item *internal.FlowItem,
	candidates []candidate,
) (types.QueueOutcome, error) {
	reqID := item.OriginalRequest().ID()
	for _, c := range candidates {
		if err := c.processor.Submit(item); err == nil {
			return types.QueueOutcomeNotYetFinalized, nil
		}
		fc.logger.V(logutil.TRACE).Info("Processor busy during fast failover, trying next candidate",
			"shardID", c.shardID, "requestID", reqID)
	}

	// All processors are busy. Attempt a single blocking submission to the least-loaded candidate.
	bestCandidate := candidates[0]
	fc.logger.V(logutil.TRACE).Info("All processors busy, attempting blocking submit to best candidate",
		"shardID", bestCandidate.shardID, "requestID", reqID)
	err := bestCandidate.processor.SubmitOrBlock(ctx, item)
	if err != nil {
		return types.QueueOutcomeRejectedOther, fmt.Errorf("%w: request not accepted: %w", types.ErrRejected, err)
	}
	return types.QueueOutcomeNotYetFinalized, nil // Success, ownership transferred.
}

// getOrStartWorker implements the lazy-loading and startup of shard processors.
// It attempts to retrieve an existing worker for a shard. If one doesn't exist, it constructs a new worker and attempts
// to register it atomically. The worker's processor goroutine is only started *after* it has successfully been
// registered, preventing race conditions where multiple goroutines create and start the same worker.
func (fc *FlowController) getOrStartWorker(shard contracts.RegistryShard) *managedWorker {
	if w, ok := fc.workers.Load(shard.ID()); ok {
		return w.(*managedWorker)
	}

	// Construct a new worker, but do not start its processor goroutine yet.
	processorCtx, cancel := context.WithCancel(fc.parentCtx)
	processor := fc.shardProcessorFactory(
		processorCtx,
		shard,
		fc.saturationDetector,
		fc.clock,
		fc.config.ExpiryCleanupInterval,
		fc.config.EnqueueChannelBufferSize,
		fc.logger.WithValues("shardID", shard.ID()),
	)
	newWorker := &managedWorker{
		processor: processor,
		cancel:    cancel,
	}

	// Atomically load or store. This is the critical step for preventing race conditions.
	actual, loaded := fc.workers.LoadOrStore(shard.ID(), newWorker)
	if loaded {
		// Another goroutine beat us to it. The `newWorker` we created was not stored.
		// We must cancel the context we created for it to prevent a leak, but we do not need to do anything else, as its
		// processor was never started.
		cancel()
		return actual.(*managedWorker)
	}

	// We won the race. The `newWorker` was successfully stored.
	// Now, and only now, do we start the processor's long-running goroutine.
	fc.wg.Add(1)
	go func() {
		defer fc.wg.Done()
		processor.Run(processorCtx)
	}()

	return newWorker
}

// reconcileProcessors is the supervisor's core garbage collection loop.
// It fetches the current list of Active shards from the registry and removes any workers whose corresponding shards
// have been fully drained and garbage collected by the registry.
func (fc *FlowController) reconcileProcessors() {
	stats := fc.registry.ShardStats()
	shards := make(map[string]struct{}, len(stats)) // `map[shardID] -> isActive`
	for _, s := range stats {
		shards[s.ID] = struct{}{}
	}

	fc.workers.Range(func(key, value any) bool {
		shardID := key.(string)
		worker := value.(*managedWorker)

		// GC check: Is the shard no longer in the registry at all?
		if _, exists := shards[shardID]; !exists {
			fc.logger.Info("Stale worker detected for GC'd shard, shutting down.", "shardID", shardID)
			worker.cancel()
			fc.workers.Delete(shardID)
		}
		return true
	})
}

// shutdown gracefully terminates all running `shardProcessor` goroutines.
// It signals all workers to stop and waits for them to complete their shutdown procedures.
func (fc *FlowController) shutdown() {
	fc.logger.Info("Shutting down FlowController and all shard processors.")
	fc.workers.Range(func(key, value any) bool {
		shardID := key.(string)
		worker := value.(*managedWorker)
		fc.logger.V(logutil.VERBOSE).Info("Sending shutdown signal to processor", "shardID", shardID)
		worker.cancel()
		return true
	})

	fc.wg.Wait()
	fc.logger.Info("All shard processors have shut down.")
}
