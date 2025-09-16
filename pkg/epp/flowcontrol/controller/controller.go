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
	"sync"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/utils/clock"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/controller/internal"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
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
	shard contracts.RegistryShard,
	dispatchFilter internal.BandFilter,
	clock clock.Clock,
	expiryCleanupInterval time.Duration,
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
		shard contracts.RegistryShard,
		dispatchFilter internal.BandFilter,
		clock clock.Clock,
		expiryCleanupInterval time.Duration,
		enqueueChannelBufferSize int,
		logger logr.Logger,
	) shardProcessor {
		return internal.NewShardProcessor(
			shard,
			dispatchFilter,
			clock,
			expiryCleanupInterval,
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
func (fc *FlowController) EnqueueAndWait(req types.FlowControlRequest) (types.QueueOutcome, error) {
	if req == nil {
		return types.QueueOutcomeRejectedOther, errors.New("request cannot be nil")
	}
	effectiveTTL := req.InitialEffectiveTTL()
	if effectiveTTL <= 0 {
		effectiveTTL = fc.config.DefaultRequestTTL
	}
	enqueueTime := fc.clock.Now()

	for {
		select {
		case <-fc.parentCtx.Done():
			return types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, types.ErrFlowControllerNotRunning)
		default:
			// The controller is running, proceed.
		}

		// We must create a fresh `FlowItem` on each attempt since finalization is idempotent.
		// However, we use the original, preserved `enqueueTime`.
		item := internal.NewItem(req, effectiveTTL, enqueueTime)
		if outcome, err := fc.distributeRequest(item); err != nil {
			return outcome, fmt.Errorf("%w: %w", types.ErrRejected, err)
		}

		// Block until the request is finalized (dispatched, rejected, or evicted).
		// The finalization logic internally monitors for context cancellation and TTL expiry.
		finalState := <-item.Done()
		if errors.Is(finalState.Err, contracts.ErrShardDraining) {
			fc.logger.V(logutil.DEBUG).Info("Shard is draining, retrying request", "requestID", req.ID())
			// Benign race with the chosen `contracts.RegistryShard` becoming Draining post selection but before the item was
			// enqueued into its respective `contracts.ManagedQueue`. Simply try again.
			continue
		}

		return finalState.Outcome, finalState.Err
	}
}

// distributeRequest implements a flow-aware, two-phase "Join-Shortest-Queue-by-Bytes" (JSQ-Bytes) distribution strategy
// with graceful backpressure. It selects the optimal worker for a given item and attempts to submit it.
//
// The algorithm operates as follows:
//  1. Candidate Selection: It identifies all Active shards for the item's flow and ranks them by the current byte size
//     of that flow's queue, from least to most loaded.
//  2. Phase 1 (Non-blocking Fast Failover): It iterates through the ranked candidates and attempts a non-blocking
//     submission. The first successful submission wins.
//  3. Phase 2 (Blocking Fallback): If all non-blocking attempts fail, it performs a single blocking submission to the
//     least-loaded candidate, providing backpressure.
func (fc *FlowController) distributeRequest(item *internal.FlowItem) (types.QueueOutcome, error) {
	key := item.OriginalRequest().FlowKey()
	reqID := item.OriginalRequest().ID()
	type candidate struct {
		processor shardProcessor
		shardID   string
		byteSize  uint64
	}
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
		return types.QueueOutcomeRejectedOther, fmt.Errorf("failed to acquire lease for request %q (flow %s): %w",
			reqID, key, err)
	}

	if len(candidates) == 0 {
		return types.QueueOutcomeRejectedCapacity, fmt.Errorf("no viable Active shards available for request %q (flow %s)",
			reqID, key)
	}

	slices.SortFunc(candidates, func(a, b candidate) int {
		return cmp.Compare(a.byteSize, b.byteSize)
	})

	// --- Phase 1: Fast, non-blocking failover attempt ---
	for _, c := range candidates {
		if err := c.processor.Submit(item); err == nil {
			return types.QueueOutcomeNotYetFinalized, nil // Success
		}
		fc.logger.V(logutil.DEBUG).Info("Processor busy during fast failover, trying next candidate",
			"shardID", c.shardID, "requestID", reqID)
	}

	// --- Phase 2: All processors busy. Attempt a single blocking send to the best candidate. ---
	bestCandidate := candidates[0]
	fc.logger.V(logutil.DEBUG).Info("All processors busy, attempting blocking submit to best candidate",
		"shardID", bestCandidate.shardID, "requestID", reqID, "queueByteSize", bestCandidate.byteSize)

	err = bestCandidate.processor.SubmitOrBlock(item.OriginalRequest().Context(), item)
	if err != nil {
		// If even the blocking attempt fails (e.g., context cancelled or processor shut down), the request is definitively
		// rejected.
		return types.QueueOutcomeRejectedCapacity, fmt.Errorf(
			"all viable shard processors are at capacity for request %q (flow %s): %w", reqID, key, err)
	}
	return types.QueueOutcomeNotYetFinalized, nil
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
	dispatchFilter := internal.NewSaturationFilter(fc.saturationDetector)
	processor := fc.shardProcessorFactory(
		shard,
		dispatchFilter,
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
