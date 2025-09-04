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
	"sync/atomic"
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

	// ready is closed by the Run method once initialization is complete, including setting the `parentCtx`.
	// This acts as a memory barrier and synchronization point for all other concurrent methods.
	ready chan struct{}

	isRunning atomic.Bool
	wg        sync.WaitGroup
}

// flowControllerOption is a function that applies a configuration change to a `FlowController`.
// test-only
type flowControllerOption func(*FlowController)

// withClock returns a test-only option to inject a clock.
// test-only
func withClock(c clock.WithTicker) flowControllerOption {
	return func(fc *FlowController) {
		fc.clock = c
	}
}

// withRegistryClient returns a test-only option to inject a mock or fake registry client.
// test-only
func withRegistryClient(client registryClient) flowControllerOption {
	return func(fc *FlowController) {
		fc.registry = client
	}
}

// withShardProcessorFactory returns a test-only option to inject a processor factory.
// test-only
func withShardProcessorFactory(factory shardProcessorFactory) flowControllerOption {
	return func(fc *FlowController) {
		fc.shardProcessorFactory = factory
	}
}

// NewFlowController creates a new `FlowController` instance.
func NewFlowController(
	config Config,
	registry contracts.FlowRegistry,
	sd contracts.SaturationDetector,
	logger logr.Logger,
	opts ...flowControllerOption,
) (*FlowController, error) {
	validatedConfig, err := newConfig(config)
	if err != nil {
		return nil, fmt.Errorf("invalid flow controller configuration: %w", err)
	}

	fc := &FlowController{
		config:             *validatedConfig,
		registry:           registry,
		saturationDetector: sd,
		clock:              clock.RealClock{},
		logger:             logger.WithName("flow-controller"),
		parentCtx:          context.Background(), // Will be set in `Run`
		ready:              make(chan struct{}),
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
	return fc, nil
}

// Run starts the `FlowController`'s main reconciliation loop.
// This loop is responsible for garbage collecting workers whose shards no longer exist in the registry.
// This method blocks until the provided context is cancelled and ALL worker goroutines have fully terminated.
func (fc *FlowController) Run(ctx context.Context) {
	if !fc.isRunning.CompareAndSwap(false, true) {
		fc.logger.Error(nil, "FlowController Run loop already started or controller is shut down")
		return
	}

	fc.parentCtx = ctx
	close(fc.ready)

	fc.logger.Info("Starting FlowController reconciliation loop.")
	defer fc.logger.Info("FlowController reconciliation loop stopped.")

	ticker := fc.clock.NewTicker(fc.config.ProcessorReconciliationInterval)
	defer ticker.Stop()

	fc.reconcileProcessors() // Initial reconciliation

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
		return types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, types.ErrNilRequest)
	}
	effectiveTTL := req.InitialEffectiveTTL()
	if effectiveTTL <= 0 {
		effectiveTTL = fc.config.DefaultRequestTTL
	}
	enqueueTime := fc.clock.Now()

	for {
		if !fc.isRunning.Load() {
			return types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, types.ErrFlowControllerNotRunning)
		}

		// We must create a fresh `FlowItem` on each attempt since finalization is idempotent.
		// However, it we use the original, preserved `enqueueTime`.
		item := internal.NewItem(req, effectiveTTL, enqueueTime)
		if outcome, err := fc.distributeRequest(item); err != nil {
			return outcome, fmt.Errorf("%w: %w", types.ErrRejected, err)
		}

		finalState := <-item.Done() // finalization handles monitoring request context cancellation and TTL expiry
		if errors.Is(finalState.Err, contracts.ErrShardDraining) {
			fc.logger.V(logutil.DEBUG).Info("Shard is draining, retrying request", "requestID", req.ID())
			// Benign race with the chosen `contracts.RegistryShard` becoming Draining post selection but before the item was
			// enqueued into its respective `contracts.ManagedQueue`. Simply try again.
			continue
		}

		return finalState.Outcome, finalState.Err
	}
}

// distributeRequest selects the optimal shard processor for a given item and attempts to submit it.
//
// # Architectural Deep Dive: Achieving Emergent Fairness with Data Parallelism
//
// To achieve high throughput and prevent a central bottleneck, the `FlowController` is built on a sharded,
// data-parallel architecture. It runs multiple `internal`ShardProcessor` workers, and every logical flow is represented
// by a dedicated queue on every Active shard. This design grants the distributor maximum flexibility to route traffic
// based on real-time load.
//
// This function implements a sophisticated distribution strategy: Flow-Aware, Two-Phase Join-Shortest-Queue-by-Bytes
// (JSQ-Bytes) with Graceful Backpressure. It is designed to balance load, prevent unnecessary rejections under
// transient spikes, and create the necessary conditions for global fairness goals to emerge from local, independent
// worker actions.
//
// # The Algorithm in Detail
//
//  1. Flow-Aware Candidate Selection: For an incoming request, the controller inspects the queue depth (in bytes) for
//     that specific flow across all Active shards. It then sorts these shards from least-loaded to most-loaded,
//     creating a ranked list of candidates.
//  2. Phase 1 (Non-blocking Fast Failover): The controller iterates through the sorted candidates and attempts a
//     non-blocking `Submit()` to each. If any processor accepts the item, the operation succeeds immediately.
//     This prevents a single, momentarily busy worker from stalling the entire system.
//  3. Phase 2 (Blocking Fallback): If all processors are busy, it attempts a single, blocking `SubmitOrBlock()` on the
//     single best candidate. This provides graceful backpressure and increases the likelihood of success during
//     transient traffic bursts.
//
// # Design Rationale and Critical Assumptions
//
// ### 1. The Flow Homogeneity Assumption
//
// The first assumption is that traffic within a single logical flow is roughly homogeneous. The `types.FlowKey` is
// the primary mechanism for grouping requests that are expected to have statistically similar behavior (e.g., prefill,
// decode). For this to be effective, a flow must meaningfully represent a single workload (e.g., the same model, user
// cohort, or task type). The more closely this assumption is satisfied in practice, the more stable and predictable the
// system dynamics will be.
//
// ### Robustness Through Real-Time Adaptation
//
// The system is designed to be robust even when the homogeneity assumption is imperfect. The distribution algorithm
// does not need to predict workload characteristics; it only needs to react to their consequences in real time.
// If a shard becomes slow or congested, the backlogs of its queues will grow. The JSQ-Bytes algorithm will naturally
// observe this increase in byte size and adaptively steer new work away from the congested shard.
//
// ### 2. The Shard Homogeneity Constraint (Enabling Stateful Policies)
//
// The second, and most critical, constraint of this data-parallel design relates to the policies executed by the
// workers. The fairness (`InterFlowDispatchPolicy`) and temporal scheduling (`IntraFlowDispatchPolicy`) policies may be
// stateful (e.g., a fairness algorithm tracking historical tokens served).
//
// For the independent decisions of these stateful policies to result in coherent, globally fair outcomes, the state
// they observe on each shard must be statistically similar. This is the shard homogeneity constraint.
//
// This constraint is actively enforced by the Flow-Aware JSQ-Bytes algorithm. By constantly balancing the load for each
// flow individually, the distributor ensures that, over time, the mix of traffic on each shard is roughly proportional.
// It actively prevents one shard from becoming specialized in serving a single, dominant flow.
//
// This creates the necessary foundation for our model: local, stateful policy decisions, when aggregated across
// statistically similar shards, result in an emergent, approximately correct global fairness objective. This is key to
// unlocking scalable performance without a central, bottlenecked scheduler.
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
		return types.QueueOutcomeRejectedOther, fmt.Errorf("failed to establish connection for request %q (flow %s): %w",
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
// It attempts to retrieve an existing worker for a shard, and if one doesn't exist, it creates, starts, and
// registers it atomically.
// This ensures that workers are only created on-demand when a shard first becomes Active.
func (fc *FlowController) getOrStartWorker(shard contracts.RegistryShard) *managedWorker {
	if w, ok := fc.workers.Load(shard.ID()); ok {
		return w.(*managedWorker)
	}

	// Atomically load or store.
	// This handles the race condition where multiple goroutines try to create the same worker.
	newWorker := fc.startNewWorker(shard)
	actual, loaded := fc.workers.LoadOrStore(shard.ID(), newWorker)
	if loaded {
		// Another goroutine beat us to it; the `newWorker` we created was not stored.
		// We must clean it up immediately to prevent resource leaks.
		newWorker.cancel()
		return actual.(*managedWorker)
	}
	return newWorker
}

// startNewWorker encapsulates the logic for creating and starting a new worker goroutine.
func (fc *FlowController) startNewWorker(shard contracts.RegistryShard) *managedWorker {
	<-fc.ready // We must wait until the parent context is initialized.
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

	worker := &managedWorker{
		processor: processor,
		cancel:    cancel,
	}

	fc.wg.Add(1)
	go func() {
		defer fc.wg.Done()
		processor.Run(processorCtx)
	}()

	return worker
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
	fc.isRunning.Store(false)
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
