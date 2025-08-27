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
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/go-logr/logr"
	"k8s.io/utils/clock"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// bandStats holds the aggregated atomic statistics for a single priority band across all shards.
type bandStats struct {
	byteSize atomic.Int64
	len      atomic.Int64
}

// FlowRegistry is the concrete implementation of the `contracts.FlowRegistry` interface.
//
// # Role: The Central Orchestrator
//
// The `FlowRegistry` is the single source of truth for all configuration and the lifecycle manager for all shards and
// flow instances (identified by `types.FlowKey`). It is responsible for complex, multi-step operations such as flow
// registration, dynamic shard scaling, and coordinating garbage collection across all shards.
//
// # Concurrency Model: The Serialized Control Plane (Actor Model)
//
// To ensure correctness during complex state transitions, the `FlowRegistry` employs an Actor-like pattern. All
// administrative operations and internal state change events are serialized.
//
// A single background goroutine (the `Run` loop) processes events from the `events` channel and a periodic `gcTicker`.
// This event loop acquires the main `mu` lock before processing any event. External administrative methods (like
// `RegisterOrUpdateFlow`) also acquire this lock. This strict serialization eliminates race conditions in the control
// plane, simplifying the complex logic of the distributed state machine.
//
// # Detailed Concurrency Strategy and Locking Rules
//
// The registry employs a multi-tiered concurrency strategy:
//
//  1. Serialized Control Plane (Actor Model): Implemented by the `Run` loop and the `mu` lock.
//
//  2. Coarse-Grained Admin Lock (`FlowRegistry.mu`): Protects the core control plane state.
//
//  3. Shard-Level R/W Lock (`registryShard.mu`): Protects a single shard's metadata.
//
//  4. Queue-Level Write Lock (`managedQueue.mu`): Each `managedQueue` uses a `sync.Mutex` to protect writes, ensuring
//     strict consistency between queue contents and statistics (required for GC correctness).
//
//  5. Lock-Free Data Path (Atomics): Statistics aggregation (Shard/Registry level) uses lock-free atomics. Statistics
//     reads at the queue level are also lock-free.
//
//  6. Strict Lock Hierarchy: To prevent deadlocks, a strict acquisition hierarchy is enforced:
//     `FlowRegistry.mu` -> `registryShard.mu` -> `managedQueue.mu`
//     If multiple locks are required, they MUST be acquired in this order.
//
//  7. Blocking Rules: Goroutines MUST NOT hold the `FlowRegistry.mu` lock while attempting to send to the `events`
//     channel. Sends block if the channel is full. Blocking while holding the lock would deadlock the event loop.
//     (e.g., use the `deferredActions` pattern).
type FlowRegistry struct {
	logger logr.Logger

	// config holds the master configuration for the entire system. It is validated and defaulted at startup and used as
	// the template for creating partitioned `ShardConfig`s.
	config *Config

	// clock provides the time abstraction for the registry and its components (like `gcTracker`).
	clock clock.WithTickerAndDelayedExecution

	// mu protects administrative operations and the internal state (shard lists, `flowState`s, etc.).
	// Acquired by both external administrative methods and the internal event loop (`Run`), ensuring serialization.
	mu sync.Mutex

	// activeShards contains shards that are operational (Active).
	// A slice is used to maintain a deterministic order, which is crucial for consistent configuration partitioning
	// during scaling events.
	activeShards []*registryShard

	// drainingShards contains shards that are being gracefully shut down.
	// A map is used for efficient O(1) removal of a shard by its ID when its draining process completes.
	drainingShards map[string]*registryShard

	// allShards is a cached, combined slice of active and draining shards.
	// This cache is updated only when the set of shards changes, optimizing read-heavy operations that need to iterate
	// over all shards.
	allShards []*registryShard

	// nextShardID is a monotonically increasing counter used to generate unique, stable IDs for shards throughout the
	// lifetime of the process.
	nextShardID uint64

	// flowStates tracks the desired state and GC state of all flow instances, keyed by the immutable `types.FlowKey`.
	flowStates map[types.FlowKey]*flowState

	// gcTicker drives the periodic garbage collection cycle.
	gcTicker clock.Ticker

	// gcGeneration is a monotonically increasing counter for GC cycles, used for the "mark-and-sweep" algorithm.
	gcGeneration uint64

	// events is a channel for all internal state change events from shards and queues.
	//
	// Note: The GC system relies on exactly-once delivery of edge-triggered events; therefore, sends to this channel must
	// not be dropped. If the buffer fills, the data path will block, applying necessary backpressure to the control
	// plane.
	events chan event

	// Globally aggregated statistics. Updated atomically via lock-free propagation.
	totalByteSize atomic.Int64
	totalLen      atomic.Int64

	// perPriorityBandStats stores *bandStats, keyed by priority (`uint`).
	// The map structure is immutable after initialization; values are updated atomically.
	perPriorityBandStats map[uint]*bandStats
}

var _ contracts.FlowRegistry = &FlowRegistry{}

// RegistryOption allows configuring the `FlowRegistry` during initialization.
type RegistryOption func(*FlowRegistry)

// withClock sets the clock abstraction for deterministic testing.
// test-only
func withClock(clk clock.WithTickerAndDelayedExecution) RegistryOption {
	return func(fr *FlowRegistry) {
		if clk != nil {
			fr.clock = clk
		}
	}
}

// NewFlowRegistry creates and initializes a new `FlowRegistry` instance.
func NewFlowRegistry(config Config, logger logr.Logger, opts ...RegistryOption) (*FlowRegistry, error) {
	validatedConfig, err := NewConfig(config)
	if err != nil {
		return nil, fmt.Errorf("master configuration is invalid: %w", err)
	}

	// Buffered channel to absorb bursts of events. See comment on the struct field for concurrency notes.
	events := make(chan event, config.EventChannelBufferSize)

	fr := &FlowRegistry{
		config:               validatedConfig,
		logger:               logger.WithName("flow-registry"),
		flowStates:           make(map[types.FlowKey]*flowState),
		events:               events,
		activeShards:         []*registryShard{},
		drainingShards:       make(map[string]*registryShard),
		perPriorityBandStats: make(map[uint]*bandStats, len(validatedConfig.PriorityBands)),
		// Initialize `generation` to 1. A generation value of 0 in `flowState` is reserved as a sentinel value to indicate a
		// brand new flow that has not yet been processed by the GC loop.
		gcGeneration: 1,
	}

	for _, opt := range opts {
		opt(fr)
	}

	if fr.clock == nil {
		fr.clock = &clock.RealClock{}
	}

	fr.gcTicker = fr.clock.NewTicker(fr.config.FlowGCTimeout)

	for i := range config.PriorityBands {
		band := &config.PriorityBands[i]
		fr.perPriorityBandStats[band.Priority] = &bandStats{}
	}

	// `UpdateShardCount` handles the initial creation and populates `activeShards`.
	if err := fr.UpdateShardCount(validatedConfig.InitialShardCount); err != nil {
		fr.gcTicker.Stop()
		return nil, fmt.Errorf("failed to initialize shards: %w", err)
	}

	fr.logger.V(logging.DEFAULT).Info("FlowRegistry initialized successfully")
	return fr, nil
}

// Run starts the registry's background event processing loop. It blocks until the provided context is cancelled.
// This loop implements the serialized control plane (Actor model), handling both asynchronous signals from data plane
// components and periodic ticks for garbage collection.
func (fr *FlowRegistry) Run(ctx context.Context) {
	fr.logger.Info("Starting FlowRegistry event loop")
	defer fr.logger.Info("FlowRegistry event loop stopped")
	defer fr.gcTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-fr.gcTicker.C():
			fr.mu.Lock()
			fr.onGCTick()
			fr.mu.Unlock()
		case evt := <-fr.events:
			fr.mu.Lock()
			switch e := evt.(type) {
			case *queueStateChangedEvent:
				fr.onQueueStateChanged(e)
			case *shardStateChangedEvent:
				fr.onShardStateChanged(e)
			case *syncEvent:
				close(e.doneCh) // Signal synchronization point reached.
			}
			fr.mu.Unlock()
		}
	}
}

// RegisterOrUpdateFlow handles the registration of a new flow instance or the update of an existing instance's
// specification (for the same `types.FlowKey`). It orchestrates the creation or update atomically across all managed
// shards.
func (fr *FlowRegistry) RegisterOrUpdateFlow(spec types.FlowSpecification) error {
	if spec.Key.ID == "" {
		return fmt.Errorf("invalid flow specification: %w", contracts.ErrFlowIDEmpty)
	}

	fr.mu.Lock()
	defer fr.mu.Unlock()

	totalShardCount := len(fr.activeShards) + len(fr.drainingShards)
	components, err := fr.buildFlowComponents(spec, totalShardCount)
	if err != nil {
		return err
	}

	fr.applyFlowSynchronizationLocked(spec, components)
	return nil
}

// applyFlowSynchronizationLocked is the "commit" step of `RegisterOrUpdateFlow`.
// It updates the central `flowState` and propagates the changes to all shards.
// It expects the registry's write lock to be held.
func (fr *FlowRegistry) applyFlowSynchronizationLocked(spec types.FlowSpecification, components []flowComponents) {
	key := spec.Key
	state, exists := fr.flowStates[key]
	allShards := fr.allShards
	if !exists {
		// This is a new flow instance.
		state = newFlowState(spec, allShards)
		fr.flowStates[key] = state
	} else {
		// This is an update to an existing flow instance (e.g., policy change, when supported).
		state.update(spec)
	}

	if len(allShards) != len(components) {
		// This indicates a severe logic error during the prepare/commit phase synchronization (a race in
		// `RegisterOrUpdateFlow`).
		panic(fmt.Sprintf("invariant violation: shard/queue/policy count mismatch during commit for flow %s", spec.Key))
	}

	// Propagate the update to all shards (Active and Draining), giving each its own dedicated policy and queue instance.
	for i, shard := range allShards {
		shard.synchronizeFlow(spec, components[i].policy, components[i].queue)
	}
	fr.logger.Info("Successfully registered or updated flow instance", "flowKey", key)
}

// Stats returns globally aggregated statistics for the entire `FlowRegistry`.
//
// Note on Concurrency and Consistency: Statistics are aggregated using high-performance, lock-free atomic updates.
// The returned stats represent a near-consistent snapshot of the system's state. It is not perfectly atomic because the
// various counters are loaded independently without a global lock.
func (fr *FlowRegistry) Stats() contracts.AggregateStats {
	// Casts from `int64` to `uint64` are safe because the non-negative invariant is strictly enforced at the
	// `managedQueue` level.
	stats := contracts.AggregateStats{
		TotalCapacityBytes:   fr.config.MaxBytes,
		TotalByteSize:        uint64(fr.totalByteSize.Load()),
		TotalLen:             uint64(fr.totalLen.Load()),
		PerPriorityBandStats: make(map[uint]contracts.PriorityBandStats, len(fr.config.PriorityBands)),
	}

	for p, s := range fr.perPriorityBandStats {
		bandCfg, err := fr.config.getBandConfig(p)
		if err != nil {
			// The stats map was populated from the config, so the config must exist for this priority.
			// This indicates severe state corruption.
			panic(fmt.Sprintf("invariant violation: priority band config (%d) missing during stats aggregation: %v", p, err))
		}
		stats.PerPriorityBandStats[p] = contracts.PriorityBandStats{
			Priority:      p,
			PriorityName:  bandCfg.PriorityName,
			CapacityBytes: bandCfg.MaxBytes,
			ByteSize:      uint64(s.byteSize.Load()),
			Len:           uint64(s.len.Load()),
		}
	}
	return stats
}


// ShardStats returns a slice of statistics, one for each internal shard (Active and Draining).
func (fr *FlowRegistry) ShardStats() []contracts.ShardStats {
	// To minimize lock contention, we acquire the lock just long enough to get the combined list of shards.
	// Iterating and gathering stats (which involves reading shard-level atomics/locks) is done outside the critical
	// section.
	fr.mu.Lock()
	allShards := fr.allShards
	fr.mu.Unlock()

	shardStats := make([]contracts.ShardStats, len(allShards))
	for i, s := range allShards {
		shardStats[i] = s.Stats()
	}
	return shardStats
}

// Shards returns a slice of accessors for all internal shards (active and draining).
// Active shards always precede draining shards in the returned slice.
func (fr *FlowRegistry) Shards() []contracts.RegistryShard {
	// Similar to `ShardStats`, minimize lock contention by getting the list under lock.
	fr.mu.Lock()
	allShards := fr.allShards
	fr.mu.Unlock()

	shardContracts := make([]contracts.RegistryShard, len(allShards))
	for i, s := range allShards {
		shardContracts[i] = s
	}
	return shardContracts
}

// --- Garbage Collection ---

// garbageCollectFlowLocked orchestrates the "Trust but Verify" garbage collection of a
// single flow instance. It implements the three steps of the pattern:
// 1. Trust the eventually consistent cache.
// 2. Verify the ground truth with a "stop-the-world" pause.
// 3. Act to delete the flow if confirmed Idle.
//
// It acquires no locks itself but expects the caller to hold the main registry lock.
func (fr *FlowRegistry) garbageCollectFlowLocked(key types.FlowKey) bool {
	state, exists := fr.flowStates[key]
	if !exists {
		return false // Already deleted. Benign race.
	}

	if !state.isIdle(fr.allShards) {
		return false
	}

	logger := fr.logger.WithValues("flowKey", key, "flowID", key.ID, "priority", key.Priority)
	if !fr.verifyFlowIsTrulyIdleLocked(key, logger) {
		return false
	}

	fr.deleteFlowLocked(key, logger)
	return true
}

// verifyFlowIsTrulyIdleLocked performs the "stop-the-world" verification step of GC.
// It acquires a write lock on ALL shards, briefly pausing the data path to get a strongly consistent view of all queue
// lengths for a given flow.
//
// # Scalability Considerations
//
// This "Verify" step requires acquiring write locks on all shards (O(N)). As the shard count (N) increases, this pause
// duration may grow, potentially impacting P99 latency. This trade-off is made explicitly to guarantee correctness and
// prevent data loss.
//
// Returns true if the flow is confirmed to be empty everywhere, false otherwise.
func (fr *FlowRegistry) verifyFlowIsTrulyIdleLocked(key types.FlowKey, logger logr.Logger) bool {
	allShards := fr.allShards
	for _, shard := range allShards {
		shard.mu.Lock()
	}
	defer func() {
		for _, shard := range allShards {
			shard.mu.Unlock()
		}
	}()

	for _, shard := range allShards {
		mq, err := shard.managedQueueLocked(key)
		if err != nil {
			panic(fmt.Sprintf("invariant violation: managed queue for flow %s not found on shard %s during GC: %v",
				key, shard.ID(), err))
		}
		if mq.Len() > 0 {
			logger.V(logging.DEBUG).Info("GC aborted: Live check revealed flow instance is active", "shardID", shard.ID)
			return false
		}
	}
	return true
}

// deleteFlowLocked performs the destructive step of GC. It removes the flow's queues from all shards and deletes its
// state from the central registry map.
func (fr *FlowRegistry) deleteFlowLocked(key types.FlowKey, logger logr.Logger) {
	logger.V(logging.VERBOSE).Info("Garbage collecting inactive flow instance")
	for _, shard := range fr.allShards {
		shard.garbageCollectLocked(key)
	}
	delete(fr.flowStates, key)
	logger.V(logging.VERBOSE).Info("Successfully garbage collected flow instance")
}

// --- Event Handling (The Control Plane Loop) ---

const gcGenerationNewFlow uint64 = 0

// onGCTick implements the periodic "mark and sweep" garbage collection cycle.
//
// # Algorithm: Generational Mark-and-Sweep
//
// The GC uses a generational algorithm to ensure any idle flow survives for at least one full GC cycle before being
// collected.
//
//  1. Mark: Any flow that is either actively serving requests OR is brand new (identified by a sentinel
//     `lastActiveGeneration` value of `gcGenerationNewFlow`) is "marked" by updating its generation timestamp. This
//     design choice keeps the `flowState` object decoupled from the GC engine's internal clock.
//
//  2. Sweep: Any flow that is Idle AND was last marked before the *previous* GC cycle is eligible for collection.
//
// It expects the registry's write lock to be held.
func (fr *FlowRegistry) onGCTick() {
	previousGeneration := fr.gcGeneration
	currentGeneration := previousGeneration + 1
	logger := fr.logger.WithValues("gcGeneration", currentGeneration)
	logger.V(logging.DEBUG).Info("Starting periodic GC cycle")

	// --- Mark Phase ---
	// A flow is "marked" for survival by updating its `lastActiveGeneration`.
	for _, state := range fr.flowStates {
		if !state.isIdle(fr.allShards) {
			// If a flow is currently Active, mark it with the current generation.
			// This protects it from collection in the next cycle.
			state.lastActiveGeneration = currentGeneration
		} else if state.lastActiveGeneration == gcGenerationNewFlow {
			// If a flow is new and Idle, mark it with the *previous* generation.
			// This grants it a grace period for the *current* sweep, but correctly identifies it as a candidate for the
			// *next* sweep if it remains Idle.
			state.lastActiveGeneration = previousGeneration
		}
	}

	// --- Sweep Phase ---
	// A flow is a candidate for collection if it is Idle AND its `lastActiveGeneration` is from before the previous
	// cycle.
	var candidates []types.FlowKey
	for key, state := range fr.flowStates {
		if state.isIdle(fr.allShards) && state.lastActiveGeneration < previousGeneration {
			candidates = append(candidates, key)
		}
	}

	if len(candidates) > 0 {
		logger.V(logging.DEBUG).Info("Found Idle flow instances for GC verification", "count", len(candidates))
		for _, key := range candidates {
			fr.garbageCollectFlowLocked(key)
		}
	}

	// Finalize the cycle by updating the registry's generation.
	fr.gcGeneration = currentGeneration
}

// onQueueStateChanged handles a state change signal from a `managedQueue`. It updates the control plane's cached view
// of the flow's Idle state.
func (fr *FlowRegistry) onQueueStateChanged(e *queueStateChangedEvent) {
	fr.logger.V(logging.DEBUG).Info("Processing queue state signal",
		"shardID", e.shardID,
		"flowKey", e.key,
		"flowID", e.key.ID,
		"priority", e.key.Priority,
		"signal", e.signal,
	)

	state, ok := fr.flowStates[e.key]
	if !ok {
		return // Flow was likely already garbage collected.
	}
	state.handleQueueSignal(e.shardID, e.signal)
}

// onShardStateChanged handles a state change signal from a `registryShard`.
func (fr *FlowRegistry) onShardStateChanged(e *shardStateChangedEvent) {
	if e.signal == shardStateSignalBecameDrained {
		logger := fr.logger.WithValues("shardID", e.shardID, "signal", e.signal)
		logger.Info("Draining shard is now empty, finalizing garbage collection")

		if _, ok := fr.drainingShards[e.shardID]; !ok {
			// A shard signaled Drained but wasn't in the Draining list (e.g., it was Active, or the signal was somehow
			// processed twice despite the atomic latch).
			// The system state is potentially corrupted.
			panic(fmt.Sprintf("invariant violation: shard %s not found in draining map during GC", e.shardID))
		}
		delete(fr.drainingShards, e.shardID)

		for _, flowState := range fr.flowStates {
			flowState.purgeShard(e.shardID)
		}
		fr.updateAllShardsCacheLocked()
	}
}

// --- Callbacks (Data Plane -> Control Plane Communication) ---

// handleQueueStateSignal is the callback passed to shards to allow them to signal queue state changes.
// It sends an event to the event channel for serialized processing by the control plane.
func (fr *FlowRegistry) handleQueueStateSignal(shardID string, key types.FlowKey, signal queueStateSignal) {
	// This must block if the channel is full. Dropping events would cause state divergence and memory leaks, as the GC
	// system is edge-triggered. Blocking provides necessary backpressure from the data plane to the control plane.
	fr.events <- &queueStateChangedEvent{
		shardID: shardID,
		key:     key,
		signal:  signal,
	}
}

// handleShardStateSignal is the callback passed to shards to allow them to signal their own state changes.
func (fr *FlowRegistry) handleShardStateSignal(shardID string, signal shardStateSignal) {
	// This must also block (see `handleQueueStateSignal`).
	fr.events <- &shardStateChangedEvent{
		shardID: shardID,
		signal:  signal,
	}
}

// --- Shard Management (Scaling) ---

// UpdateShardCount dynamically adjusts the number of internal state shards.
func (fr *FlowRegistry) UpdateShardCount(n int) error {
	if n <= 0 {
		return fmt.Errorf("%w: shard count must be a positive integer, but got %d", contracts.ErrInvalidShardCount, n)
	}

	fr.mu.Lock()
	currentActiveShards := len(fr.activeShards)
	if n == currentActiveShards {
		fr.mu.Unlock()
		return nil
	}

	// deferredActions holds functions to be executed after the lock is released. This pattern is used to cleanly separate
	// state mutations (under lock) from side effects that might block (like sending to the `events` channel).
	var deferredActions []func()

	if n > currentActiveShards {
		if err := fr.executeScaleUpLocked(n); err != nil {
			fr.mu.Unlock()
			return err
		}
	} else {
		fr.executeScaleDownLocked(n, &deferredActions)
	}
	fr.mu.Unlock()

	// Execute all deferred side effects outside the lock.
	for _, action := range deferredActions {
		action()
	}
	return nil
}

// executeScaleUpLocked handles adding new shards using a "prepare-commit" pattern.
//
// First, in a "prepare" phase, all new shards are fully created and initialized in a temporary slice. This includes
// all fallible work. If any part of this phase fails, the operation is aborted without modifying the `FlowRegistry`'s
// state. If preparation succeeds, a "commit" phase atomically applies the changes to the registry.
//
// # Scalability Considerations
//
// The preparation phase synchronizes all existing flows onto the new shards. This requires O(M*K) operations
// (M=flows, K=new shards) performed while holding the main control plane lock. If M is large, this operation will block
// the control plane (including event processing) for a significant duration.
//
// Expects the registry's write lock to be held.
func (fr *FlowRegistry) executeScaleUpLocked(newTotalActive int) error {
	currentActive := len(fr.activeShards)
	numToAdd := newTotalActive - currentActive

	fr.logger.Info("Scaling up shards", "currentActive", currentActive, "newTotalActive", newTotalActive)

	// --- Prepare Phase ---
	// Create all new shards in a temporary slice. This phase is fallible and performs no mutations on the
	// `FlowRegistry`'s state.
	preparedShards := make([]*registryShard, numToAdd)
	for i := range numToAdd {
		shardID := fmt.Sprintf("shard-%d", fr.nextShardID+uint64(i))
		shardIndex := currentActive + i
		partitionedConfig := fr.config.partition(shardIndex, newTotalActive)

		callbacks := shardCallbacks{
			propagateStatsDelta: fr.propagateStatsDelta,
			signalQueueState:    fr.handleQueueStateSignal,
			signalShardState:    fr.handleShardStateSignal,
		}
		shard, err := newShard(shardID, partitionedConfig, fr.logger, callbacks, fr.config.interFlowDispatchPolicyFactory)
		if err != nil {
			return fmt.Errorf("failed to create new shard %s: %w", shardID, err)
		}

		// Synchronize all existing flows onto this newly created shard.
		for _, state := range fr.flowStates {
			components, err := fr.buildFlowComponents(state.spec, 1)
			if err != nil {
				// This is unlikely as the flow was already validated, but we handle it defensively.
				return fmt.Errorf("failed to prepare synchronization for flow %s on new shard %s: %w",
					state.spec.Key, shardID, err)
			}
			shard.synchronizeFlow(state.spec, components[0].policy, components[0].queue)
		}
		preparedShards[i] = shard
	}

	// --- Commit Phase ---
	// Preparation succeeded. Atomically apply all changes to the registry's state.
	// This phase must be infallible.
	fr.activeShards = append(fr.activeShards, preparedShards...)

	for _, shard := range preparedShards {
		for _, state := range fr.flowStates {
			state.emptyOnShards[shard.id] = true
		}
	}

	fr.nextShardID += uint64(numToAdd)
	fr.repartitionShardConfigsLocked()
	fr.updateAllShardsCacheLocked()
	return nil
}

// executeScaleDownLocked handles marking shards for graceful draining and re-partitioning.
// It appends the necessary draining actions to the `deferredActions` slice. These actions MUST be executed by the
// caller after the registry lock is released to prevent deadlocks.
// It expects the registry's write lock to be held.
func (fr *FlowRegistry) executeScaleDownLocked(newTotalActive int, deferredActions *[]func()) {
	currentActive := len(fr.activeShards)
	fr.logger.Info("Scaling down shards", "currentActive", currentActive, "newTotalActive", newTotalActive)

	// Identify the shards to drain. These are the ones at the end of the Active list.
	shardsToDrain := fr.activeShards[newTotalActive:]
	fr.activeShards = fr.activeShards[:newTotalActive]
	for _, shard := range shardsToDrain {
		fr.drainingShards[shard.id] = shard
	}

	// Defer the `markAsDraining` calls, which may block if the `events` channel is full.
	for _, shard := range shardsToDrain {
		s := shard
		*deferredActions = append(*deferredActions, func() {
			s.markAsDraining()
		})
	}

	fr.repartitionShardConfigsLocked()
	fr.updateAllShardsCacheLocked()
}

// repartitionShardConfigsLocked updates the partitioned configuration for all active shards.
// It expects the registry's write lock to be held.
func (fr *FlowRegistry) repartitionShardConfigsLocked() {
	numActive := len(fr.activeShards)
	for i, shard := range fr.activeShards {
		newPartitionedConfig := fr.config.partition(i, numActive)
		shard.updateConfig(newPartitionedConfig)
	}
}

// --- Internal Helpers ---

// flowComponents holds the set of plugin instances created for a single shard.
type flowComponents struct {
	policy framework.IntraFlowDispatchPolicy
	queue  framework.SafeQueue
}

// buildFlowComponents instantiates the necessary plugin components for a new flow instance.
// It creates a distinct instance of each component for each shard to ensure state isolation.
func (fr *FlowRegistry) buildFlowComponents(spec types.FlowSpecification, numInstances int) ([]flowComponents, error) {
	priority := spec.Key.Priority
	bandConfig, err := fr.config.getBandConfig(priority)
	if err != nil {
		return nil, fmt.Errorf("failed to get configuration for priority %d: %w", priority, err)
	}

	// TODO: When flow-level queue/policy overrides are implemented, check `spec` first.
	policyName := bandConfig.IntraFlowDispatchPolicy
	queueName := bandConfig.Queue
	allComponents := make([]flowComponents, numInstances)

	for i := range numInstances {
		policy, err := fr.config.intraFlowDispatchPolicyFactory(policyName)
		if err != nil {
			return nil, fmt.Errorf("failed to instantiate intra-flow policy %s for flow %s: %w", policyName, spec.Key, err)
		}

		q, err := fr.config.queueFactory(queueName, policy.Comparator())
		if err != nil {
			return nil, fmt.Errorf("failed to instantiate queue %s for flow %s: %w", queueName, spec.Key, err)
		}
		allComponents[i] = flowComponents{policy: policy, queue: q}
	}

	return allComponents, nil
}

// updateAllShardsCacheLocked recalculates and updates the cached `allShards` slice.
// It must be called any time the `activeShards` or `drainingShards` collections are modified.
// It expects the registry's lock to be held.
func (fr *FlowRegistry) updateAllShardsCacheLocked() {
	allShards := make([]*registryShard, 0, len(fr.activeShards)+len(fr.drainingShards))
	allShards = append(allShards, fr.activeShards...)

	// Note: Iteration over a map is non-deterministic. However, the order of Draining shards in the combined slice does
	// not impact correctness. Active shards are always first and in a stable order.
	for _, shard := range fr.drainingShards {
		allShards = append(allShards, shard)
	}
	fr.allShards = allShards
}

// propagateStatsDelta is the top-level, lock-free aggregator for all statistics changes from all shards.
// It uses atomic operations to maintain high performance under concurrent updates from multiple shards.
// As a result, its counters are eventually consistent and may be transiently inaccurate during high-contention races.
func (fr *FlowRegistry) propagateStatsDelta(priority uint, lenDelta, byteSizeDelta int64) {
	stats, ok := fr.perPriorityBandStats[priority]
	if !ok {
		// Stats are being propagated for a priority that wasn't initialized.
		panic(fmt.Sprintf("invariant violation: priority band (%d) stats missing during propagation", priority))
	}

	stats.len.Add(lenDelta)
	stats.byteSize.Add(byteSizeDelta)
	fr.totalLen.Add(lenDelta)
	fr.totalByteSize.Add(byteSizeDelta)
}
