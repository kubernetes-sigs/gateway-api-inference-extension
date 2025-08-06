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
	"slices"
	"sync"
	"sync/atomic"

	"github.com/go-logr/logr"
	"k8s.io/utils/clock"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	intra "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// bandStats holds the aggregated atomic statistics for a single priority band across all shards.
type bandStats struct {
	byteSize atomic.Uint64
	len      atomic.Uint64
}

// FlowRegistry is the concrete implementation of the `contracts.FlowRegistry` interface. It is the top-level
// administrative object and the stateful control plane for the entire flow control system.
//
// # Role: The Central Orchestrator
//
// The `FlowRegistry` is the single source of truth for all configuration and the lifecycle manager for all shards and
// flows. It is responsible for complex, multi-step operations such as flow registration, dynamic shard scaling, and
// coordinating garbage collection across all shards.
//
// # Concurrency: The Serialized Control Plane (Actor Model)
//
// To ensure correctness during complex state transitions, the `FlowRegistry` employs an Actor-like pattern. All
// administrative operations and internal state change events (e.g., GC timers, queue signals) are serialized.
//
// A single background goroutine (the `Run` loop) processes events from the `events` channel. This event loop acquires
// the main `mu` lock before processing any event. External administrative methods (like `RegisterOrUpdateFlow`) also
// acquire this lock. This strict serialization eliminates race conditions in the control plane, simplifying the complex
// logic of the distributed state machine.
//
// (See package documentation in `doc.go` for the full overview of the Architecture and Concurrency Model)
type FlowRegistry struct {
	config *Config
	logger logr.Logger

	// clock provides the time abstraction for the registry and its components (like `gcTracker`).
	clock clock.WithTickerAndDelayedExecution

	// mu protects all administrative operations and the internal state of the registry (shard lists, `flowState`s,
	// `nextShardID`).
	// It ensures that these complex, multi-step operations appear atomic to the rest of the system.
	mu sync.Mutex

	// activeShards contains shards that are currently processing requests (Active).
	// We use a slice to maintain deterministic ordering, which is crucial for configuration partitioning.
	activeShards []*registryShard

	// drainingShards contains shards that are being gracefully shut down (Draining or Drained).
	drainingShards []*registryShard

	// nextShardID is a monotonically increasing counter used to generate unique, stable IDs for shards throughout the
	// lifetime of the process.
	nextShardID uint64

	// flowStates tracks the desired state and GC state of all flows, keyed by flow ID.
	flowStates map[string]*flowState

	// gc is the decoupled timer manager.
	gc *gcTracker

	// events is a channel for all internal state change events from shards and queues.
	// This channel drives the serialized event loop (the control plane).
	//
	// CRITICAL: This is a buffered channel. Sends to this channel MUST NOT be dropped, as the GC system relies on
	// exactly-once delivery of edge-triggered events. If this buffer fills, the data path (which sends the events) WILL
	// block, applying necessary backpressure to the control plane.
	events chan event

	// Globally aggregated statistics.
	totalByteSize atomic.Uint64
	totalLen      atomic.Uint64
	// perPriorityBandStats stores *bandStats, keyed by priority (uint).
	perPriorityBandStats sync.Map
}

var _ contracts.FlowRegistry = &FlowRegistry{}

// RegistryOption allows configuring the `FlowRegistry` during initialization using functional options.
type RegistryOption func(*FlowRegistry)

// WithClock sets the clock abstraction used by the registry (primarily for GC timers).
// This is essential for deterministic testing. If `clk` is nil, the option is ignored.
func WithClock(clk clock.WithTickerAndDelayedExecution) RegistryOption {
	return func(fr *FlowRegistry) {
		if clk != nil {
			fr.clock = clk
		}
	}
}

// NewFlowRegistry creates and initializes a new `FlowRegistry` instance.
func NewFlowRegistry(
	config *Config,
	initialShardCount uint,
	logger logr.Logger,
	opts ...RegistryOption,
) (*FlowRegistry, error) {
	if err := config.validateAndApplyDefaults(); err != nil {
		return nil, fmt.Errorf("master configuration is invalid: %w", err)
	}

	// Buffered channel to absorb bursts of events. See comment on the struct field for concurrency notes.
	events := make(chan event, config.EventChannelBufferSize)

	fr := &FlowRegistry{
		config:     config,
		logger:     logger.WithName("flow-registry"),
		flowStates: make(map[string]*flowState),
		// gc initialized below after options
		events:       events,
		activeShards: []*registryShard{},
	}

	// Apply functional options (e.g., injecting a `FakeClock`)
	for _, opt := range opts {
		opt(fr)
	}

	// If no clock was provided, default to the real system clock.
	if fr.clock == nil {
		fr.clock = &clock.RealClock{}
	}

	fr.gc = newGCTracker(events, fr.clock)

	for i := range config.PriorityBands {
		band := &config.PriorityBands[i]
		fr.perPriorityBandStats.Store(band.Priority, &bandStats{})
	}

	// `UpdateShardCount` handles the initial creation and populates `activeShards`.
	if err := fr.UpdateShardCount(initialShardCount); err != nil {
		return nil, fmt.Errorf("failed to initialize shards: %w", err)
	}

	fr.logger.Info("FlowRegistry initialized successfully", "initialShardCount", initialShardCount)
	return fr, nil
}

// Run starts the registry's background event processing loop. It blocks until the provided context is cancelled.
// This loop implements the serialized control plane (Actor model).
func (fr *FlowRegistry) Run(ctx context.Context) {
	fr.logger.Info("Starting FlowRegistry event loop")
	defer fr.logger.Info("FlowRegistry event loop stopped")

	for {
		select {
		case <-ctx.Done():
			return
		case evt := <-fr.events:
			// Acquire the lock to serialize event processing with administrative operations.
			fr.mu.Lock()
			switch e := evt.(type) {
			case *gcTimerFiredEvent:
				fr.onGCTimerFired(e)
			case *queueStateChangedEvent:
				fr.onQueueStateChanged(e)
			case *shardStateChangedEvent:
				fr.onShardStateChanged(e)
			case *syncEvent:
				close(e.doneCh) // Synchronization point reached. Acknowledge the caller.
			}
			fr.mu.Unlock()
		}
	}
}

// RegisterOrUpdateFlow handles the registration of a new flow or the update of an existing flow's specification.
// It orchestrates the validation and commit phases atomically across all shards.
//
// Optimization: It employs a "prepare-commit" pattern. The potentially expensive preparation phase (plugin
// instantiation and allocation) is performed outside the main lock to minimize contention. A revalidation step ensures
// consistency if the shard count changes concurrently.
func (fr *FlowRegistry) RegisterOrUpdateFlow(spec types.FlowSpecification) error {
	if spec.ID == "" {
		return contracts.ErrFlowIDEmpty
	}

	// 1. Get a snapshot of the current total shard count under lock.
	// We need the count of ALL shards (active + draining) as the update must be applied everywhere.
	fr.mu.Lock()
	initialTotalShardCount := len(fr.activeShards) + len(fr.drainingShards)
	fr.mu.Unlock()

	// 2. Phase 1: Preparation (Outside the lock).
	// This involves potentially slow allocations and plugin initialization.
	policy, queues, err := fr.buildFlowComponents(spec, initialTotalShardCount)
	if err != nil {
		return err
	}

	// 3. Phase 2: Commit (Inside the lock).
	fr.mu.Lock()
	defer fr.mu.Unlock()

	currentTotalShardCount := len(fr.activeShards) + len(fr.drainingShards)

	// 4. Revalidation: Check if the shard count changed while we were preparing.
	if currentTotalShardCount != initialTotalShardCount {
		// Rare race condition: Shard scaling occurred concurrently.
		// Instead of re-preparing everything, we optimize by adjusting the existing `queues` slice.
		fr.logger.V(logging.DEBUG).Info("Total shard count changed during preparation, adjusting queues under lock",
			"flowID", spec.ID, "initialCount", initialTotalShardCount, "currentCount", currentTotalShardCount)
		if currentTotalShardCount > initialTotalShardCount {
			// Scale-up: Prepare queues only for the new shards.
			numNewShards := currentTotalShardCount - initialTotalShardCount
			_, newQueues, err := fr.buildFlowComponents(spec, numNewShards)
			if err != nil {
				return err // Unlikely, but handled defensively.
			}
			queues = append(queues, newQueues...)
		} else {
			// Scale-down: Truncate the slice of prepared queues.
			// This is safe because scale-down always removes shards from the end of the list, and the `allShards` slice is
			// ordered with active shards first.
			queues = queues[:currentTotalShardCount]
		}
	}

	// 5. Apply the changes. This phase is infallible.
	fr.applyFlowSynchronizationLocked(spec, policy, queues)
	return nil
}

// UpdateShardCount dynamically adjusts the number of internal state shards.
func (fr *FlowRegistry) UpdateShardCount(n uint) error {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	targetActiveShards := int(n)

	if targetActiveShards == 0 {
		return fmt.Errorf("%w: shard count must be a positive integer, but got %d", contracts.ErrInvalidShardCount, n)
	}

	currentActiveShards := len(fr.activeShards)

	if targetActiveShards == currentActiveShards {
		return nil
	}

	if targetActiveShards > currentActiveShards {
		return fr.scaleUpLocked(targetActiveShards)
	}
	return fr.scaleDownLocked(targetActiveShards)
}

// Stats returns globally aggregated statistics for the entire `FlowRegistry`.
//
// Note: This method is lock-free as it only reads atomic counters and the configuration.
func (fr *FlowRegistry) Stats() contracts.AggregateStats {
	// No lock needed here. We are reading atomics, a `sync.Map`, and the (effectively immutable) config.

	stats := contracts.AggregateStats{
		TotalCapacityBytes:   fr.config.MaxBytes,
		TotalByteSize:        fr.totalByteSize.Load(),
		TotalLen:             fr.totalLen.Load(),
		PerPriorityBandStats: make(map[uint]contracts.PriorityBandStats, len(fr.config.PriorityBands)),
	}

	fr.perPriorityBandStats.Range(func(key, value any) bool {
		priority := key.(uint)
		bandStats := value.(*bandStats)
		bandCfg, err := fr.config.getBandConfig(priority)
		if err != nil {
			// The stats map was populated from the config, so the config must exist for this priority.
			fr.logger.Error(err, "Invariant violation: priority band config missing during stats aggregation",
				"priority", priority)
			return true
		}

		stats.PerPriorityBandStats[priority] = contracts.PriorityBandStats{
			Priority:      priority,
			PriorityName:  bandCfg.PriorityName,
			CapacityBytes: bandCfg.MaxBytes,
			ByteSize:      bandStats.byteSize.Load(),
			Len:           bandStats.len.Load(),
		}
		return true
	})

	return stats
}

// getAllShardsLocked returns a combined slice of active and draining shards.
// Active shards always precede draining shards in the returned slice.
// It expects the registry's lock to be held.
func (fr *FlowRegistry) getAllShardsLocked() []*registryShard {
	allShards := make([]*registryShard, 0, len(fr.activeShards)+len(fr.drainingShards))
	allShards = append(allShards, fr.activeShards...)
	allShards = append(allShards, fr.drainingShards...)
	return allShards
}

// ShardStats returns a slice of statistics, one for each internal shard (active and draining).
func (fr *FlowRegistry) ShardStats() []contracts.ShardStats {
	// To minimize lock contention, we acquire the lock just long enough to get the combined list of shards.
	// Iterating and gathering stats (which involves reading shard-level atomics/locks) is done outside the critical
	// section.
	fr.mu.Lock()
	allShards := fr.getAllShardsLocked()
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
	allShards := fr.getAllShardsLocked()
	fr.mu.Unlock()

	// This conversion is necessary because the method signature requires a slice of interfaces.
	shardContracts := make([]contracts.RegistryShard, len(allShards))
	for i, s := range allShards {
		shardContracts[i] = s
	}
	return shardContracts
}

// --- Internal Methods ---

const shardIDFormat = "shard-%d"

// scaleUpLocked handles adding new shards to the registry.
// It expects the registry's write lock to be held.
func (fr *FlowRegistry) scaleUpLocked(newTotalActive int) error {
	currentActive := len(fr.activeShards)
	numToAdd := newTotalActive - currentActive

	fr.logger.Info("Scaling up shards", "currentActive", currentActive, "newTotalActive", newTotalActive)

	// 1. Create the new shards.
	newShards := make([]*registryShard, numToAdd)
	for i := 0; i < numToAdd; i++ {
		// Shard index is based on its position in the final active list (for partitioning).
		shardIndex := currentActive + i

		// Generate a unique, stable ID using the monotonic counter.
		shardID := fmt.Sprintf(shardIDFormat, fr.nextShardID)
		fr.nextShardID++

		// Note: The config is partitioned based on the *new* total active count.
		partitionedConfig, err := fr.config.partition(shardIndex, newTotalActive)
		if err != nil {
			return fmt.Errorf("failed to partition config for new shard %s: %w", shardID, err)
		}

		callbacks := shardCallbacks{
			propagateStatsDelta: fr.propagateStatsDelta,
			signalQueueState:    fr.handleQueueStateSignal,
			signalShardState:    fr.handleShardStateSignal,
		}
		shard, err := newShard(shardID, partitionedConfig, fr.logger, callbacks)
		if err != nil {
			return fmt.Errorf("failed to create new shard %s: %w", shardID, err)
		}

		// Synchronize all existing flows onto the new shard.
		for _, state := range fr.flowStates {
			// We only need 1 queue instance for this specific new shard.
			policy, queues, err := fr.buildFlowComponents(state.spec, 1)
			if err != nil {
				// This is unlikely as the flow was already validated, but we handle it defensively.
				return fmt.Errorf("failed to prepare synchronization for flow %q on new shard %s: %w",
					state.spec.ID, shardID, err)
			}

			shard.synchronizeFlow(state.spec, policy, queues[0])
			// Initialize the GC tracking state for the new shard.
			state.activeQueueEmptyOnShards[shardID] = true
		}
		newShards[i] = shard
	}

	// 2. Add the new shards to the active list.
	fr.activeShards = append(fr.activeShards, newShards...)

	// 3. Re-partition the config for all active shards (old and new).
	return fr.repartitionShardConfigsLocked()
}

// scaleDownLocked handles marking shards for graceful draining and re-partitioning.
// It expects the registry's write lock to be held.
func (fr *FlowRegistry) scaleDownLocked(newTotalActive int) error {
	currentActive := len(fr.activeShards)

	fr.logger.Info("Scaling down shards", "currentActive", currentActive, "newTotalActive", newTotalActive)

	// Identify the shards to drain. These are the ones at the end of the active list.
	// The slice from index `newTotalActive` to the end contains the shards to move.
	shardsToDrain := fr.activeShards[newTotalActive:]

	// Update the active list to only include the remaining active shards.
	fr.activeShards = fr.activeShards[:newTotalActive]

	// Move them to the draining list and mark them.
	fr.drainingShards = append(fr.drainingShards, shardsToDrain...)
	for _, shard := range shardsToDrain {
		shard.markAsDraining()
	}

	// Re-partition the config across the remaining active shards.
	return fr.repartitionShardConfigsLocked()
}

// applyFlowSynchronizationLocked is the "commit" step of `RegisterOrUpdateFlow`.
// It updates the central `flowState` and propagates the changes to all shards.
// It expects the registry's write lock to be held.
func (fr *FlowRegistry) applyFlowSynchronizationLocked(
	spec types.FlowSpecification,
	policy framework.IntraFlowDispatchPolicy,
	queues []framework.SafeQueue,
) {
	flowID := spec.ID
	state, exists := fr.flowStates[flowID]
	var oldPriority uint
	isPriorityChange := false

	// Get the combined list of all shards for state updates and propagation.
	allShards := fr.getAllShardsLocked()

	if !exists {
		// This is a new flow.
		state = newFlowState(spec, allShards)
		fr.flowStates[flowID] = state
	} else {
		// This is an update to an existing flow.
		if state.spec.Priority != spec.Priority {
			isPriorityChange = true
			oldPriority = state.spec.Priority
		}
		state.update(spec, allShards)
	}

	// Propagate the update to all shards (active and draining).
	// The `queues` slice must align exactly with the `allShards` slice.
	if len(allShards) != len(queues) {
		// This indicates a severe logic error during the prepare/commit phase synchronization (a race in
		// `RegisterOrUpdateFlow`).
		panic(fmt.Sprintf(
			"invariant violation: shard count (%d) and prepared queue count (%d) mismatch during commit for flow %s",
			len(allShards), len(queues), flowID))
	}

	for i, shard := range allShards {
		shard.synchronizeFlow(spec, policy, queues[i])
	}

	// If this was a priority change, attempt to GC the newly-draining queue immediately if it's already empty.
	if isPriorityChange {
		fr.garbageCollectDrainedQueueLocked(flowID, oldPriority)
	}

	// Always re-evaluate the GC state after any change.
	fr.evaluateFlowGCStateLocked(state)
	fr.logger.Info("Successfully registered or updated flow", "flowID", spec.ID, "priority", spec.Priority,
		"generation", state.generation)
}

// repartitionShardConfigsLocked updates the partitioned configuration for all active shards.
// It expects the registry's write lock to be held.
func (fr *FlowRegistry) repartitionShardConfigsLocked() error {
	numActive := len(fr.activeShards)

	for i, shard := range fr.activeShards {
		newPartitionedConfig, err := fr.config.partition(i, numActive)
		if err != nil {
			return fmt.Errorf("failed to re-partition config for active shard %s: %w", shard.id, err)
		}
		shard.updateConfig(newPartitionedConfig)
	}
	return nil
}

// buildFlowComponents centralizes the fallible logic of validating a flow spec, fetching its configuration, and
// instantiating its policy and queue. This constitutes the "validation" phase of `RegisterOrUpdateFlow`.
// This function does not require the registry lock.
func (fr *FlowRegistry) buildFlowComponents(
	spec types.FlowSpecification,
	numQueues int,
) (framework.IntraFlowDispatchPolicy, []framework.SafeQueue, error) {
	bandConfig, err := fr.config.getBandConfig(spec.Priority)
	if err != nil {
		return nil, nil, err
	}

	policy, err := intra.NewPolicyFromName(bandConfig.IntraFlowDispatchPolicy)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to instantiate intra-flow policy %q for flow %q: %w",
			bandConfig.IntraFlowDispatchPolicy, spec.ID, err)
	}

	// Perform compatibility check. (This check is also done during config validation, but repeated here defensively).
	if err := validateBandCompatibility(*bandConfig); err != nil {
		return nil, nil, err
	}

	// TODO: Optimization: Consider evolving the plugin framework to allow static capability checks (e.g.,
	// `queue.CapabilitiesByName(name)`) to avoid instantiation here if the only goal was validation. Currently, we must
	// instantiate to check capabilities and to get the required comparator.

	queueInstances := make([]framework.SafeQueue, numQueues)
	for i := range numQueues {
		q, err := queue.NewQueueFromName(bandConfig.Queue, policy.Comparator())
		if err != nil {
			// This would be a critical, unexpected error, as we've already validated the plugins exist.
			return nil, nil, fmt.Errorf("failed to instantiate queue %q for flow %q: %w", bandConfig.Queue, spec.ID, err)
		}
		queueInstances[i] = q
	}

	return policy, queueInstances, nil
}

// --- Garbage Collection and State Verification ---
// This section implements the core logic for the "Trust but Verify" garbage collection pattern.

// garbageCollectFlowLocked attempts to garbage collect an entire flow (e.g., due to inactivity).
// It implements the complete "Trust but Verify" pattern and is the sole authority on flow deletion.
// It expects the registry's write lock to be held by the caller.
func (fr *FlowRegistry) garbageCollectFlowLocked(flowID string) bool {
	// We must re-fetch the state here as the GC logic relies on it.
	state, exists := fr.flowStates[flowID]
	if !exists {
		return false // Flow was already deleted, nothing to do.
	}

	// --- "Trust but Verify" Step 1: TRUST (the cache) ---
	// Check the eventually consistent `flowState`. If the cached state already reflects activity, we can short-circuit
	// without the overhead of the expensive live check across all shards.
	if !state.isIdle(fr.activeShards) {
		// The flowState already reflects activity. This happens if the activity event was processed just before this GC
		// attempt. Abort GC.
		return false
	}

	// --- "Trust but Verify" Step 2: VERIFY (the ground truth) ---
	// The cache suggests the flow is idle. We must perform the definitive live check against the atomic counters on the
	// live `managedQueue` instances. This prevents race conditions where a flow is incorrectly GC'd just as it becomes
	// active (i.e., the GC trigger is processed before the `QueueBecameNonEmpty` signal).
	if !fr.isFlowTrulyIdleLocked(flowID) {
		// Race resolved: The flow became active just as the GC was triggered. Abort.
		fr.logger.V(logging.DEBUG).Info("GC aborted: Live check revealed flow is active", "flowID", flowID)
		return false
	}

	// --- Step 3: ACT (Destruction) ---
	fr.logger.Info("Garbage collecting inactive flow", "flowID", flowID)

	// Collect all priorities associated with this flow (active and draining).
	prioritiesToGC := make(map[uint]struct{})
	prioritiesToGC[state.spec.Priority] = struct{}{}
	for priority := range state.drainingQueuesEmptyOnShards {
		prioritiesToGC[priority] = struct{}{}
	}

	// GC all queue instances for this flow from all shards.
	for _, shard := range fr.getAllShardsLocked() {
		for priority := range prioritiesToGC {
			shard.garbageCollect(flowID, priority)
		}
	}

	delete(fr.flowStates, flowID)
	fr.gc.stop(flowID) // Ensure any timer associated with the flow is stopped.
	fr.logger.Info("Successfully garbage collected flow", "flowID", flowID)
	return true
}

// garbageCollectDrainedQueueLocked attempts to remove a specific draining queue instance.
// It implements the complete "Trust but Verify" pattern and is the sole authority on draining queue deletion.
// It expects the registry's write lock to be held.
func (fr *FlowRegistry) garbageCollectDrainedQueueLocked(flowID string, priority uint) bool {
	// We must re-fetch the state here as the GC logic relies on it.
	state, ok := fr.flowStates[flowID]
	if !ok {
		// Flow might have been GC'd concurrently by the inactivity timer.
		return false
	}

	// --- "Trust but Verify" Step 1: TRUST (the cache) ---
	// Check the eventually consistent `flowState`. If the cached state indicates the draining queue is not yet empty
	// globally, we can short-circuit without the overhead of the expensive live check.
	if !state.isDrained(priority, fr.getAllShardsLocked()) {
		// Not all shards have reported completion yet according to the cache.
		return false
	}

	// --- "Trust but Verify" Step 2: VERIFY (the ground truth) ---
	// The cache suggests the queue is drained globally. We must perform the definitive live check against the atomic
	// counters. This prevents race conditions where a draining queue is incorrectly GC'd because the control plane's
	// cached state is stale (e.g., a priority update occurs before a `QueueBecameNonEmpty` signal for the old priority
	// has been processed).
	if !fr.isFlowTrulyDrainedLocked(flowID, priority) {
		// Race resolved: An item was enqueued to the draining queue just as the GC was triggered. Abort.
		fr.logger.V(logging.DEBUG).Info("Draining queue GC aborted: Live check revealed queue is not empty",
			"flowID", flowID, "priority", priority)
		return false
	}

	// --- Step 3: ACT (Destruction) ---
	fr.logger.Info("All shards empty for draining queue, triggering garbage collection", "flowID", flowID,
		"priority", priority)

	// GC from all shards (active and draining).
	for _, shard := range fr.getAllShardsLocked() {
		shard.garbageCollect(flowID, priority)
	}
	// Remove the tracking state for this specific priority.
	delete(state.drainingQueuesEmptyOnShards, priority)
	return true
}

// evaluateFlowGCStateLocked is the single source of truth for deciding whether to start or stop a flow's inactivity GC
// timer.
// It expects the registry's write lock to be held.
func (fr *FlowRegistry) evaluateFlowGCStateLocked(state *flowState) {
	// GC evaluation only considers the state on active shards. Draining shards do not count towards activity.
	if state.isIdle(fr.activeShards) {
		fr.logger.V(logging.DEBUG).Info("Flow is now inactive globally, starting GC timer", "flowID", state.spec.ID,
			"timeout", fr.config.FlowGCTimeout, "generation", state.generation)
		fr.gc.start(state.spec.ID, state.generation, fr.config.FlowGCTimeout)
	} else {
		// We stop the timer unconditionally if the flow is active. The `gcTracker` handles the case where no timer is
		// running. We do not log this as it happens frequently on the hot path.
		fr.gc.stop(state.spec.ID)
	}
}

// isFlowTrulyIdleLocked implements the "Verify" step of the "Trust but Verify" pattern for inactivity GC.
// It performs a synchronous live check of the active queue instances across all active shards by reading their atomic
// counters.
//
// Rationale: The centralized `flowState` is eventually consistent. This check provides the definitive ground truth,
// preventing race conditions where a flow is incorrectly GC'd because an activity event hasn't yet been processed by
// the control plane (e.g., the timer event arrives before the `QueueBecameNonEmpty` signal).
//
// It expects the registry's write lock to be held.
func (fr *FlowRegistry) isFlowTrulyIdleLocked(flowID string) bool {
	// Iterate only over active shards, as activity on draining shards doesn't prevent inactivity GC.
	for _, shard := range fr.activeShards {
		// Note: `shard.ActiveManagedQueue` acquires the shard's `RLock` internally. This is safe because the lock hierarchy
		// (`Registry.mu` -> `Shard.mu`) is strictly maintained.
		mq, err := shard.ActiveManagedQueue(flowID)
		if err != nil {
			// If the flow exists in `fr.flowStates` (which it must at this point), it MUST have an active queue instance on
			// all active shards, guaranteed by `fr.mu` serialization.
			// If we cannot find it, the system state is corrupted.
			panic(fmt.Sprintf("invariant violation: active flow %s not found on shard %s during live GC check: %v",
				flowID, shard.ID(), err))
		}

		// Check the live atomic length. This is the ground truth.
		if mq.Len() > 0 {
			// Found a request in the active live queue. The flow is definitively NOT idle.
			return false
		}
	}
	// All active queues on all active shards are empty.
	return true
}

// isFlowTrulyDrainedLocked implements the "Verify" step of the "Trust but Verify" pattern for draining queue GC.
// It performs a synchronous live check of a draining queue's instances across all shards (active and draining) by
// reading their atomic counters.
//
// Rationale: This mirrors `isFlowTrulyIdleLocked`. It provides the definitive ground truth, preventing race conditions
// where a draining queue is incorrectly GC'd because the control plane's cached state is stale (e.g., a recent enqueue
// to the draining queue hasn't been processed).
//
// It expects the registry's write lock to be held.
func (fr *FlowRegistry) isFlowTrulyDrainedLocked(flowID string, priority uint) bool {
	// We must check all shards (active and draining) because the draining queue instance exists on all of them until
	// GC'd.
	for _, shard := range fr.getAllShardsLocked() {
		// Note: `shard.ManagedQueue` acquires the shard's `RLock` internally. This is safe because the lock hierarchy
		// (`Registry.mu` -> `Shard.mu`) is strictly maintained.
		mq, err := shard.ManagedQueue(flowID, priority)
		if err != nil {
			// If the flow is being tracked centrally, a queue instance MUST exist on every shard.
			// Receiving an error here, especially ErrFlowInstanceNotFound, indicates a severe state inconsistency.
			panic(fmt.Sprintf("invariant violation: unexpected error getting queue %s/%d on shard %s during live GC check: %v",
				flowID, priority, shard.ID(), err))
		}

		// Check the live atomic length. This is the ground truth.
		if mq.Len() > 0 {
			// Found an item, so it's not complete.
			return false
		}
	}
	// All instances of this draining queue on all shards are empty.
	return true
}

// --- Event Handling (The Control Plane Loop) ---
// These methods are called by the Run loop and expect the registry's write lock (`fr.mu`) to be held.

// onGCTimerFired handles a garbage collection timer expiration.
func (fr *FlowRegistry) onGCTimerFired(e *gcTimerFiredEvent) {
	state, exists := fr.flowStates[e.flowID]
	if !exists {
		// Flow was already deleted, nothing to do.
		return
	}

	// If the generation doesn't match, this is a stale timer (e.g., the flow was updated/re-registered).
	if state.generation != e.generation {
		fr.logger.V(logging.DEBUG).Info("Ignoring stale GC timer event", "flowID", e.flowID,
			"eventGeneration", e.generation, "currentGeneration", state.generation)
		// Re-evaluate the flow's current idle state. This is crucial because the flow might still be idle but the previous
		// timer was invalidated by an update. This ensures a new, correct timer is started if necessary.
		fr.evaluateFlowGCStateLocked(state)
		return
	}

	// The timer is valid for the current generation. Attempt the garbage collection.
	fr.garbageCollectFlowLocked(e.flowID)
}

// onQueueStateChanged handles a state change signal from a `managedQueue`.
func (fr *FlowRegistry) onQueueStateChanged(e *queueStateChangedEvent) {
	state, ok := fr.flowStates[e.spec.ID]
	if !ok {
		// Flow was likely already garbage collected (e.g., by inactivity timer).
		return
	}

	// Update the centralized tracking state.
	state.handleQueueSignal(e.shardID, e.spec.Priority, e.signal)

	if e.signal == queueStateSignalBecameDrained {
		// A draining queue instance signaled completion on one shard. Attempt to GC the entire draining queue globally.
		fr.garbageCollectDrainedQueueLocked(e.spec.ID, e.spec.Priority)
	} else {
		// Active flow GC evaluation (BecameEmpty/BecameNonEmpty) only considers active shards.
		// `evaluateFlowGCStateLocked` handles the logic of checking only active shards.
		fr.evaluateFlowGCStateLocked(state)
	}
}

// onShardStateChanged handles a state change signal from a registryShard.
func (fr *FlowRegistry) onShardStateChanged(e *shardStateChangedEvent) {
	if e.signal == shardStateSignalBecameDrained {
		fr.logger.Info("Draining shard is now empty, finalizing garbage collection", "shardID", e.shard.id)

		// 1. CRITICAL: Defensively purge the shard's state from all flows first.
		// This prevents memory leaks (stale shard IDs remaining in maps) even if the shard removal below fails or if the
		// signal was duplicated (which the atomic state transition should prevent, but we are defensive).
		for _, flowState := range fr.flowStates {
			flowState.purgeShard(e.shard.id)
		}

		// 2. Remove the shard from the draining list.
		oldLen := len(fr.drainingShards)
		fr.drainingShards = slices.DeleteFunc(fr.drainingShards, func(s *registryShard) bool {
			return s == e.shard
		})

		// 3. Check for invariant violation.
		if len(fr.drainingShards) == oldLen {
			// A shard signaled drained but wasn't in the draining list (e.g., it was active, or the signal was somehow
			// processed twice despite the atomic latch).
			// The system state is potentially corrupted.
			panic(fmt.Sprintf("invariant violation: shard %s not found in draining list during garbage collection",
				e.shard.id))
		}
	}
}

// --- Callbacks (Data Plane -> Control Plane Communication) ---

// handleQueueStateSignal is the callback passed to shards to allow them to signal queue state changes.
// It sends an event to the event channel for serialized processing by the control plane.
func (fr *FlowRegistry) handleQueueStateSignal(shardID string, spec types.FlowSpecification, signal queueStateSignal) {
	// This must block if the channel is full. Dropping events would cause state divergence and memory leaks, as the GC
	// system is edge-triggered. Blocking provides necessary backpressure from the data plane to the control plane.
	fr.events <- &queueStateChangedEvent{
		shardID: shardID,
		spec:    spec,
		signal:  signal,
	}
}

// handleShardStateSignal is the callback passed to shards to allow them to signal their own state changes.
func (fr *FlowRegistry) handleShardStateSignal(shard *registryShard, signal shardStateSignal) {
	// This must also block (see `handleQueueStateSignal`).
	fr.events <- &shardStateChangedEvent{
		shard:  shard,
		signal: signal,
	}
}

// propagateStatsDelta is the callback passed to shards to allow them to propagate their statistics changes up to the
// global registry. This is lock-free.
func (fr *FlowRegistry) propagateStatsDelta(priority uint, lenDelta, byteSizeDelta int64) {
	// This function uses two's complement arithmetic to atomically add or subtract from the unsigned counters.
	// Casting a negative `int64` to `uint64` results in its two's complement representation, which, when added, is
	// equivalent to subtraction. This is a standard and efficient pattern for atomic updates on unsigned integers.
	fr.totalLen.Add(uint64(lenDelta))
	fr.totalByteSize.Add(uint64(byteSizeDelta))

	if bandVal, ok := fr.perPriorityBandStats.Load(priority); ok {
		band := bandVal.(*bandStats)
		band.len.Add(uint64(lenDelta))
		band.byteSize.Add(uint64(byteSizeDelta))
	} else {
		// Stats are being propagated for a priority that wasn't initialized.
		panic(fmt.Sprintf("invariant violation: priority band (%d) stats missing during propagation", priority))
	}
}
