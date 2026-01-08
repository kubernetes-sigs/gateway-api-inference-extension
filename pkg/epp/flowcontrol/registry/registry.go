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
	"cmp"
	"context"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/utils/clock"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/common/util/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/intraflow"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// propagateStatsDeltaFunc defines the callback function used to propagate statistics changes (deltas) up the hierarchy
// (Queue -> Shard -> Registry).
// Implementations MUST be non-blocking (relying on atomics).
type propagateStatsDeltaFunc func(priority int, lenDelta, byteSizeDelta int64)

// bandStats holds the aggregated atomic statistics for a single priority band across all shards.
type bandStats struct {
	byteSize atomic.Int64
	len      atomic.Int64
}

// flowState tracks the lifecycle and usage of a specific flow instance.
//
// It uses a mutex-protected reference counter to arbitrate between active request processing and garbage collection.
// This structure allows the registry to safely determine if a flow is currently in use or eligible for deletion.
type flowState struct {
	key types.FlowKey

	// mu protects the lifecycle fields (leaseCount, becameIdleAt, closing).
	// We use a mutex instead of independent atomics to ensure that state transitions (e.g., Active -> Idle) are atomic
	// and consistent.
	mu sync.Mutex

	// leaseCount tracks the number of concurrent, in-flight connections using this flow.
	// - count > 0: Active. The flow is pinned and cannot be garbage collected.
	// - count == 0: Idle. The flow is eligible for garbage collection if the timeout is exceeded.
	leaseCount int

	// becameIdleAt tracks the timestamp when leaseCount last dropped to zero.
	// A zero value (time.Time{}) indicates the flow is currently Active.
	becameIdleAt time.Time

	// markedForDeletion indicates that the Garbage Collector has selected this flow for deletion.
	// If true, incoming requests must back off and allow the flow to be cleaned up.
	markedForDeletion bool

	// initialized ensures that the heavy-weight infrastructure provisioning (creating queues on shards) happens exactly
	// once per flowState instance.
	// This prevents race conditions where multiple concurrent requests might attempt to provision the same flow
	// simultaneously.
	initialized sync.Once
}

// priorityBandState tracks the lifecycle state for a dynamically provisioned priority band.
// Unlike flowState, which tracks individual flows, this tracks an entire priority level and persists
// after all flows at that priority are garbage collected to enforce idle timeout before band deletion.
type priorityBandState struct {
	priority int

	// gcLock protects the band's lifecycle state during GC operations.
	// - The Garbage Collector takes an exclusive write lock to safely delete the band.
	// - Concurrent flow creation at this priority takes a shared read lock to prevent deletion
	//   while allowing other operations to proceed concurrently.
	gcLock sync.RWMutex

	// becameIdleAt tracks the time at which all flows and queues at this priority became empty.
	// A zero value indicates the band currently has active flows or buffered requests.
	// This field is always protected by the gcLock's exclusive write lock during modifications.
	becameIdleAt time.Time
}

// FlowRegistry is the concrete implementation of the contracts.FlowRegistry interface.
//
// The FlowRegistry manages the mapping between abstract FlowKeys and the concrete managed queues distributed across
// internal shards. It serves as the single source of truth for flow control configuration and lifecycle management.
//
// # Concurrency Model
//
// The registry employs a split concurrency model to maximize throughput:
//  1. Request Hot Path (Flows): Uses lock-free atomic tracking and sync.Map for high-frequency operations
//     (Connect/Release). This allows request processing to proceed without contention from the garbage collector or
//     other flows.
//  2. Administrative Path (Topology): Uses mutex-based synchronization (fr.mu) for infrequent operations such as
//     scaling, configuration updates, or dynamic priority band provisioning.
type FlowRegistry struct {
	// --- Immutable dependencies (set at construction) ---
	config *Config
	logger logr.Logger
	clock  clock.WithTicker

	// --- Lock-free / Concurrent state (hot path) ---

	// flowStates tracks all active flow instances, keyed by FlowKey.
	// Access to this map is lock-free; lifecycle management is handled via the flowState atomics.
	flowStates sync.Map // FlowKey -> *flowState

	// priorityBandStates tracks dynamically provisioned bands, keyed by priority (int)
	priorityBandStates sync.Map // stores `int` -> *priorityBandState

	// Globally aggregated statistics, updated atomically via lock-free propagation.
	totalByteSize atomic.Int64
	totalLen      atomic.Int64

	// perPriorityBandStats tracks aggregated stats per priority.
	// Key: int (priority), Value: *bandStats
	// We use sync.Map here to allow for lock-free reads on the hot path (Stats) while allowing dynamic provisioning to
	// add new keys safely.
	perPriorityBandStats sync.Map

	// --- Administrative state (protected by `mu`) ---

	mu             sync.RWMutex
	activeShards   []*registryShard
	drainingShards map[string]*registryShard
	allShards      []*registryShard // Cached, sorted combination of Active and Draining shards
	nextShardID    uint64
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
func NewFlowRegistry(config *Config, logger logr.Logger, opts ...RegistryOption) (*FlowRegistry, error) {
	cfg := config.Clone()
	fr := &FlowRegistry{
		config:         cfg,
		logger:         logger.WithName("flow-registry"),
		activeShards:   []*registryShard{},
		drainingShards: make(map[string]*registryShard),
	}

	for _, opt := range opts {
		opt(fr)
	}
	if fr.clock == nil {
		fr.clock = &clock.RealClock{}
	}

	for prio := range cfg.PriorityBands {
		fr.perPriorityBandStats.Store(prio, &bandStats{})
	}

	if err := fr.updateShardCount(cfg.InitialShardCount); err != nil {
		return nil, fmt.Errorf("failed to initialize shards: %w", err)
	}
	fr.logger.V(logging.DEFAULT).Info("FlowRegistry initialized successfully")
	return fr, nil
}

// Run starts the registry's background garbage collection loop.
// It blocks until the provided context is cancelled.
func (fr *FlowRegistry) Run(ctx context.Context) {
	fr.logger.Info("Starting FlowRegistry background garbage collection loop")
	defer fr.logger.Info("FlowRegistry background garbage collection loop stopped")
	gcTicker := fr.clock.NewTicker(fr.config.FlowGCTimeout)
	defer gcTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-gcTicker.C():
			fr.executeGCCycle()
		}
	}
}

// --- `contracts.FlowRegistryDataPlane` Implementation ---

// WithConnection establishes a managed session for the specified flow.
//
// It guarantees that the flow's associated resources are pinned and valid for the duration of the provided callback fn.
// This method relies on an atomic leasing mechanism, ensuring that active flows are never garbage collected while
// requests are in flight.
//
// If the flow does not exist, it is provisioned Just-In-Time (JIT).
func (fr *FlowRegistry) WithConnection(key types.FlowKey, fn func(conn contracts.ActiveFlowConnection) error) error {
	if key.ID == "" {
		return contracts.ErrFlowIDEmpty
	}

	// 1. Acquire lease: Pin the flow state in memory.
	state := fr.pinActiveFlow(key)
	defer fr.releaseFlow(state)

	// 2. JIT provisioning: Ensure physical resources exist on shards.
	// We use sync.Once to ensure we only pay the initialization cost (building components, locking shards) exactly once
	// per flowState object.
	var jitErr error
	state.initialized.Do(func() {
		jitErr = fr.ensureFlowInfrastructure(key)
	})

	if jitErr != nil {
		// If provisioning failed, this state object is invalid.
		// We remove it from the map so that subsequent requests will attempt to create a fresh state object.
		fr.flowStates.Delete(key)
		return fmt.Errorf("failed to provision JIT flow resources: %w", jitErr)
	}

	// 3. Execute callback.
	// The lease is held throughout the execution of fn, preventing GC.
	return fn(&connection{registry: fr, key: key})
}

// pinActiveFlow locates or creates the flow state and increments its lease count.
//
// It uses an optimistic loop to handle race conditions where the Garbage Collector might delete the object from the map
// concurrently. It ensures that the returned state object is both authoritative (present in the map) and leased
// (count > 0).
func (fr *FlowRegistry) pinActiveFlow(key types.FlowKey) *flowState {
	for {
		val, ok := fr.flowStates.Load(key) // Optimization: Check Load first to avoid allocation on the hot path.
		if !ok {
			val, _ = fr.flowStates.LoadOrStore(key, &flowState{key: key})
		}
		state := val.(*flowState)

		state.mu.Lock()
		if state.markedForDeletion {
			// The GC has marked this flow for deletion.
			// We must back off and let it die. We will retry and create a fresh one.
			state.mu.Unlock()
			continue
		}
		state.leaseCount++
		state.becameIdleAt = time.Time{} // Mark as Active
		state.mu.Unlock()

		// Did the GC delete this object while we were acquiring it?
		currentVal, ok := fr.flowStates.Load(key)
		if !ok || currentVal != state {
			// We acquired a "stale" object. Back off and retry.
			fr.releaseFlow(state)
			continue
		}
		return state
	}
}

// releaseFlow decrements the lease count for a flow.
// If the lease count reaches zero, the flow is marked as idle with the current timestamp.
func (fr *FlowRegistry) releaseFlow(state *flowState) {
	state.mu.Lock()
	defer state.mu.Unlock()
	state.leaseCount--
	if state.leaseCount == 0 {
		state.becameIdleAt = fr.clock.Now()
	}
}

// ensureFlowInfrastructure guarantees that the Priority Band exists and that the flow's queues are synchronized across
// all active shards.
func (fr *FlowRegistry) ensureFlowInfrastructure(key types.FlowKey) error {
	// 1. Ensure Priority Band exists.
	fr.mu.RLock()
	_, exists := fr.config.PriorityBands[key.Priority]
	fr.mu.RUnlock()

	if !exists {
		if err := fr.ensurePriorityBand(key.Priority); err != nil {
			return err
		}
	}

	// Now we know the band exists (or we errored). Re-acquire Read Lock to safely read the topology and build components.

	// 2. Synchronize shards.
	// Acquire Read Lock to iterate the shard topology safely.
	fr.mu.RLock()
	defer fr.mu.RUnlock()

	components, err := fr.buildFlowComponents(key, len(fr.allShards))
	if err != nil {
		return err
	}

	for i, shard := range fr.allShards {
		shard.synchronizeFlow(key, components[i].policy, components[i].queue)
	}

	fr.logger.V(logging.DEBUG).Info("JIT provisioned flow infrastructure", "flowKey", key)
	return nil
}

// ensurePriorityBand safely provisions a new priority band.
func (fr *FlowRegistry) ensurePriorityBand(priority int) error {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	// Double-Check: Someone might have created it while we swapped locks in prepareNewFlow.
	if _, ok := fr.config.PriorityBands[priority]; ok {
		return nil
	}

	fr.logger.Info("Dynamically provisioning new priority band", "priority", priority)

	newBand := *fr.config.DefaultPriorityBand
	newBand.Priority = priority
	newBand.PriorityName = fmt.Sprintf("Dynamic-%d", priority)
	fr.config.PriorityBands[priority] = &newBand

	fr.perPriorityBandStats.LoadOrStore(priority, &bandStats{})

	fr.priorityBandStates.Store(priority, &priorityBandState{
		priority:     priority,
		becameIdleAt: time.Time{}, // Mark the priority band as Active.
	})

	fr.repartitionShardConfigsLocked()

	for _, shard := range fr.activeShards {
		shard.addPriorityBand(priority)
	}

	return nil
}

// isBandEmpty returns true if the priority band has no flows and no buffered items across all shards.
// This method checks both structural emptiness (no queues) and statistical emptiness (zero bytes/items).
// The caller is responsible for holding appropriate locks if needed.
func (fr *FlowRegistry) isBandEmpty(priority int) bool {
	// Get stable shard snapshot
	fr.mu.RLock()
	shards := fr.allShards
	fr.mu.RUnlock()

	for _, shard := range shards {
		// If the band doesn't exist on this shard it is not partitioned here.
		if val, ok := shard.priorityBands.Load(priority); ok {
			band := val.(*priorityBand)

			// Check queue count under lock
			shard.mu.RLock()
			queueCount := len(band.queues)
			shard.mu.RUnlock()

			if queueCount > 0 {
				return false
			}

			// Check atomic statistics (lock-free)
			if band.len.Load() > 0 || band.byteSize.Load() > 0 {
				return false
			}
		}
	}

	return true
}

// deletePriorityBand removes a priority band from the registry and all shards.
// This method should only be called after verifying the band is safe to delete (empty across all shards).
// Follows locking order: FlowRegistry.mu → registryShard.mu
func (fr *FlowRegistry) deletePriorityBand(priority int) {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	// Delete from registry config
	delete(fr.config.PriorityBands, priority)

	// Delete from stats tracking
	fr.perPriorityBandStats.Delete(priority)

	// Delete from all shards (both active and draining)
	for _, shard := range fr.allShards {
		shard.deletePriorityBand(priority)
	}

	// Delete lifecycle state
	fr.priorityBandStates.Delete(priority)

	fr.logger.Info("Successfully deleted priority band", "priority", priority)
}

// --- `contracts.FlowRegistryObserver` Implementation ---

// Stats returns globally aggregated statistics for the entire `FlowRegistry`.
//
// Statistics are aggregated using high-performance, lock-free atomic updates.
// The returned stats represent a near-consistent snapshot of the system's state.
func (fr *FlowRegistry) Stats() contracts.AggregateStats {
	// Casts from `int64` to `uint64` are safe because the non-negativity invariant is strictly enforced at the
	// `managedQueue` level.
	stats := contracts.AggregateStats{
		TotalCapacityBytes:   fr.config.MaxBytes,
		TotalByteSize:        uint64(fr.totalByteSize.Load()),
		TotalLen:             uint64(fr.totalLen.Load()),
		PerPriorityBandStats: make(map[int]contracts.PriorityBandStats, len(fr.config.PriorityBands)),
	}

	fr.perPriorityBandStats.Range(func(key, value any) bool {
		priority := key.(int)
		bandStats := value.(*bandStats)
		bandCfg := fr.config.PriorityBands[priority]
		stats.PerPriorityBandStats[priority] = contracts.PriorityBandStats{
			Priority:      priority,
			PriorityName:  bandCfg.PriorityName,
			CapacityBytes: bandCfg.MaxBytes,
			ByteSize:      uint64(bandStats.byteSize.Load()),
			Len:           uint64(bandStats.len.Load()),
		}
		return true
	})
	return stats
}

// ShardStats returns a slice of statistics, one for each internal shard.
func (fr *FlowRegistry) ShardStats() []contracts.ShardStats {
	fr.mu.RLock()
	allShards := fr.allShards
	fr.mu.RUnlock()

	shardStats := make([]contracts.ShardStats, len(allShards))
	for i, s := range allShards {
		shardStats[i] = s.Stats()
	}
	return shardStats
}

// --- Garbage Collection ---

// executeGCCycle orchestrates the periodic GC of Idle flows, idle priority bands, and Drained shards.
func (fr *FlowRegistry) executeGCCycle() {
	fr.logger.V(logging.DEBUG).Info("Starting periodic GC scan")
	fr.gcFlows()

	// After flows are collected, update which bands have become empty
	fr.updateIdleBands()

	// Scan and collect idle priority bands
	var bandCandidates []int
	fr.priorityBandStates.Range(func(priority, value interface{}) bool {
		state := value.(*priorityBandState)
		state.gcLock.RLock()
		// A band is a candidate if it has been idle for longer than the timeout
		if !state.becameIdleAt.IsZero() {
			if fr.clock.Since(state.becameIdleAt) > fr.config.PriorityBandGCTimeout {
				bandCandidates = append(bandCandidates, priority.(int))
			}
		}
		state.gcLock.RUnlock()
		return true
	})
	if len(bandCandidates) > 0 {
		fr.verifyAndSweepPriorityBands(bandCandidates)
	}

	// Sweep draining shards
	fr.sweepDrainingShards()
}

// gcFlows performs the Mark-and-Sweep of Idle flows.
//
// It iterates through all tracked flows and identifies candidates that have zero active leases and have exceeded the
// configured idle timeout. These flows are first removed from the internal map (Logical Delete) and then cleaned up
// from the shards (Physical Delete).
func (fr *FlowRegistry) gcFlows() {
	var flowsToClean []types.FlowKey
	fr.flowStates.Range(func(key, value interface{}) bool {
		state := value.(*flowState)
		state.mu.Lock()

		// 1. Check Lease.
		if state.leaseCount > 0 {
			state.mu.Unlock()
			return true
		}

		// 2. Check Idle Timeout.
		if state.becameIdleAt.IsZero() || fr.clock.Since(state.becameIdleAt) < fr.config.FlowGCTimeout {
			state.mu.Unlock()
			return true // Not yet expired or active.
		}

		// 3. Mark for Deletion.
		state.markedForDeletion = true
		idleTime := state.becameIdleAt // Captured for logging
		state.mu.Unlock()

		// 4. Logical Delete.
		// Remove from the map. Concurrent WithConnection calls will now create a fresh instance.
		fr.flowStates.Delete(key)
		flowsToClean = append(flowsToClean, key.(types.FlowKey))
		fr.logger.V(logging.VERBOSE).Info("Garbage collecting flow", "flowKey", key, "becameIdleAt", idleTime)
		return true
	})

	// 5. Physical Cleanup.
	// Performed outside the map iteration to avoid blocking or complex lock interactions.
	if len(flowsToClean) > 0 {
		fr.cleanupFlowResources(flowsToClean)
	}
}

// cleanupFlowResources removes queue resources from the shards for the specified flows.
func (fr *FlowRegistry) cleanupFlowResources(keys []types.FlowKey) {
	fr.mu.Lock() // Exclusive lock to prevent race with ensureFlowInfrastructure.
	defer fr.mu.Unlock()

	for _, key := range keys {
		if _, exists := fr.flowStates.Load(key); exists {
			continue // 'Zombie' flow
		}
		for _, shard := range fr.allShards {
			shard.deleteFlow(key)
		}
	}
}

// updateIdleBands updates the becameIdleAt timestamp for dynamically provisioned priority bands.
// Called after flow GC to mark bands that have become empty.
func (fr *FlowRegistry) updateIdleBands() {
	fr.priorityBandStates.Range(func(key, value interface{}) bool {
		priority := key.(int)
		state := value.(*priorityBandState)

		state.gcLock.Lock()

		isEmpty := fr.isBandEmpty(priority)

		if isEmpty && state.becameIdleAt.IsZero() {
			state.becameIdleAt = fr.clock.Now()
			fr.logger.V(logging.DEBUG).Info("Priority band became idle",
				"priority", priority, "becameIdleAt", state.becameIdleAt)
		} else if !isEmpty && !state.becameIdleAt.IsZero() {
			// Band became active again: reset timestamp
			state.becameIdleAt = time.Time{}
			fr.logger.V(logging.DEBUG).Info("Priority band became active again",
				"priority", priority)
		}

		state.gcLock.Unlock()
		return true
	})
}

// verifyAndSweepPriorityBands performs the "verify" and "sweep" phases of GC for idle priority bands.
// For each candidate, it acquires an exclusive lock on that band's state, re-verifies it is still empty, and
// then safely performs the deletion.
func (fr *FlowRegistry) verifyAndSweepPriorityBands(candidates []int) {
	fr.logger.V(logging.DEBUG).Info("Starting GC Verify and Sweep phase for priority bands", "candidateCount", len(candidates))

	var collectedCount int
	for _, priority := range candidates {
		val, ok := fr.priorityBandStates.Load(priority)
		if !ok {
			// Benign race: the band was already deleted by a previous GC cycle or another process. We can safely ignore it.
			continue
		}
		state := val.(*priorityBandState)

		// Acquire the exclusive write lock for this specific band, ensuring the state is stable for our check.
		state.gcLock.Lock()

		// Verify Phase: Double-check band is still empty
		if !fr.isBandEmpty(priority) {
			// Verification failed. The band became non-empty between our initial scan and acquiring the lock.
			// The band is Active again, so we leave it alone.
			fr.logger.V(logging.DEBUG).Info("GC of priority band aborted: re-verification failed (band is not empty)",
				"priority", priority, "becameIdleAt", state.becameIdleAt)
			state.gcLock.Unlock()
			continue
		}

		// Sweep Phase: Safe to delete
		fr.deletePriorityBand(priority)
		fr.logger.V(logging.VERBOSE).Info("Successfully verified and swept priority band",
			"priority", priority, "becameIdleAt", state.becameIdleAt)
		collectedCount++
		state.gcLock.Unlock()
	}

	fr.logger.V(logging.DEBUG).Info("GC Verify and Sweep phase completed for priority bands", "bandsCollected", collectedCount)
}

// sweepDrainingShards finalizes the removal of drained shards.
func (fr *FlowRegistry) sweepDrainingShards() {
	// Acquire a full write lock on the registry as we may be modifying the shard topology.
	fr.mu.Lock()
	defer fr.mu.Unlock()

	var shardsToDelete []string
	for id, shard := range fr.drainingShards {
		// A Draining shard is ready for GC once it is completely empty.
		// Draining shards do not accept new work (enforced at `managedQueue.Add`), so `shard.totalLen.Load()` can only
		// monotonically decrease.
		if shard.totalLen.Load() == 0 {
			shardsToDelete = append(shardsToDelete, id)
		}
	}

	if len(shardsToDelete) > 0 {
		fr.logger.V(logging.DEBUG).Info("Garbage collecting drained shards", "shardIDs", shardsToDelete)
		for _, id := range shardsToDelete {
			delete(fr.drainingShards, id)
		}
		fr.updateAllShardsCacheLocked()
	}
}

// --- Shard Management (Scaling) ---

// updateShardCount dynamically adjusts the number of internal state shards.
func (fr *FlowRegistry) updateShardCount(n int) error {
	if n <= 0 {
		return fmt.Errorf("%w: shard count must be a positive integer, but got %d", contracts.ErrInvalidShardCount, n)
	}

	// Use a full write lock as this is a major structural change to the shard topology.
	fr.mu.Lock()
	defer fr.mu.Unlock()

	currentActiveShards := len(fr.activeShards)
	if n == currentActiveShards {
		return nil
	}

	if n > currentActiveShards {
		return fr.executeScaleUpLocked(n)
	}
	fr.executeScaleDownLocked(n)
	return nil
}

// executeScaleUpLocked handles adding new shards.
// It pre-provisions all existing active flows onto the new shards to ensure continuity.
func (fr *FlowRegistry) executeScaleUpLocked(newTotalActive int) error {
	currentActive := len(fr.activeShards)
	numToAdd := newTotalActive - currentActive
	fr.logger.Info("Scaling up shards", "currentActive", currentActive, "newTotalActive", newTotalActive)

	// Prepare All New Shard Objects (Infallible):
	newShards := make([]*registryShard, numToAdd)
	for i := range numToAdd {
		shardID := fmt.Sprintf("shard-%04d", fr.nextShardID+uint64(i))
		partitionedConfig := fr.config.partition(currentActive+i, newTotalActive)
		newShards[i] = newShard(shardID, partitionedConfig, fr.logger, fr.propagateStatsDelta)
	}

	// Prepare All Components for All New Shards (Fallible):
	// Pre-build every component for every existing flow on every new shard.
	// If any single component fails to build, the entire scale-up operation is aborted, and all prepared data is
	// discarded, leaving the system state clean.
	allComponents := make(map[types.FlowKey][]flowComponents)
	var rangeErr error
	fr.flowStates.Range(func(key, _ interface{}) bool {
		flowKey := key.(types.FlowKey)
		components, err := fr.buildFlowComponents(flowKey, len(newShards))
		if err != nil {
			rangeErr = fmt.Errorf("failed to prepare components for flow %s on new shards: %w", flowKey, err)
			return false
		}
		allComponents[flowKey] = components
		return true
	})
	if rangeErr != nil {
		return rangeErr
	}

	// Commit (Infallible):
	for i, shard := range newShards {
		for key, components := range allComponents {
			shard.synchronizeFlow(key, components[i].policy, components[i].queue)
		}
	}
	fr.activeShards = append(fr.activeShards, newShards...)
	fr.nextShardID += uint64(numToAdd)
	fr.repartitionShardConfigsLocked()
	fr.updateAllShardsCacheLocked()
	return nil
}

// executeScaleDownLocked handles marking shards for graceful draining.
// Expects the registry's write lock to be held.
func (fr *FlowRegistry) executeScaleDownLocked(newTotalActive int) {
	currentActive := len(fr.activeShards)
	fr.logger.Info("Scaling down shards", "currentActive", currentActive, "newTotalActive", newTotalActive)

	shardsToDrain := fr.activeShards[newTotalActive:]
	fr.activeShards = fr.activeShards[:newTotalActive]
	for _, shard := range shardsToDrain {
		fr.drainingShards[shard.id] = shard
		shard.markAsDraining()
	}

	fr.repartitionShardConfigsLocked()
	fr.updateAllShardsCacheLocked()
}

// repartitionShardConfigsLocked updates the configuration for all active shards.
// Expects the registry's write lock to be held.
func (fr *FlowRegistry) repartitionShardConfigsLocked() {
	numActive := len(fr.activeShards)
	for i, shard := range fr.activeShards {
		newPartitionedConfig := fr.config.partition(i, numActive)
		shard.updateConfig(newPartitionedConfig)
	}
}

// --- Internal Helpers ---

// flowComponents holds the plugin instances created for a single flow on a single shard.
type flowComponents struct {
	policy framework.IntraFlowDispatchPolicy
	queue  framework.SafeQueue
}

// buildFlowComponents instantiates the necessary plugin components for a new flow instance.
// It creates a distinct instance of each component for each shard to ensure state isolation.
func (fr *FlowRegistry) buildFlowComponents(key types.FlowKey, numInstances int) ([]flowComponents, error) {
	bandConfig, ok := fr.config.PriorityBands[key.Priority]
	if !ok {
		return nil, fmt.Errorf("priority band %d not found: %w", key.Priority, contracts.ErrPriorityBandNotFound)
	}

	allComponents := make([]flowComponents, numInstances)
	for i := range numInstances {
		policy, err := intraflow.NewPolicyFromName(bandConfig.IntraFlowDispatchPolicy)
		if err != nil {
			return nil, fmt.Errorf("failed to instantiate intra-flow policy %q for flow %s: %w",
				bandConfig.IntraFlowDispatchPolicy, key, err)
		}
		q, err := queue.NewQueueFromName(bandConfig.Queue, policy.Comparator())
		if err != nil {
			return nil, fmt.Errorf("failed to instantiate queue %q for flow %s: %w",
				bandConfig.Queue, key, err)
		}
		allComponents[i] = flowComponents{policy: policy, queue: q}
	}
	return allComponents, nil
}

// updateAllShardsCacheLocked recalculates the cached `allShards` slice.
// It ensures the slice is sorted by shard ID to maintain a deterministic order.
// Expects the registry's write lock to be held.
func (fr *FlowRegistry) updateAllShardsCacheLocked() {
	allShards := make([]*registryShard, 0, len(fr.activeShards)+len(fr.drainingShards))
	allShards = append(allShards, fr.activeShards...)
	for _, shard := range fr.drainingShards {
		allShards = append(allShards, shard)
	}

	// Sort the combined slice by shard ID.
	// This provides a stable, deterministic order for all consumers of the shard list, which is critical because map
	// iteration for `drainingShards` is non-deterministic.
	// While this is a lexicographical sort, our shard ID format is padded with leading zeros (e.g., "shard-0001"),
	// ensuring that the string sort produces the same result as a natural numerical sort.
	slices.SortFunc(allShards, func(a, b *registryShard) int {
		return cmp.Compare(a.id, b.id)
	})
	fr.allShards = allShards
}

// propagateStatsDelta is the top-level, lock-free aggregator for all statistics.
func (fr *FlowRegistry) propagateStatsDelta(priority int, lenDelta, byteSizeDelta int64) {
	val, _ := fr.perPriorityBandStats.Load(priority)
	stats := val.(*bandStats)
	stats.len.Add(lenDelta)
	stats.byteSize.Add(byteSizeDelta)
	fr.totalLen.Add(lenDelta)
	fr.totalByteSize.Add(byteSizeDelta)
}
