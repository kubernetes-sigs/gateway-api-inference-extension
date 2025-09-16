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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
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

// flowState holds all tracking state for a single flow instance within the registry.
type flowState struct {
	key types.FlowKey

	// gcLock protects the flow's lifecycle state.
	// - The Garbage Collector takes an exclusive write lock to safely delete the flow.
	// - Active connections take a shared read lock for the duration of their operation, preventing the GC from running
	//   while allowing other connections to proceed concurrently.
	gcLock sync.RWMutex

	// leaseCount is an atomic reference counter for all concurrent, in-flight connections.
	// It is the sole source of truth for determining if a flow is Idle.
	leaseCount atomic.Int64

	// becameIdleAt tracks the time at which the lease count last dropped to zero.
	// A zero value indicates the flow is currently Active.
	// This field is always protected by the gcLock's exclusive write lock during modifications.
	becameIdleAt time.Time
}

// FlowRegistry is the concrete implementation of the `contracts.FlowRegistry` interface.
//
// # Role: The Central Orchestrator
//
// The `FlowRegistry` is the single source of truth for flow control configuration and the lifecycle manager for all
// shards and flow instances. It provides a highly concurrent data path for request processing while ensuring
// correctness for administrative tasks like scaling and garbage collection.
//
// # Concurrency Model: A Multi-Layered Strategy
//
// The registry is designed for high throughput by separating the concurrency domains of the request hot path, garbage
// collection, and administrative tasks.
//
//  1. `sync.Map` for `flowStates` (Hot Path): The `WithConnection` method uses a `sync.Map` for highly concurrent,
//     often lock-free, lookups and Just-In-Time registration of different flows.
//  2. `flowState.gcLock` (`sync.RWMutex`) for Per-Flow Lifecycle: Each flow has its own `RWMutex` to arbitrate between
//     active connections and the garbage collector. This surgical locking prevents GC on one flow from impacting any
//     other. The interaction is as follows:
//  3. `mu (sync.RWMutex)` for Global Topology: A single registry-wide mutex protects the overall shard topology during
//     infrequent administrative operations like scaling.
//
// # Flow Lifecycle: Lease-Based with Surgical GC
//
// A flow's lifecycle is managed by a lease-based reference count.
//
//  1. Lease Acquisition: A client calls `WithConnection` to begin a managed session. This acquires a lease by
//     incrementing an atomic counter.
//  2. Lease Release: Lease release is automatic and guaranteed. When the callback function provided to `WithConnection`
//     returns, the lease is released by decrementing the atomic counter. When the count reaches zero, the flow is
//     marked Idle with a timestamp.
//  3. Garbage Collection: A background task scans for Idle flows. To prevent a TOCTOU race, the GC acquires an
//     exclusive lock on the specific `flowState.gcLock` before re-verifying the lease count is still zero and
//     proceeding with deletion.
//
// # Locking Order
//
// To prevent deadlocks, locks MUST be acquired in the following order:
//
//  1. `FlowRegistry.mu` (Registry-level write lock)
//  2. `registryShard.mu` (Shard-level write lock)
//  3. `flowState.gcLock` (Per-flow GC lock)
type FlowRegistry struct {
	// --- Immutable dependencies (set at construction) ---
	config *Config
	logger logr.Logger
	clock  clock.WithTicker

	// --- Lock-free / Concurrent state (hot path) ---

	// flowStates tracks all flow instances, keyed by `types.FlowKey`.
	flowStates sync.Map // stores `types.FlowKey` -> *flowState

	// Globally aggregated statistics, updated atomically via lock-free propagation.
	totalByteSize        atomic.Int64
	totalLen             atomic.Int64
	perPriorityBandStats map[int]*bandStats // Keyed by priority.

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
func NewFlowRegistry(config Config, logger logr.Logger, opts ...RegistryOption) (*FlowRegistry, error) {
	cfg := config.deepCopy()
	fr := &FlowRegistry{
		config:               cfg,
		logger:               logger.WithName("flow-registry"),
		activeShards:         []*registryShard{},
		drainingShards:       make(map[string]*registryShard),
		perPriorityBandStats: make(map[int]*bandStats, len(cfg.PriorityBands)),
	}

	for _, opt := range opts {
		opt(fr)
	}
	if fr.clock == nil {
		fr.clock = &clock.RealClock{}
	}

	for i := range config.PriorityBands {
		band := &config.PriorityBands[i]
		fr.perPriorityBandStats[band.Priority] = &bandStats{}
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

// Connect establishes a session for a given flow, acquiring a lifecycle lease.
// This is the primary entry point for the data path.
// If the flow does not exist, it is registered Just-In-Time (JIT).
func (fr *FlowRegistry) WithConnection(key types.FlowKey, fn func(conn contracts.ActiveFlowConnection) error) error {
	if key.ID == "" {
		return contracts.ErrFlowIDEmpty
	}

	// --- JIT Registration ---
	val, ok := fr.flowStates.Load(key)
	if !ok {
		newFlowState, err := fr.prepareNewFlow(key)
		if err != nil {
			return fmt.Errorf("failed to prepare JIT registration for flow %s: %w", key, err)
		}

		actual, loaded := fr.flowStates.LoadOrStore(key, newFlowState)
		val = actual
		if loaded {
			// Another goroutine won the race. Use its state and discard ours.
			// If future changes make the `managedQueue` or its components more stateful (e.g., by adding background
			// goroutines, registering with a metrics system, or using `sync.Pool`), a deterministic cleanup function MUST be
			// called here to release those resources promptly and prevent leaks.
			fr.logger.V(logging.DEBUG).Info("Concurrent JIT registration detected for flow",
				"flowKey", key, "flowID", key.ID, "priority", key.Priority)
		}
	}

	// --- Lease Acquisition & Guaranteed Release ---
	state := val.(*flowState)
	state.gcLock.Lock()
	state.leaseCount.Add(1)
	state.becameIdleAt = time.Time{} // Mark the flow as Active.
	state.gcLock.Unlock()
	defer func() {
		if state.leaseCount.Add(-1) == 0 {
			// This was the last active lease; mark the flow as Idle.
			state.gcLock.Lock()
			state.becameIdleAt = fr.clock.Now()
			state.gcLock.Unlock()
		}
	}()

	// --- Callback Execution ---
	// We acquire a read lock. This has two effects:
	// 1. It allows many connections to execute this section concurrently.
	// 2. It prevents the GC from acquiring a write lock, thus guaranteeing the flow state cannot be deleted while `fn()`
	//    is running.
	state.gcLock.RLock()
	defer state.gcLock.RUnlock()
	return fn(&connection{registry: fr, key: key})
}

// prepareNewFlow creates a new `flowState` and synchronizes its queues and policies onto all existing shards.
func (fr *FlowRegistry) prepareNewFlow(key types.FlowKey) (*flowState, error) {
	// Get a stable snapshot of the shard topology.
	// An RLock is sufficient because while the list of shards must be stable, the internal state of each shard is
	// protected by its own lock.
	fr.mu.RLock()
	defer fr.mu.RUnlock()

	components, err := fr.buildFlowComponents(key, len(fr.allShards))
	if err != nil {
		return nil, err
	}

	for i, shard := range fr.allShards {
		shard.synchronizeFlow(types.FlowSpecification{Key: key}, components[i].policy, components[i].queue)
	}

	fr.logger.Info("Successfully prepared and synchronized new flow instance",
		"flowKey", key, "flowID", key.ID, "priority", key.Priority)
	return &flowState{key: key}, nil
}

// --- `contracts.FlowRegistryObserver` Implementation ---

// Stats returns globally aggregated statistics for the entire `FlowRegistry`.
//
// Statistics are aggregated using high-performance, lock-free atomic updates.
// The returned stats represent a near-consistent snapshot of the system's state.
// It is not perfectly atomic because the various counters are loaded independently without a global lock.
func (fr *FlowRegistry) Stats() contracts.AggregateStats {
	// Casts from `int64` to `uint64` are safe because the non-negativity invariant is strictly enforced at the
	// `managedQueue` level.
	stats := contracts.AggregateStats{
		TotalCapacityBytes:   fr.config.MaxBytes,
		TotalByteSize:        uint64(fr.totalByteSize.Load()),
		TotalLen:             uint64(fr.totalLen.Load()),
		PerPriorityBandStats: make(map[int]contracts.PriorityBandStats, len(fr.config.PriorityBands)),
	}

	for p, s := range fr.perPriorityBandStats {
		bandCfg, err := fr.config.getBandConfig(p)
		if err != nil {
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

// executeGCCycle orchestrates the periodic GC of Idle flows and Drained shards.
func (fr *FlowRegistry) executeGCCycle() {
	fr.logger.V(logging.DEBUG).Info("Starting periodic GC scan")
	var flowCandidates []types.FlowKey
	fr.flowStates.Range(func(key, value interface{}) bool {
		state := value.(*flowState)
		state.gcLock.RLock()
		// A flow is a candidate if its lease count is zero and its idleness timeout has expired.
		if state.leaseCount.Load() == 0 && !state.becameIdleAt.IsZero() {
			if fr.clock.Since(state.becameIdleAt) > fr.config.FlowGCTimeout {
				flowCandidates = append(flowCandidates, key.(types.FlowKey))
			}
		}
		state.gcLock.RUnlock()
		return true
	})
	if len(flowCandidates) > 0 {
		fr.verifyAndSweepFlows(flowCandidates)
	}
	fr.sweepDrainingShards()
}

// verifyAndSweepFlows performs the "verify" and "sweep" phases of GC for Idle flows.
// For each candidate, it acquires an exclusive lock on that specific flow's state, re-verifies it is still Idle, and
// then safely performs the deletion.
func (fr *FlowRegistry) verifyAndSweepFlows(candidates []types.FlowKey) {
	fr.logger.V(logging.DEBUG).Info("Starting GC Verify and Sweep phase for flows", "candidateCount", len(candidates))

	// Get a stable snapshot of the shard topology, so the list of shards does not change while we are preparing to delete
	// queues from them.
	fr.mu.RLock()
	shardsSnapshot := fr.allShards
	fr.mu.RUnlock()

	var collectedCount int
	for _, key := range candidates {
		val, ok := fr.flowStates.Load(key)
		if !ok {
			// Benign race: the flow was already deleted by a previous GC cycle or another process. We can safely ignore it.
			continue
		}
		state := val.(*flowState)

		// Acquire the exclusive write lock for this specific flow, blocking any new `Connect/Close` operations for this
		// flow only and ensuring the state is stable for our check. All other flows are unaffected.
		state.gcLock.Lock()

		// Verify Phase:
		if state.leaseCount.Load() > 0 {
			// Verification failed. A new lease was acquired between our initial scan and acquiring the lock.
			// The flow is Active again, so we leave it alone.
			fr.logger.V(logging.DEBUG).Info("GC of flow aborted: re-verification failed (flow is Active)",
				"flowKey", key, "flowID", key.ID, "priority", key.Priority,
				"leaseCount", state.leaseCount.Load(), "becameIdleAt", state.becameIdleAt)
			state.gcLock.Unlock()
			continue
		}

		// Sweep Phase:
		for _, shard := range shardsSnapshot {
			shard.deleteFlow(key)
		}
		fr.flowStates.Delete(key)
		fr.logger.V(logging.VERBOSE).Info("Successfully verified and swept flow",
			"flowKey", key, "flowID", key.ID, "priority", key.Priority, "becameIdleAt", state.becameIdleAt)
		collectedCount++
		state.gcLock.Unlock()
	}

	fr.logger.V(logging.DEBUG).Info("GC Verify and Sweep phase completed", "flowsCollected", collectedCount)
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
// It uses a "prepare-then-commit" pattern to ensure that the entire scale-up operation is transactional and never
// leaves the system in a partially-synchronized, inconsistent state.
//
// The preparation phase iterates over all existing flows once, pre-building all necessary components for every new
// shard. This requires O(M*K) operations (M=flows, K=new shards) and is performed while holding the main control plane
// lock. If M is large, this operation may block the control plane for a significant duration.
//
// Expects the registry's write lock to be held.
func (fr *FlowRegistry) executeScaleUpLocked(newTotalActive int) error {
	currentActive := len(fr.activeShards)
	numToAdd := newTotalActive - currentActive
	fr.logger.Info("Scaling up shards", "currentActive", currentActive, "newTotalActive", newTotalActive)

	// Prepare All New Shard Objects (Fallible):
	newShards := make([]*registryShard, numToAdd)
	for i := range numToAdd {
		// Using a padding of 4 allows for up to 9999 shards, which is a very safe upper bound.
		shardID := fmt.Sprintf("shard-%04d", fr.nextShardID+uint64(i))
		partitionedConfig := fr.config.partition(currentActive+i, newTotalActive)
		shard, err := newShard(
			shardID,
			partitionedConfig,
			fr.logger,
			fr.propagateStatsDelta,
			fr.config.interFlowDispatchPolicyFactory,
		)
		if err != nil {
			return fmt.Errorf("failed to create new shard object %s: %w", shardID, err)
		}
		newShards[i] = shard
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
			shard.synchronizeFlow(types.FlowSpecification{Key: key}, components[i].policy, components[i].queue)
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
	bandConfig, err := fr.config.getBandConfig(key.Priority)
	if err != nil {
		return nil, err
	}

	allComponents := make([]flowComponents, numInstances)
	for i := range numInstances {
		policy, err := fr.config.intraFlowDispatchPolicyFactory(bandConfig.IntraFlowDispatchPolicy)
		if err != nil {
			return nil, fmt.Errorf("failed to instantiate intra-flow policy %q for flow %s: %w",
				bandConfig.IntraFlowDispatchPolicy, key, err)
		}
		q, err := fr.config.queueFactory(bandConfig.Queue, policy.Comparator())
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
	stats, ok := fr.perPriorityBandStats[priority]
	if !ok {
		panic(fmt.Sprintf("invariant violation: priority band (%d) stats missing during propagation", priority))
	}

	stats.len.Add(lenDelta)
	stats.byteSize.Add(byteSizeDelta)
	fr.totalLen.Add(lenDelta)
	fr.totalByteSize.Add(byteSizeDelta)
}
