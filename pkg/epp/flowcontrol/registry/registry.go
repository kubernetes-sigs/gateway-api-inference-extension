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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
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
type flowState struct {
	leasedState
	key flowcontrol.FlowKey

	// initialized ensures that the heavy-weight infrastructure provisioning (creating queues on shards) happens exactly
	// once per flowState instance.
	// This prevents race conditions where multiple concurrent requests might attempt to provision the same flow
	// simultaneously.
	initialized sync.Once
	// initErr stores the result of the strictly-once initialization.
	// This allows concurrent waiters to see if the initialization failed.
	initErr error
}

// priorityBandState tracks the lifecycle state for a dynamically provisioned priority band.
type priorityBandState struct {
	leasedState
	priority int
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
	// Access to this map is lock-free; lifecycle management is handled via fine-grained per-flow mutexes.
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

	mu    sync.RWMutex
	shard *registryShard
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
		config: cfg,
		logger: logger.WithName("flow-registry"),
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

	if err := fr.createShard(); err != nil {
		return nil, fmt.Errorf("failed to initialize shard: %w", err)
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
//
// When a NEW flow is created, this method also increments the corresponding priority band's lease count,
// establishing the invariant: bandState.leaseCount = number of active flows at this priority.
func (fr *FlowRegistry) WithConnection(key flowcontrol.FlowKey, fn func(conn contracts.ActiveFlowConnection) error) error {
	if key.ID == "" {
		return contracts.ErrFlowIDEmpty
	}

	// 1. Acquire lease: Pin the flow state in memory.
	state, isNewFlow := pinLeasedResource(
		&fr.flowStates,
		key,
		func() *flowState { return &flowState{key: key} },
		fr.clock,
	)
	if isNewFlow {
		// If this is a newly created flow, increment the band's lease count.
		// Band leases track the number of active *flows* (not requests).
		// Every flow in the map holds exactly one band lease.
		pinLeasedResource(
			&fr.priorityBandStates,
			key.Priority,
			func() *priorityBandState { return &priorityBandState{priority: key.Priority} },
			fr.clock,
		)
	}
	defer state.unpin(fr.clock.Now())

	// 2. JIT provisioning: Ensure physical resources exist on shards.
	// We use sync.Once to ensure we only pay the initialization cost (building components, locking shards) exactly once
	// per flowState object.
	state.initialized.Do(func() {
		state.initErr = fr.ensureFlowInfrastructure(key)
	})

	if state.initErr != nil {
		// If provisioning failed, this state object is invalid.
		// We remove it from the map so that subsequent requests will attempt to create a fresh state object.
		fr.flowStates.Delete(key)

		// Release the band lease if we created the flow.
		// If JIT provisioning fails for a new flow, we must release that lease to prevent leaking band leases.
		if isNewFlow {
			if bandVal, ok := fr.priorityBandStates.Load(key.Priority); ok {
				bandVal.(*priorityBandState).unpin(fr.clock.Now())
			}
		}

		return fmt.Errorf("failed to provision JIT flow resources: %w", state.initErr)
	}

	// 3. Execute callback.
	// The flow lease is held throughout the execution of fn, preventing GC.
	return fn(&connection{registry: fr, key: key})
}

// ensureFlowInfrastructure guarantees that the Priority Band exists and that the flow's queues are synchronized across
// all active shards.
//
// NOTE: The caller (WithConnection) must already hold a lease on the priority band to prevent GC during this operation.
func (fr *FlowRegistry) ensureFlowInfrastructure(key flowcontrol.FlowKey) error {
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

	components, err := fr.buildFlowComponents(key, 1)
	if err != nil {
		return err
	}

	fr.shard.synchronizeFlow(key, components[0].policy, components[0].queue)

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

	fr.logger.V(logging.DEFAULT).Info("Dynamically provisioning new priority band", "priority", priority)

	newBand := *fr.config.DefaultPriorityBand
	newBand.Priority = priority
	fr.config.PriorityBands[priority] = &newBand

	fr.perPriorityBandStats.LoadOrStore(priority, &bandStats{})

	fr.priorityBandStates.LoadOrStore(priority, &priorityBandState{
		priority: priority,
	})

	fr.repartitionShardConfigsLocked()

	fr.shard.addPriorityBand(priority)

	return nil
}

// deletePriorityBand removes a priority band from the registry and all shards.
func (fr *FlowRegistry) deletePriorityBand(priority int) {
	fr.priorityBandStates.Delete(priority)           // Logical delete
	fr.cleanupPriorityBandResources([]int{priority}) // Physical cleanup
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
		TotalCapacityBytes:    fr.config.MaxBytes,
		TotalCapacityRequests: fr.config.MaxRequests,
		TotalByteSize:         uint64(fr.totalByteSize.Load()),
		TotalLen:              uint64(fr.totalLen.Load()),
		PerPriorityBandStats:  make(map[int]contracts.PriorityBandStats, len(fr.config.PriorityBands)),
	}

	fr.perPriorityBandStats.Range(func(key, value any) bool {
		priority := key.(int)
		bandStats := value.(*bandStats)
		bandCfg := fr.config.PriorityBands[priority]
		stats.PerPriorityBandStats[priority] = contracts.PriorityBandStats{
			Priority:         priority,
			CapacityBytes:    bandCfg.MaxBytes,
			CapacityRequests: bandCfg.MaxRequests,
			ByteSize:         uint64(bandStats.byteSize.Load()),
			Len:              uint64(bandStats.len.Load()),
		}
		return true
	})
	return stats
}

// ShardStats returns a slice of statistics, one for each internal shard.
func (fr *FlowRegistry) ShardStats() []contracts.ShardStats {
	shardStats := make([]contracts.ShardStats, 1)
	shardStats[0] = fr.shard.Stats()
	return shardStats
}

// --- Garbage Collection ---

// executeGCCycle orchestrates the periodic GC of Idle flows, idle priority bands, and Drained shards.
func (fr *FlowRegistry) executeGCCycle() {
	fr.logger.V(logging.DEBUG).Info("Starting periodic GC scan")
	fr.gcFlows()
	fr.gcPriorityBands()
}

// gcFlows removes idle flows.
func (fr *FlowRegistry) gcFlows() {
	deletedFlows := collectLeasedResources[flowcontrol.FlowKey, *flowState](
		&fr.flowStates,
		fr.config.FlowGCTimeout,
		fr.clock,
	)

	if len(deletedFlows) > 0 {
		keysToClean := make([]flowcontrol.FlowKey, 0, len(deletedFlows))
		for _, v := range deletedFlows {
			fr.logger.V(logging.VERBOSE).Info("Garbage collecting flow", "flowKey", v.key, "becameIdleAt", v.becameIdleAt)
			// Release the band lease.
			// Every flow in the map holds exactly one band lease.
			if bandVal, ok := fr.priorityBandStates.Load(v.key.Priority); ok {
				bandVal.(*priorityBandState).unpin(fr.clock.Now())
			}
			keysToClean = append(keysToClean, v.key)
		}

		fr.cleanupFlowResources(keysToClean)
	}
}

// cleanupFlowResources removes queue resources from the shards for the specified flows.
func (fr *FlowRegistry) cleanupFlowResources(keys []flowcontrol.FlowKey) {
	fr.mu.Lock() // Exclusive lock to prevent race with ensureFlowInfrastructure.
	defer fr.mu.Unlock()

	for _, key := range keys {
		if _, exists := fr.flowStates.Load(key); exists {
			continue // 'Zombie' flow
		}
		fr.shard.deleteFlow(key)
	}
}

// gcPriorityBands removes idle priority bands.
func (fr *FlowRegistry) gcPriorityBands() {
	deletedBands := collectLeasedResources[int, *priorityBandState](
		&fr.priorityBandStates,
		fr.config.PriorityBandGCTimeout,
		fr.clock,
	)

	if len(deletedBands) > 0 {
		keysToClean := make([]int, 0, len(deletedBands))
		for _, v := range deletedBands {
			fr.logger.V(logging.VERBOSE).Info("Garbage collecting priority band",
				"priority", v.priority, "becameIdleAt", v.becameIdleAt)
			keysToClean = append(keysToClean, v.priority)
		}
		fr.cleanupPriorityBandResources(keysToClean)
	}
}

// cleanupPriorityBandResources removes priority band configuration and resources from the registry and all shards.
func (fr *FlowRegistry) cleanupPriorityBandResources(priorities []int) {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	for _, priority := range priorities {
		// Zombie protection: verify band was actually deleted from map
		if _, exists := fr.priorityBandStates.Load(priority); exists {
			continue
		}

		// Delete from registry config
		delete(fr.config.PriorityBands, priority)

		// Delete from stats tracking
		fr.perPriorityBandStats.Delete(priority)

		// Delete from the shard
		fr.shard.deletePriorityBand(priority)

		fr.logger.V(logging.DEFAULT).Info("Successfully deleted priority band", "priority", priority)
	}
}

// --- Shard Management (Scaling) ---

// createShard creates the shard.
func (fr *FlowRegistry) createShard() error {
	// Use a full write lock as this is a major structural change to the shard topology.
	fr.mu.Lock()
	defer fr.mu.Unlock()

	// Prepare Shard Object (Infallible)
	partitionedConfig := fr.config.partition(0, 1)
	shard := newShard("shard-0", partitionedConfig, fr.logger, fr.propagateStatsDelta)

	// Prepare All Components for All New Shards (Fallible):
	// Pre-build every component for every existing flow on every new shard.
	// If any single component fails to build, the entire scale-up operation is aborted, and all prepared data is
	// discarded, leaving the system state clean.
	allComponents := make(map[flowcontrol.FlowKey][]flowComponents)
	var rangeErr error
	fr.flowStates.Range(func(key, _ any) bool {
		flowKey := key.(flowcontrol.FlowKey)
		components, err := fr.buildFlowComponents(flowKey, 1)
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
	for key, components := range allComponents {
		shard.synchronizeFlow(key, components[0].policy, components[0].queue)
	}
	fr.shard = shard
	fr.repartitionShardConfigsLocked()
	return nil
}

// repartitionShardConfigsLocked updates the configuration for all active shards.
// Expects the registry's write lock to be held.
func (fr *FlowRegistry) repartitionShardConfigsLocked() {
	newPartitionedConfig := fr.config.partition(0, 1)
	fr.shard.updateConfig(newPartitionedConfig)
}

// --- Internal Helpers ---

// flowComponents holds the plugin instances created for a single flow on a single shard.
type flowComponents struct {
	policy flowcontrol.OrderingPolicy
	queue  contracts.SafeQueue
}

// buildFlowComponents instantiates the necessary plugin components for a new flow instance.
// It creates a distinct instance of each component for each shard to ensure state isolation.
func (fr *FlowRegistry) buildFlowComponents(key flowcontrol.FlowKey, numInstances int) ([]flowComponents, error) {
	bandConfig, ok := fr.config.PriorityBands[key.Priority]
	if !ok {
		return nil, fmt.Errorf("priority band %d not found: %w", key.Priority, contracts.ErrPriorityBandNotFound)
	}

	allComponents := make([]flowComponents, numInstances)
	for i := range numInstances {
		q, err := queue.NewQueueFromName(bandConfig.Queue, bandConfig.OrderingPolicy)
		if err != nil {
			return nil, fmt.Errorf("failed to instantiate queue %q for flow %s: %w",
				bandConfig.Queue, key, err)
		}
		allComponents[i] = flowComponents{policy: bandConfig.OrderingPolicy, queue: q}
	}
	return allComponents, nil
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
