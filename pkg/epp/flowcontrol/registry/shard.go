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
	"fmt"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/go-logr/logr"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/interflow"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// priorityBand holds all `managedQueues` and configuration for a single priority level within a shard.
type priorityBand struct {
	// --- Immutable (set at construction) ---

	// config is the local copy of the band's definition.
	// It is updated during dynamic scaling events (updateConfig), protected by the parent shard's mutex.
	config                  PriorityBandConfig
	interFlowDispatchPolicy framework.InterFlowDispatchPolicy

	// --- State Protected by the parent shard's `mu` ---

	// queues holds all `managedQueue` instances within this band, keyed by their logical `ID` string.
	// The priority is implicit from the parent `priorityBand`.
	queues map[string]*managedQueue

	// --- Concurrent-Safe State (Atomics) ---

	// Band-level statistics, updated via lock-free propagation from child queues.
	byteSize atomic.Int64
	len      atomic.Int64
}

// registryShard implements the `contracts.RegistryShard` interface.
//
// # Role: The Data Plane Slice
//
// It represents a single, concurrent-safe slice of the registry's total state, acting as an independent, parallel
// execution unit. It provides a read-optimized view for a `controller.FlowController` worker, partitioning the overall
// system state to enable horizontal scalability.
//
// # Concurrency Model: Hybrid Lock/Lock-Free
//
// The registryShard optimizes for the hot path (request processing) while ensuring safety for the cold path (dynamic
// provisioning):
//
//   - sync.Map (Hot Path): The priorityBands map holds the active queues. It allows lock-free lookups during every
//     enqueue/dequeue operation and safe concurrent writes when new priority bands are dynamically provisioned.
//   - sync.RWMutex (Cold Path): Protects the shard's configuration state (`config`) and the orderedPriorityLevels
//     slice. These are read frequently by administrative processes (like stats scraping) but modified rarely (only
//     during scaling or dynamic provisioning).
//   - Atomics: Aggregated statistics and lifecycle flags use atomic operations for zero-contention updates.
type registryShard struct {
	// --- Immutable Identity & Dependencies (set at construction) ---
	id           string
	logger       logr.Logger
	onStatsDelta propagateStatsDeltaFunc

	// --- Configuration State (Protected by `mu`) ---

	// mu protects the shard's configuration and ordered topology lists.
	mu sync.RWMutex

	// config holds the partitioned configuration for this shard.
	config *ShardConfig

	// orderedPriorityLevels is a sorted list of active priority levels.
	// It is updated dynamically when new bands are provisioned.
	orderedPriorityLevels []int

	// --- Operational State (Concurrent-Safe / Lock-Free) ---

	// priorityBands is the primary container for all managed queues on this shard.
	// We use sync.Map to allow lock-free lookups on the hot path (Stats/Propagation) while enabling safe dynamic addition
	// of new priority bands.
	// Key: int (priority), Value: *priorityBand
	priorityBands sync.Map

	// isDraining indicates if the shard is gracefully shutting down.
	isDraining atomic.Bool

	// Shard-level statistics, updated via lock-free propagation from child queues.
	totalByteSize atomic.Int64
	totalLen      atomic.Int64
}

var _ contracts.RegistryShard = &registryShard{}

// newShard creates a new `registryShard` instance from a partitioned configuration.
func newShard(
	id string,
	config *ShardConfig,
	logger logr.Logger,
	onStatsDelta propagateStatsDeltaFunc,
) (*registryShard, error) {
	shardLogger := logger.WithName("registry-shard").WithValues("shardID", id)
	s := &registryShard{
		id:           id,
		logger:       shardLogger,
		config:       config,
		onStatsDelta: onStatsDelta,
	}

	for _, bandConfig := range config.PriorityBands {
		interPolicy, err := interflow.NewPolicyFromName(bandConfig.InterFlowDispatchPolicy)
		if err != nil {
			return nil, fmt.Errorf("failed to create inter-flow policy %q for priority band %d: %w",
				bandConfig.InterFlowDispatchPolicy, bandConfig.Priority, err)
		}

		band := &priorityBand{
			config:                  *bandConfig,
			queues:                  make(map[string]*managedQueue),
			interFlowDispatchPolicy: interPolicy,
		}
		s.priorityBands.Store(bandConfig.Priority, band)
		s.orderedPriorityLevels = append(s.orderedPriorityLevels, bandConfig.Priority)
	}

	s.sortPriorityLevels()
	s.logger.V(logging.DEFAULT).Info("Registry shard initialized successfully",
		"orderedPriorities", s.orderedPriorityLevels)
	return s, nil
}

// addPriorityBand dynamically provisions a new priority band on this shard.
// It looks up the definition in s.config, which must have been updated by the Registry via updateConfig/repartition.
func (s *registryShard) addPriorityBand(priority int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Idempotency check.
	if _, ok := s.priorityBands.Load(priority); ok {
		return
	}

	// Lookup definition from local Config (populated by repartitionShardConfigsLocked)
	bandConfig := s.config.PriorityBands[priority]

	interPolicy, _ := interflow.NewPolicyFromName(bandConfig.InterFlowDispatchPolicy)
	band := &priorityBand{
		config:                  *bandConfig,
		queues:                  make(map[string]*managedQueue),
		interFlowDispatchPolicy: interPolicy,
	}
	s.priorityBands.Store(priority, band)

	s.orderedPriorityLevels = append(s.orderedPriorityLevels, priority)
	s.sortPriorityLevels()

	s.logger.Info("Dynamically added priority band", "priority", priority)
}

// sortPriorityLevels sorts the orderedPriorityLevels slice in descending order (highest priority first).
// Expects the shard lock to be held.
func (s *registryShard) sortPriorityLevels() {
	sort.Slice(s.orderedPriorityLevels, func(i, j int) bool {
		return s.orderedPriorityLevels[i] > s.orderedPriorityLevels[j]
	})
}

// ID returns the unique identifier for this shard.
func (s *registryShard) ID() string { return s.id }

// IsActive returns true if the shard is active and accepting new requests.
// This is a lock-free read, making it efficient for the hot path.
func (s *registryShard) IsActive() bool {
	return !s.isDraining.Load()
}

// ManagedQueue retrieves a specific `contracts.ManagedQueue` instance from this shard.
func (s *registryShard) ManagedQueue(key types.FlowKey) (contracts.ManagedQueue, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	val, ok := s.priorityBands.Load(key.Priority)
	if !ok {
		return nil, fmt.Errorf("failed to get managed queue for flow %q: %w", key, contracts.ErrPriorityBandNotFound)
	}
	band := val.(*priorityBand)

	mq, ok := band.queues[key.ID]
	if !ok {
		return nil, fmt.Errorf("failed to get managed queue for flow %q: %w", key, contracts.ErrFlowInstanceNotFound)
	}
	return mq, nil
}

// IntraFlowDispatchPolicy retrieves a flow's configured `framework.IntraFlowDispatchPolicy`.
func (s *registryShard) IntraFlowDispatchPolicy(key types.FlowKey) (framework.IntraFlowDispatchPolicy, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	val, ok := s.priorityBands.Load(key.Priority)
	if !ok {
		return nil, fmt.Errorf("failed to get intra-flow policy for flow %q: %w", key, contracts.ErrPriorityBandNotFound)
	}
	band := val.(*priorityBand)

	mq, ok := band.queues[key.ID]
	if !ok {
		return nil, fmt.Errorf("failed to get intra-flow policy for flow %q: %w", key, contracts.ErrFlowInstanceNotFound)
	}
	// The policy is stored on the `managedQueue` and is immutable after creation.
	return mq.dispatchPolicy, nil
}

// InterFlowDispatchPolicy retrieves a priority band's configured `framework.InterFlowDispatchPolicy`.
// This read is lock-free as the policy instance is immutable after the shard is initialized.
func (s *registryShard) InterFlowDispatchPolicy(priority int) (framework.InterFlowDispatchPolicy, error) {
	val, ok := s.priorityBands.Load(priority)
	if !ok {
		return nil, fmt.Errorf("failed to get inter-flow policy for priority %d: %w",
			priority, contracts.ErrPriorityBandNotFound)
	}
	band := val.(*priorityBand)
	return band.interFlowDispatchPolicy, nil
}

// PriorityBandAccessor retrieves a read-only view for a given priority level.
func (s *registryShard) PriorityBandAccessor(priority int) (framework.PriorityBandAccessor, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	val, ok := s.priorityBands.Load(priority)
	if !ok {
		return nil, fmt.Errorf("failed to get priority band accessor for priority %d: %w",
			priority, contracts.ErrPriorityBandNotFound)
	}
	band := val.(*priorityBand)
	return &priorityBandAccessor{shard: s, band: band}, nil
}

// AllOrderedPriorityLevels returns a cached, sorted slice of all configured priority levels for this shard.
// This is a lock-free read.
func (s *registryShard) AllOrderedPriorityLevels() []int {
	return s.orderedPriorityLevels
}

// Stats returns a snapshot of the aggregated statistics for this specific shard.
//
// Note on Concurrency: Statistics are aggregated using high-performance, lock-free atomic updates.
// The returned stats represent a near-consistent snapshot. We acquire a Read Lock to ensure that
// configuration metadata (like names and capacity limits) remains stable during the iteration.
func (s *registryShard) Stats() contracts.ShardStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	stats := contracts.ShardStats{
		ID:                   s.id,
		IsActive:             s.IsActive(),
		TotalCapacityBytes:   s.config.MaxBytes,
		TotalByteSize:        uint64(s.totalByteSize.Load()),
		TotalLen:             uint64(s.totalLen.Load()),
		PerPriorityBandStats: make(map[int]contracts.PriorityBandStats),
	}

	s.priorityBands.Range(func(key, value any) bool {
		priority := key.(int)
		band := value.(*priorityBand)

		stats.PerPriorityBandStats[priority] = contracts.PriorityBandStats{
			Priority:      priority,
			PriorityName:  band.config.PriorityName,
			CapacityBytes: band.config.MaxBytes, // This is the partitioned capacity.
			ByteSize:      uint64(band.byteSize.Load()),
			Len:           uint64(band.len.Load()),
		}
		return true
	})
	return stats
}

//  --- Internal Administrative/Lifecycle Methods ---

// synchronizeFlow is the internal administrative method for creating a flow instance on this shard.
// It is an idempotent "create if not exists" operation.
func (s *registryShard) synchronizeFlow(
	spec types.FlowSpecification,
	policy framework.IntraFlowDispatchPolicy,
	q framework.SafeQueue,
) {
	s.mu.Lock()
	defer s.mu.Unlock()

	key := spec.Key

	val, _ := s.priorityBands.Load(key.Priority)
	band := val.(*priorityBand)
	if _, ok := band.queues[key.ID]; ok {
		return
	}

	s.logger.V(logging.TRACE).Info("Creating new queue for flow instance.",
		"flowKey", key, "queueType", q.Name())

	// Create a closure that captures the shard's `isDraining` atomic field.
	// This provides the queue with a way to check the shard's status without creating a tight coupling or circular
	// dependency.
	isDrainingFunc := func() bool {
		return s.isDraining.Load()
	}

	mq := newManagedQueue(q, policy, spec.Key, s.logger, s.propagateStatsDelta, isDrainingFunc)
	band.queues[key.ID] = mq
}

// deleteFlow removes a queue instance from the shard.
func (s *registryShard) deleteFlow(key types.FlowKey) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.logger.Info("Deleting queue instance.", "flowKey", key)
	if val, ok := s.priorityBands.Load(key.Priority); ok {
		band := val.(*priorityBand)
		delete(band.queues, key.ID)
	}
}

// markAsDraining transitions the shard to a Draining state. This method is lock-free.
func (s *registryShard) markAsDraining() {
	s.isDraining.Store(true)
	s.logger.V(logging.DEBUG).Info("Shard marked as Draining")
}

// updateConfig atomically replaces the shard's configuration. This is used during scaling events to re-partition
// capacity allocations.
func (s *registryShard) updateConfig(newConfig *ShardConfig) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.config = newConfig
	s.priorityBands.Range(func(key, value any) bool {
		priority := key.(int)
		band := value.(*priorityBand)
		newBandConfig := newConfig.PriorityBands[priority]
		band.config = *newBandConfig
		return true
	})
	s.logger.Info("Shard configuration updated")
}

// --- Internal Callback ---

// propagateStatsDelta is the single point of entry for all statistics changes within the shard.
// It atomically updates the relevant band's stats, the shard's total stats, and propagates the delta to the parent
// registry.
func (s *registryShard) propagateStatsDelta(priority int, lenDelta, byteSizeDelta int64) {
	val, _ := s.priorityBands.Load(priority)
	band := val.(*priorityBand)
	band.len.Add(lenDelta)
	band.byteSize.Add(byteSizeDelta)
	s.totalLen.Add(lenDelta)
	s.totalByteSize.Add(byteSizeDelta)

	// Propagate the delta up to the parent registry. This propagation is lock-free and eventually consistent.
	s.onStatsDelta(priority, lenDelta, byteSizeDelta)
}

// --- `priorityBandAccessor` ---

// priorityBandAccessor implements `framework.PriorityBandAccessor`. It provides a read-only, concurrent-safe view of a
// single priority band within a shard.
type priorityBandAccessor struct {
	shard *registryShard
	band  *priorityBand
}

var _ framework.PriorityBandAccessor = &priorityBandAccessor{}

// Priority returns the numerical priority level of this band.
func (a *priorityBandAccessor) Priority() int {
	a.shard.mu.RLock()
	defer a.shard.mu.RUnlock()
	return a.band.config.Priority
}

// PriorityName returns the human-readable name of this priority band.
func (a *priorityBandAccessor) PriorityName() string {
	a.shard.mu.RLock()
	defer a.shard.mu.RUnlock()
	return a.band.config.PriorityName
}

// FlowKeys returns a slice of all flow keys within this priority band.
//
// To minimize lock contention, this implementation first snapshots the flow IDs under a read lock and then constructs
// the final slice of `types.FlowKey` structs outside of the lock.
func (a *priorityBandAccessor) FlowKeys() []types.FlowKey {
	a.shard.mu.RLock()
	ids := make([]string, 0, len(a.band.queues))
	for id := range a.band.queues {
		ids = append(ids, id)
	}
	a.shard.mu.RUnlock()

	flowKeys := make([]types.FlowKey, len(ids))
	for i, id := range ids {
		flowKeys[i] = types.FlowKey{ID: id, Priority: a.Priority()}
	}
	return flowKeys
}

// Queue returns a `framework.FlowQueueAccessor` for the specified logical `ID` within this priority band.
func (a *priorityBandAccessor) Queue(id string) framework.FlowQueueAccessor {
	a.shard.mu.RLock()
	defer a.shard.mu.RUnlock()

	mq, ok := a.band.queues[id]
	if !ok {
		return nil
	}
	return mq.FlowQueueAccessor()
}

// IterateQueues executes the given `callback` for each `framework.FlowQueueAccessor` in this priority band.
//
// To minimize lock contention, this implementation snapshots the queue accessors under a read lock and then executes
// the callback on the snapshot, outside of the lock. This ensures that a potentially slow policy (the callback) does
// not block other operations on the shard.
func (a *priorityBandAccessor) IterateQueues(callback func(queue framework.FlowQueueAccessor) bool) {
	a.shard.mu.RLock()
	accessors := make([]framework.FlowQueueAccessor, 0, len(a.band.queues))
	for _, mq := range a.band.queues {
		accessors = append(accessors, mq.FlowQueueAccessor())
	}
	a.shard.mu.RUnlock()

	for _, accessor := range accessors {
		if !callback(accessor) {
			return
		}
	}
}
