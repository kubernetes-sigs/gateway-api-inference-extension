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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// priorityBand holds all `managedQueues` and configuration for a single priority level within a shard.
type priorityBand struct {
	// --- Immutable (set at construction) ---
	config                  ShardPriorityBandConfig
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
// # Concurrency Model: `RWMutex` for Topology, Atomics for Stats
//
// The `registryShard` balances read performance with write safety using a hybrid model:
//
//   - `mu (sync.RWMutex)`: Protects the shard's internal topology (the maps of priority bands and queues) during
//     administrative operations like flow registration, garbage collection, and configuration updates.
//     Read locks are used on the hot path to look up queues, while write locks are used for infrequent structural
//     changes.
//   - Atomics: Aggregated statistics (`totalByteSize`, `totalLen`, etc.) and the `isDraining` flag use atomic
//     operations, allowing for high-frequency, lock-free updates and reads of the shard's status and load, which is
//     critical for the performance of the data path and statistics propagation.
type registryShard struct {
	// --- Immutable Identity & Dependencies (set at construction) ---
	id     string
	logger logr.Logger

	// onStatsDelta is the callback used to propagate statistics changes up to the parent registry.
	onStatsDelta propagateStatsDeltaFunc
	// orderedPriorityLevels is a cached, sorted list of priority levels.
	orderedPriorityLevels []int

	// --- State Protected by `mu` ---

	// mu protects the shard's internal topology (`priorityBands`) and `config`.
	// TODO: This is a priority inversion issue. Administrative operations (e.g., GC) for a low-priority flow block all
	// data path operations for high priority flows on this shard. We should replace `s.mu` with granular per-band locks.
	// This is safe since the priority band map structure is immutable at initialization.
	mu sync.RWMutex
	// config holds the partitioned configuration for this shard, derived from the `FlowRegistry`'s global `Config`.
	config *ShardConfig
	// priorityBands is the primary lookup table for all managed queues on this shard.
	priorityBands map[int]*priorityBand

	// --- Concurrent-Safe State (Atomics) ---

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
	interFlowFactory interFlowDispatchPolicyFactory,
) (*registryShard, error) {
	shardLogger := logger.WithName("registry-shard").WithValues("shardID", id)
	s := &registryShard{
		id:            id,
		logger:        shardLogger,
		config:        config,
		onStatsDelta:  onStatsDelta,
		priorityBands: make(map[int]*priorityBand, len(config.PriorityBands)),
	}

	for _, bandConfig := range config.PriorityBands {
		interPolicy, err := interFlowFactory(bandConfig.InterFlowDispatchPolicy)
		if err != nil {
			return nil, fmt.Errorf("failed to create inter-flow policy %q for priority band %d: %w",
				bandConfig.InterFlowDispatchPolicy, bandConfig.Priority, err)
		}

		s.priorityBands[bandConfig.Priority] = &priorityBand{
			config:                  bandConfig,
			queues:                  make(map[string]*managedQueue),
			interFlowDispatchPolicy: interPolicy,
		}
		s.orderedPriorityLevels = append(s.orderedPriorityLevels, bandConfig.Priority)
	}
	sort.Slice(s.orderedPriorityLevels, func(i, j int) bool {
		return s.orderedPriorityLevels[i] > s.orderedPriorityLevels[j]
	})
	s.logger.V(logging.DEFAULT).Info("Registry shard initialized successfully",
		"priorityBandCount", len(s.priorityBands), "orderedPriorities", s.orderedPriorityLevels)
	return s, nil
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

	band, ok := s.priorityBands[key.Priority]
	if !ok {
		return nil, fmt.Errorf("failed to get managed queue for flow %q: %w", key, contracts.ErrPriorityBandNotFound)
	}
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

	band, ok := s.priorityBands[key.Priority]
	if !ok {
		return nil, fmt.Errorf("failed to get intra-flow policy for flow %q: %w", key, contracts.ErrPriorityBandNotFound)
	}
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
	// This read is safe because the `priorityBands` map structure is immutable after initialization.
	band, ok := s.priorityBands[priority]
	if !ok {
		return nil, fmt.Errorf("failed to get inter-flow policy for priority %d: %w",
			priority, contracts.ErrPriorityBandNotFound)
	}
	return band.interFlowDispatchPolicy, nil
}

// PriorityBandAccessor retrieves a read-only view for a given priority level.
func (s *registryShard) PriorityBandAccessor(priority int) (framework.PriorityBandAccessor, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	band, ok := s.priorityBands[priority]
	if !ok {
		return nil, fmt.Errorf("failed to get priority band accessor for priority %d: %w",
			priority, contracts.ErrPriorityBandNotFound)
	}
	return &priorityBandAccessor{shard: s, band: band}, nil
}

// AllOrderedPriorityLevels returns a cached, sorted slice of all configured priority levels for this shard.
// This is a lock-free read.
func (s *registryShard) AllOrderedPriorityLevels() []int {
	return s.orderedPriorityLevels
}

// Stats returns a snapshot of the aggregated statistics for this specific shard.
//
// Note on Concurrency and Consistency: Statistics are aggregated using high-performance, lock-free atomic updates
// propagated from the underlying `managedQueue` instances. The returned stats represent a near-consistent snapshot of
// the shard's state. It is not perfectly atomic because the various counters are loaded independently without a global
// lock.
func (s *registryShard) Stats() contracts.ShardStats {
	// Acquire `RLock` only to ensure the config/map structure isn't changing during iteration.
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Casts from `int64` to `uint64` are safe because the non-negative invariant is strictly enforced at the
	// `managedQueue` level.
	stats := contracts.ShardStats{
		ID:                   s.id,
		IsActive:             s.IsActive(),
		TotalCapacityBytes:   s.config.MaxBytes,
		TotalByteSize:        uint64(s.totalByteSize.Load()),
		TotalLen:             uint64(s.totalLen.Load()),
		PerPriorityBandStats: make(map[int]contracts.PriorityBandStats, len(s.priorityBands)),
	}

	for priority, band := range s.priorityBands {
		stats.PerPriorityBandStats[priority] = contracts.PriorityBandStats{
			Priority:      priority,
			PriorityName:  band.config.PriorityName,
			CapacityBytes: band.config.MaxBytes, // This is the partitioned capacity.
			ByteSize:      uint64(band.byteSize.Load()),
			Len:           uint64(band.len.Load()),
		}
	}
	return stats
}

//  --- Internal Administrative/Lifecycle Methods (called by `FlowRegistry`) ---

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

	// Find the correct priority band. A missing band is an invariant violation, as the `FlowRegistry` should have already
	// validated this.
	band, ok := s.priorityBands[key.Priority]
	if !ok {
		panic(fmt.Sprintf("invariant violation: attempt to synchronize flow on non-existent priority band %d", key.Priority))
	}

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

// deleteFlowLocked removes a queue instance from the shard.
func (s *registryShard) deleteFlow(key types.FlowKey) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.logger.Info("Deleting queue instance.", "flowKey", key, "flowID", key.ID, "priority", key.Priority)
	if band, ok := s.priorityBands[key.Priority]; ok {
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
	// Update the partitioned config for each priority band as well.
	for priority, band := range s.priorityBands {
		newBandConfig, err := newConfig.getBandConfig(priority)
		if err != nil {
			// An invariant was violated: a priority exists in the shard but not in the new config.
			panic(fmt.Errorf("invariant violation: priority band (%d) missing in new shard configuration during update: %w",
				priority, err))
		}
		band.config = *newBandConfig
	}
	s.logger.Info("Shard configuration updated")
}

// --- Internal Callback ---

// propagateStatsDelta is the single point of entry for all statistics changes within the shard.
// It atomically updates the relevant band's stats, the shard's total stats, and propagates the delta to the parent
// registry.
func (s *registryShard) propagateStatsDelta(priority int, lenDelta, byteSizeDelta int64) {
	// This read is safe because the `priorityBands` map structure is immutable after initialization.
	band, ok := s.priorityBands[priority]
	if !ok {
		// This should be impossible if the `managedQueue` calling this is correctly registered.
		panic(fmt.Sprintf("invariant violation on shard %s: received stats propagation for unknown priority band (%d)",
			s.id, priority))
	}

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
func (a *priorityBandAccessor) Priority() int { return a.band.config.Priority }

// PriorityName returns the human-readable name of this priority band.
func (a *priorityBandAccessor) PriorityName() string { return a.band.config.PriorityName }

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
