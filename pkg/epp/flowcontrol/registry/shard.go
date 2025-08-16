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
	"slices"
	"sync"
	"sync/atomic"

	"github.com/go-logr/logr"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// shardCallbacks groups the callback functions that a `registryShard` uses to communicate with its parent registry.
type shardCallbacks struct {
	propagateStatsDelta propagateStatsDeltaFunc
	signalQueueState    func(shardID string, key types.FlowKey, signal queueStateSignal)
	signalShardState    signalShardStateFunc
}

// priorityBand holds all the `managedQueues` and configuration for a single priority level within a shard.
type priorityBand struct {
	// config holds the partitioned config for this specific band within this shard.
	config ShardPriorityBandConfig

	// queues holds all `managedQueue` instances within this band, keyed by their logical `ID` string.
	// The priority is implicit from the parent `priorityBand`.
	queues map[string]*managedQueue

	// Band-level statistics. Updated atomically via lock-free propagation.
	byteSize atomic.Int64
	len      atomic.Int64

	// Cached policy instance for this band, created at initialization.
	interFlowDispatchPolicy framework.InterFlowDispatchPolicy
}

// registryShard implements the `contracts.RegistryShard` interface.
//
// # Role: The Data Plane Slice
//
// It represents a single, concurrent-safe slice of the registry's total state. It provides a read-optimized view for a
// `controller.FlowController` worker.
//
// # Concurrency: `RWMutex` and Atomics
//
// The `registryShard` balances read performance with write safety:
//
//   - `sync.RWMutex` (mu): Protects the shard's internal maps (`priorityBands`) during administrative operations.
//
//   - Atomics (Stats): Aggregated statistics (`totalByteSize`, `totalLen`) use atomics for lock-free updates during
//     delta propagation.
//
//   - Atomic Lifecycle (Status): The lifecycle state is managed via an atomic `status` enum.
type registryShard struct {
	id     string
	logger logr.Logger

	// config holds the partitioned configuration for this shard, derived from the `FlowRegistry`'s global `Config`.
	// It contains only the settings and capacity limits relevant to this specific shard.
	config *ShardConfig

	// status tracks the lifecycle state of the shard (Active, Draining, Drained).
	// It is stored as an `int32` for atomic operations.
	status atomic.Int32 // `componentStatus`

	// parentCallbacks provides the communication channels back to the parent registry.
	parentCallbacks shardCallbacks

	// mu protects the shard's internal maps (`priorityBands`).
	mu sync.RWMutex

	// priorityBands is the primary lookup table for all managed queues on this shard, organized by `priority`.
	priorityBands map[uint]*priorityBand

	// orderedPriorityLevels is a cached, sorted list of `priority` levels.
	// It is populated at initialization to avoid repeated map key iteration and sorting during the dispatch loop,
	// ensuring a deterministic, ordered traversal from highest to lowest priority.
	orderedPriorityLevels []uint

	// Shard-level statistics. Updated atomically via lock-free propagation.
	totalByteSize atomic.Int64
	totalLen      atomic.Int64
}

var _ contracts.RegistryShard = &registryShard{}

// newShard creates a new `registryShard` instance from a partitioned configuration.
func newShard(
	id string,
	config *ShardConfig,
	logger logr.Logger,
	parentCallbacks shardCallbacks,
	interFlowFactory interFlowDispatchPolicyFactory,
) (*registryShard, error) {
	shardLogger := logger.WithName("registry-shard").WithValues("shardID", id)
	s := &registryShard{
		id:              id,
		logger:          shardLogger,
		config:          config,
		parentCallbacks: parentCallbacks,
		priorityBands:   make(map[uint]*priorityBand, len(config.PriorityBands)),
	}
	s.status.Store(int32(componentStatusActive))

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

	slices.Sort(s.orderedPriorityLevels)
	s.logger.V(logging.DEFAULT).Info("Registry shard initialized successfully",
		"priorityBandCount", len(s.priorityBands), "orderedPriorities", s.orderedPriorityLevels)
	return s, nil
}

// ID returns the unique identifier for this shard.
func (s *registryShard) ID() string { return s.id }

// IsActive returns true if the shard is active and accepting new requests.
// This is used by the `controller.FlowController` to determine if it should use this shard for new enqueue operations.
func (s *registryShard) IsActive() bool {
	return componentStatus(s.status.Load()) == componentStatusActive
}

// ManagedQueue retrieves a specific `contracts.ManagedQueue` instance from this shard.
func (s *registryShard) ManagedQueue(key types.FlowKey) (contracts.ManagedQueue, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.managedQueueLocked(key)
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
	// The policy is stored on the `managedQueue`.
	return mq.dispatchPolicy, nil
}

// InterFlowDispatchPolicy retrieves a priority band's configured `framework.InterFlowDispatchPolicy`.
// This read is lock-free as the policy instance is immutable after the shard is initialized.
func (s *registryShard) InterFlowDispatchPolicy(priority uint) (framework.InterFlowDispatchPolicy, error) {
	band, ok := s.priorityBands[priority]
	if !ok {
		return nil, fmt.Errorf("failed to get inter-flow policy for priority %d: %w",
			priority, contracts.ErrPriorityBandNotFound)
	}
	return band.interFlowDispatchPolicy, nil
}

// PriorityBandAccessor retrieves a read-only view for a given priority level.
func (s *registryShard) PriorityBandAccessor(priority uint) (framework.PriorityBandAccessor, error) {
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
// The slice is sorted from highest to lowest priority (ascending numerical order).
func (s *registryShard) AllOrderedPriorityLevels() []uint {
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
		TotalCapacityBytes:   s.config.MaxBytes,
		TotalByteSize:        uint64(s.totalByteSize.Load()),
		TotalLen:             uint64(s.totalLen.Load()),
		PerPriorityBandStats: make(map[uint]contracts.PriorityBandStats, len(s.priorityBands)),
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
// Since a flow instance (identified by its immutable `FlowKey`) cannot be updated yet, this function is a simple
// "create if not exists" operation. It is idempotent.
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

	callbacks := managedQueueCallbacks{
		propagateStatsDelta: s.propagateStatsDelta,
		signalQueueState: func(key types.FlowKey, signal queueStateSignal) {
			s.parentCallbacks.signalQueueState(s.id, key, signal)
		},
	}
	mq := newManagedQueue(q, policy, spec.Key, s.logger, callbacks)
	band.queues[key.ID] = mq
}

// garbageCollectLocked removes a queue instance from the shard.
// This must be called under the shard's write lock.
func (s *registryShard) garbageCollectLocked(key types.FlowKey) {
	s.logger.Info("Garbage collecting queue instance.", "flowKey", key, "flowID", key.ID, "priority", key.Priority)
	if band, ok := s.priorityBands[key.Priority]; ok {
		delete(band.queues, key.ID)
	}
}

// markAsDraining transitions the shard to a Draining state. This method is lock-free.
func (s *registryShard) markAsDraining() {
	// Attempt to transition from Active to Draining atomically.
	if s.status.CompareAndSwap(int32(componentStatusActive), int32(componentStatusDraining)) {
		s.logger.V(logging.DEBUG).Info("Shard status changed",
			"from", componentStatusActive,
			"to", componentStatusDraining,
		)
	}

	// Check if the shard is *already* empty when marked as draining. If so, immediately attempt the transition to
	// Drained to ensure timely GC. This handles the race where the shard becomes empty just before or during being
	// marked Draining.
	if s.totalLen.Load() == 0 {
		// Attempt to transition from Draining to Drained atomically.
		if s.status.CompareAndSwap(int32(componentStatusDraining), int32(componentStatusDrained)) {
			s.parentCallbacks.signalShardState(s.id, shardStateSignalBecameDrained)
		}
	}
}

// managedQueueLocked retrieves a specific `contracts.ManagedQueue` instance from this shard.
// This must be called under the shard's read lock.
func (s *registryShard) managedQueueLocked(key types.FlowKey) (*managedQueue, error) {
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
			// This should be impossible if the registry's logic is correct.
			panic(fmt.Errorf("invariant violation: priority band (%d) missing in new shard configuration during update: %w",
				priority, err))
		}
		band.config = *newBandConfig
	}
	s.logger.Info("Shard configuration updated")
}

// --- Internal Callback Methods ---

// propagateStatsDelta is the single point of entry for all statistics changes within the shard.
// It updates the relevant band's stats, the shard's total stats, and handles the shard's lifecycle signaling before
// propagating the delta to the parent registry.
// It uses atomic operations to maintain high performance under concurrent updates from multiple shards.
// As a result, its counters are eventually consistent and may be transiently inaccurate during high-contention races.
func (s *registryShard) propagateStatsDelta(priority uint, lenDelta, byteSizeDelta int64) {
	band, ok := s.priorityBands[priority]
	if !ok {
		// This should be impossible if the `managedQueue` calling this is correctly registered.
		panic(fmt.Sprintf("invariant violation on shard %s: received stats propagation for unknown priority band (%d)",
			s.id, priority))
	}

	band.len.Add(lenDelta)
	band.byteSize.Add(byteSizeDelta)
	newTotalLen := s.totalLen.Add(lenDelta)
	s.totalByteSize.Add(byteSizeDelta)

	// Following the strict bottom-up signaling pattern, we evaluate and signal our own state change *before* propagating
	// the statistics to the parent registry.
	s.evaluateDrainingState(newTotalLen)
	s.parentCallbacks.propagateStatsDelta(priority, lenDelta, byteSizeDelta)
}

// evaluateDrainingState checks if the shard has transitioned to the Drained state and signals the parent.
func (s *registryShard) evaluateDrainingState(currentLen int64) {
	if currentLen == 0 {
		// Attempt transition from Draining to Drained atomically.
		// This acts as the exactly-once latch. If it succeeds, this goroutine is solely responsible for signaling.
		if s.status.CompareAndSwap(int32(componentStatusDraining), int32(componentStatusDrained)) {
			s.parentCallbacks.signalShardState(s.id, shardStateSignalBecameDrained)
		}
	}
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
func (a *priorityBandAccessor) Priority() uint { return a.band.config.Priority }

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
