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
	inter "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/interflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// shardCallbacks groups the callback functions that a `registryShard` uses to communicate with its parent registry.
type shardCallbacks struct {
	propagateStatsDelta propagateStatsDeltaFunc
	signalQueueState    func(shardID string, spec types.FlowSpecification, signal queueStateSignal)
	signalShardState    signalShardStateFunc
}

// registryShard implements the `contracts.RegistryShard` interface. It is the data plane counterpart to the
// `FlowRegistry`'s control plane.
//
// # Role: The Data Plane Slice
//
// It represents a single, concurrent-safe slice of the registry's total state, holding the partitioned configuration
// and the actual queue instances (`managedQueue`) for its assigned partition. It provides a read-optimized, operational
// view for a single `controller.FlowController` worker.
//
// The registryShard is deliberately kept simple regarding coordination logic; it relies on the parent `FlowRegistry`
// (the control plane) to orchestrate complex lifecycle events like registration and garbage collection.
//
// # Concurrency: `RWMutex` and Atomics
//
// The `registryShard` uses a hybrid approach to manage concurrency, balancing read performance with write safety:
//
//   - `sync.RWMutex` (mu): Protects the shard's internal maps (`priorityBands`, `activeFlows`) during administrative
//     operations (flow synchronization). This ensures that the set of available queues appears atomic to the
//     `controller.FlowController` workers. All read-oriented methods acquire the read lock, allowing parallel access.
//
//   - Atomics (Stats): All statistics (`totalByteSize`, `totalLen`) are implemented using atomics. This allows for
//     lock-free, high-performance updates on the data path hot path.
//
//   - Atomic Lifecycle (Status): The shard's lifecycle state (Active, Draining, Drained) is managed via an atomic
//     `status` enum. This ensures robust, atomic state transitions and provides the mechanism for exactly-once signaling
//     when the shard finishes draining (the transition from Draining to Drained).
type registryShard struct {
	id     string
	logger logr.Logger
	config *Config // Holds the *partitioned* config for this shard.

	// status tracks the lifecycle state of the shard (Active, Draining, Drained).
	// It is stored as an `int32` for atomic operations.
	status atomic.Int32 // `componentStatus`

	// parentCallbacks provides the communication channels back to the parent registry.
	parentCallbacks shardCallbacks

	// mu protects the shard's internal maps (`priorityBands` and `activeFlows`).
	mu sync.RWMutex

	// priorityBands is the primary lookup table for all managed queues on this shard, organized by priority, then by
	// flow ID. This map contains BOTH active and draining queues.
	priorityBands map[uint]*priorityBand

	// activeFlows is a flattened map for O(1) access to the SINGLE active queue for a given logical flow ID.
	// This is the critical lookup for the `Enqueue` path. If a flow ID is not in this map, it has no active queue on this
	// shard.
	activeFlows map[string]*managedQueue

	// orderedPriorityLevels is a cached, sorted list of `priority` levels.
	// It is populated at initialization to avoid repeated map key iteration and sorting during the dispatch loop,
	// ensuring a deterministic, ordered traversal from highest to lowest priority.
	orderedPriorityLevels []uint

	// Shard-level statistics, which are updated atomically to ensure they are safe for concurrent access without locks.
	totalByteSize atomic.Uint64
	totalLen      atomic.Uint64
}

// priorityBand holds all the `managedQueues` and configuration for a single priority level within a shard.
// It acts as a logical grouping for all state related to a specific priority.
type priorityBand struct {
	// config holds the partitioned config for this specific band.
	config PriorityBandConfig

	// queues holds all `managedQueue` instances within this band, keyed by `flowID`. This includes both active and
	// draining queues.
	queues map[string]*managedQueue

	// Band-level statistics, which are updated atomically.
	byteSize atomic.Uint64
	len      atomic.Uint64

	// Cached policy instance for this band, created at initialization.
	interFlowDispatchPolicy framework.InterFlowDispatchPolicy
}

var _ contracts.RegistryShard = &registryShard{}

// newShard creates a new `registryShard` instance from a partitioned configuration.
func newShard(
	id string,
	partitionedConfig *Config,
	logger logr.Logger,
	parentCallbacks shardCallbacks,
) (*registryShard, error) {
	shardLogger := logger.WithName("registry-shard").WithValues("shardID", id)
	s := &registryShard{
		id:              id,
		logger:          shardLogger,
		config:          partitionedConfig,
		parentCallbacks: parentCallbacks,
		priorityBands:   make(map[uint]*priorityBand, len(partitionedConfig.PriorityBands)),
		activeFlows:     make(map[string]*managedQueue),
	}
	// Initialize the shard in the Active state.
	s.status.Store(int32(componentStatusActive))

	for _, bandConfig := range partitionedConfig.PriorityBands {
		interPolicy, err := inter.NewPolicyFromName(bandConfig.InterFlowDispatchPolicy)
		if err != nil {
			return nil, fmt.Errorf("failed to create inter-flow policy %q for priority band %d: %w",
				bandConfig.InterFlowDispatchPolicy, bandConfig.Priority, err)
		}

		// The intra-flow policy is instantiated on-demand by the `FlowRegistry`, not cached here.
		s.priorityBands[bandConfig.Priority] = &priorityBand{
			config:                  bandConfig,
			queues:                  make(map[string]*managedQueue),
			interFlowDispatchPolicy: interPolicy,
		}
		s.orderedPriorityLevels = append(s.orderedPriorityLevels, bandConfig.Priority)
	}

	// Sort priorities in ascending order (0 is highest priority).
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

// ActiveManagedQueue returns the currently active `contracts.ManagedQueue` for a given flow.
func (s *registryShard) ActiveManagedQueue(flowID string) (contracts.ManagedQueue, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	mq, ok := s.activeFlows[flowID]
	if !ok {
		// We do not check the shard's status here. Even if the shard is draining, the specific queue might still be active
		// if the flow configuration hasn't changed. The queue itself will reject the `Add` if it is also draining.
		return nil, fmt.Errorf("failed to get active queue for flow %q: %w", flowID, contracts.ErrFlowInstanceNotFound)
	}
	return mq, nil
}

// ManagedQueue retrieves a specific (potentially draining or drained) `contracts.ManagedQueue` instance from this
// shard.
func (s *registryShard) ManagedQueue(flowID string, priority uint) (contracts.ManagedQueue, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	band, ok := s.priorityBands[priority]
	if !ok {
		return nil, fmt.Errorf("failed to get managed queue for flow %q: %w", flowID, contracts.ErrPriorityBandNotFound)
	}
	mq, ok := band.queues[flowID]
	if !ok {
		return nil, fmt.Errorf("failed to get managed queue for flow %q at priority %d: %w",
			flowID, priority, contracts.ErrFlowInstanceNotFound)
	}
	return mq, nil
}

// IntraFlowDispatchPolicy retrieves a flow's configured `framework.IntraFlowDispatchPolicy`.
func (s *registryShard) IntraFlowDispatchPolicy(
	flowID string,
	priority uint,
) (framework.IntraFlowDispatchPolicy, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	band, ok := s.priorityBands[priority]
	if !ok {
		return nil, fmt.Errorf("failed to get intra-flow policy for flow %q: %w", flowID, contracts.ErrPriorityBandNotFound)
	}
	mq, ok := band.queues[flowID]
	if !ok {
		return nil, fmt.Errorf("failed to get intra-flow policy for flow %q at priority %d: %w",
			flowID, priority, contracts.ErrFlowInstanceNotFound)
	}
	// The policy is stored on the managed queue.
	return mq.dispatchPolicy, nil
}

// InterFlowDispatchPolicy retrieves a priority band's configured `framework.InterFlowDispatchPolicy`.
func (s *registryShard) InterFlowDispatchPolicy(priority uint) (framework.InterFlowDispatchPolicy, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

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
//
// This list is cached at initialization to provide a stable, ordered view for the `controller.FlowController`'s
// dispatch loop, avoiding repeated map key iteration and sorting on the hot path.
func (s *registryShard) AllOrderedPriorityLevels() []uint {
	// This is cached and read-only, so no lock is needed.
	return s.orderedPriorityLevels
}

// Stats returns a snapshot of the statistics for this specific shard.
func (s *registryShard) Stats() contracts.ShardStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	stats := contracts.ShardStats{
		TotalCapacityBytes:   s.config.MaxBytes,
		TotalByteSize:        s.totalByteSize.Load(),
		TotalLen:             s.totalLen.Load(),
		PerPriorityBandStats: make(map[uint]contracts.PriorityBandStats, len(s.priorityBands)),
	}

	for priority, band := range s.priorityBands {
		stats.PerPriorityBandStats[priority] = contracts.PriorityBandStats{
			Priority:      priority,
			PriorityName:  band.config.PriorityName,
			CapacityBytes: band.config.MaxBytes, // This is the partitioned capacity.
			ByteSize:      band.byteSize.Load(),
			Len:           band.len.Load(),
		}
	}
	return stats
}

//  --- Internal Administrative/Lifecycle Methods (called by `FlowRegistry`) ---

// synchronizeFlow is the internal administrative method for creating or updating a flow on this shard.
// The parent `FlowRegistry` handles validation and policy instantiation. This method implements the state machine for
// the `managedQueue` lifecycle and instantiates the underlying `framework.SafeQueue` only when necessary.
//
// This function atomically handles the following transitions under the shard's write lock:
//  1. Creation: If no queue exists, a new Active one is created.
//  2. No-Op Update: If an Active queue already exists at the target priority, nothing changes (no allocation).
//  3. Priority Change: The old Active queue is transitioned to Draining, and a new Active queue is created.
//  4. Reactivation: If a Draining/Drained queue exists at the target priority, it is transitioned back to Active (no
//     allocation).
func (s *registryShard) synchronizeFlow(spec types.FlowSpecification, policy framework.IntraFlowDispatchPolicy, q framework.SafeQueue) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Check if an active queue already exists for this flow.
	if existingActive, ok := s.activeFlows[spec.ID]; ok {
		// If it's at the same priority, it's a no-op.
		if existingActive.flowSpec.Priority == spec.Priority {
			return
		}
		// It's a priority change. Mark the old one as draining.
		s.logger.V(logging.TRACE).Info("Flow priority changed, marking old queue as draining.", "flowID", spec.ID,
			"oldPriority", existingActive.flowSpec.Priority, "newPriority", spec.Priority)
		existingActive.markAsDraining()
		delete(s.activeFlows, spec.ID)
	}

	// Now, either no active queue existed, or we just marked the old one as draining.
	// We need to establish an active queue at the new priority.
	targetBand := s.priorityBands[spec.Priority]

	// Case 1: A queue (Draining or Drained) exists at the target priority. Reactivate it.
	if existingQueue, ok := targetBand.queues[spec.ID]; ok {
		s.logger.V(logging.TRACE).Info("Found existing queue at target priority, reactivating.", "flowID", spec.ID, "priority", spec.Priority)
		existingQueue.reactivate()
		s.activeFlows[spec.ID] = existingQueue
		return
	}

	// Case 2: No queue exists at the target priority. Create a new one.
	s.logger.V(logging.TRACE).Info("Creating new active queue for flow.", "flowID", spec.ID, "priority", spec.Priority,
		"queueType", q.Name())

	callbacks := managedQueueCallbacks{
		propagateStatsDelta: s.propagateStatsDelta,
		signalQueueState: func(spec types.FlowSpecification, signal queueStateSignal) {
			s.parentCallbacks.signalQueueState(s.id, spec, signal)
		},
	}
	// The provided `q` and `policy` are guaranteed to be non-nil by the caller (`FlowRegistry`).
	mq := newManagedQueue(q, policy, spec, s.logger, callbacks)
	targetBand.queues[spec.ID] = mq
	s.activeFlows[spec.ID] = mq
}

// garbageCollect removes a queue instance from the shard.
// This must be called under the shard's write lock.
func (s *registryShard) garbageCollect(flowID string, priority uint) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.logger.Info("Garbage collecting queue instance.", "flowID", flowID, "priority", priority)

	// Remove from the priority band's map, which contains all instances (Active/Draining/Drained).
	if band, ok := s.priorityBands[priority]; ok {
		delete(band.queues, flowID)
	}

	// If this queue was the active one, also remove it from the activeFlows map.
	// Note: A flow might be GC'd entirely (e.g., due to inactivity), in which case its active queue is removed here.
	if activeQueue, ok := s.activeFlows[flowID]; ok {
		if activeQueue.flowSpec.Priority == priority {
			delete(s.activeFlows, flowID)
		}
	}
}

// markAsDraining transitions the shard to a draining state. It will no longer be considered active for new work by the
// controller. This method is lock-free, relying on atomics for safe state transitions.
func (s *registryShard) markAsDraining() {
	// Attempt to transition from Active to Draining atomically.
	if s.status.CompareAndSwap(int32(componentStatusActive), int32(componentStatusDraining)) {
		s.logger.Info("Shard marked as draining")

		// CRITICAL: Mark all constituent queues as draining as well.
		// To prevent deadlocks, we collect the queues under a read lock, release it, and then perform the marking. This
		// ensures we are not holding a lock when we invoke the callbacks inside `mq.markAsDraining()`.
		var queuesToMark []*managedQueue
		s.mu.RLock()
		for _, band := range s.priorityBands {
			for _, mq := range band.queues {
				queuesToMark = append(queuesToMark, mq)
			}
		}
		s.mu.RUnlock()

		for _, mq := range queuesToMark {
			// This ensures that even if a specific flow hasn't changed configuration, its queue on this specific shard stops
			// accepting new traffic.
			mq.markAsDraining()
		}
	}

	// CRITICAL: Check if the shard is *already* empty at the moment it's marked as draining (or if it was already
	// draining and empty). If so, we must immediately attempt the transition to Drained to ensure timely GC. This handles
	// the race where the shard becomes empty just before or during being marked draining.
	if s.totalLen.Load() == 0 {
		// Attempt to transition from Draining to Drained atomically.
		if s.status.CompareAndSwap(int32(componentStatusDraining), int32(componentStatusDrained)) {
			s.parentCallbacks.signalShardState(s, shardStateSignalBecameDrained)
		}
	}
}

// updateConfig atomically replaces the shard's configuration. This is used during shard scaling events to re-partition
// capacity allocations.
func (s *registryShard) updateConfig(newConfig *Config) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.config = newConfig
	// Update the partitioned config for each priority band as well.
	for priority, band := range s.priorityBands {
		newBandConfig, err := newConfig.getBandConfig(priority)
		if err != nil {
				// An invariant was violated: a priority exists in the shard but not in the new config.
				// This should be impossible if the registry's logic is correct.
				panic(fmt.Errorf("invariant violation: priority band (%d) missing in new configuration during update: %w",
					priority, err))
		}
		band.config = *newBandConfig
	}
	s.logger.Info("Shard configuration updated")
}

// --- Internal Callback Methods ---

// propagateStatsDelta is the single point of entry for all statistics changes within the shard. It updates the relevant
// band's stats, the shard's total stats, and propagates the delta to the parent registry. It also handles the shard's
// lifecycle signaling.
func (s *registryShard) propagateStatsDelta(priority uint, lenDelta, byteSizeDelta int64) {
	// This function uses two's complement arithmetic to atomically add or subtract from the unsigned counters.
	// Casting a negative `int64` to `uint64` results in its two's complement representation, which, when added, is
	// equivalent to subtraction. This is a standard and efficient pattern for atomic updates on unsigned integers.
	newTotalLen := s.totalLen.Add(uint64(lenDelta))
	s.totalByteSize.Add(uint64(byteSizeDelta))

	if band, ok := s.priorityBands[priority]; ok {
		band.len.Add(uint64(lenDelta))
		band.byteSize.Add(uint64(byteSizeDelta))
	} else {
		// This should be impossible if the `managedQueue` calling this is correctly registered.
		panic(fmt.Sprintf("invariant violation: received stats propagation for unknown priority band (%d)", priority))
	}

	s.logger.V(logging.TRACE).Info("Propagated shard stats delta", "priority", priority,
		"lenDelta", lenDelta, "byteSizeDelta", byteSizeDelta)

	s.parentCallbacks.propagateStatsDelta(priority, lenDelta, byteSizeDelta)

	// --- State Machine Signaling Logic ---

	// Check for Draining -> Drained transition (Exactly-Once).
	// This must happen if the total length just hit zero.
	if newTotalLen == 0 {
		// Attempt to transition from Draining to Drained atomically.
		// This acts as the exactly-once latch. If it succeeds, we are the single goroutine responsible for signaling.
		if s.status.CompareAndSwap(int32(componentStatusDraining), int32(componentStatusDrained)) {
			s.parentCallbacks.signalShardState(s, shardStateSignalBecameDrained)
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

// FlowIDs returns a slice of all flow IDs within this priority band.
func (a *priorityBandAccessor) FlowIDs() []string {
	a.shard.mu.RLock()
	defer a.shard.mu.RUnlock()

	flowIDs := make([]string, 0, len(a.band.queues))
	for id := range a.band.queues {
		flowIDs = append(flowIDs, id)
	}
	return flowIDs
}

// Queue returns a `framework.FlowQueueAccessor` for the specified `flowID` within this priority band.
func (a *priorityBandAccessor) Queue(flowID string) framework.FlowQueueAccessor {
	a.shard.mu.RLock()
	defer a.shard.mu.RUnlock()

	mq, ok := a.band.queues[flowID]
	if !ok {
		return nil
	}
	return mq.FlowQueueAccessor()
}

// IterateQueues executes the given `callback` for each `framework.FlowQueueAccessor` in this priority band.
// The callback is executed under the shard's read lock, so it should be efficient and non-blocking.
// If the callback returns false, iteration stops.
func (a *priorityBandAccessor) IterateQueues(callback func(queue framework.FlowQueueAccessor) bool) {
	a.shard.mu.RLock()
	defer a.shard.mu.RUnlock()

	for _, mq := range a.band.queues {
		if !callback(mq.FlowQueueAccessor()) {
			return
		}
	}
}
