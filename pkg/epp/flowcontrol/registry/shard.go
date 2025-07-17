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
	intra "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// registryShard implements the `contracts.RegistryShard` interface. It represents a single, concurrent-safe slice of
// the `FlowRegistry`'s state, providing an operational view for a single `controller.FlowController` worker.
//
// # Responsibilities
//
//   - Holding the partitioned configuration and state (queues, policies) for its assigned shard.
//   - Providing read-only access to its state for the `controller.FlowController`'s dispatch loop.
//   - Aggregating statistics from its `managedQueue` instances.
//
// # Concurrency
//
// The `registryShard` uses a combination of an `RWMutex` and atomic operations to manage concurrency.
//   - The `mu` RWMutex protects the shard's internal maps (`priorityBands`, `activeFlows`) during administrative
//     operations like flow registration or updates. This ensures that the set of active or draining queues appears
//     atomic to a `controller.FlowController` worker. All read-oriented methods on the shard take a read lock.
//   - All statistics (`totalByteSize`, `totalLen`, etc.) are implemented as `atomic.Uint64` to allow for lock-free,
//     high-performance updates from many concurrent queue operations.
type registryShard struct {
	id           string
	logger       logr.Logger
	config       *Config // Holds the *partitioned* config for this shard.
	isActive     bool
	reconcileFun parentStatsReconciler

	// mu protects the shard's internal maps (`priorityBands` and `activeFlows`).
	mu sync.RWMutex

	// priorityBands is the primary lookup table for all managed queues on this shard, organized by `priority`, then by
	// `flowID`. This map contains BOTH active and draining queues.
	priorityBands map[uint]*priorityBand

	// activeFlows is a flattened map for O(1) access to the SINGLE active queue for a given logical flow ID.
	// This is the critical lookup for the `Enqueue` path. If a `flowID` is not in this map, it has no active queue on
	// this shard.
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
type priorityBand struct {
	// config holds the partitioned config for this specific band.
	config PriorityBandConfig

	// queues holds all `managedQueue` instances within this band, keyed by `flowID`. This includes both active and
	// draining queues.
	queues map[string]*managedQueue

	// Band-level statistics, which are updated atomically.
	byteSize atomic.Uint64
	len      atomic.Uint64

	// Cached policy instances for this band, created at initialization.
	interFlowDispatchPolicy        framework.InterFlowDispatchPolicy
	defaultIntraFlowDispatchPolicy framework.IntraFlowDispatchPolicy
}

// newShard creates a new `registryShard` instance from a partitioned configuration.
func newShard(
	id string,
	partitionedConfig *Config,
	logger logr.Logger,
	reconcileFunc parentStatsReconciler,
) (*registryShard, error) {
	shardLogger := logger.WithName("registry-shard").WithValues("shardID", id)
	s := &registryShard{
		id:            id,
		logger:        shardLogger,
		config:        partitionedConfig,
		isActive:      true,
		reconcileFun:  reconcileFunc,
		priorityBands: make(map[uint]*priorityBand, len(partitionedConfig.PriorityBands)),
		activeFlows:   make(map[string]*managedQueue),
	}

	for _, bandConfig := range partitionedConfig.PriorityBands {
		interPolicy, err := inter.NewPolicyFromName(bandConfig.InterFlowDispatchPolicy)
		if err != nil {
			return nil, fmt.Errorf("failed to create inter-flow policy %q for priority band %d: %w",
				bandConfig.InterFlowDispatchPolicy, bandConfig.Priority, err)
		}

		intraPolicy, err := intra.NewPolicyFromName(bandConfig.IntraFlowDispatchPolicy)
		if err != nil {
			return nil, fmt.Errorf("failed to create intra-flow policy %q for priority band %d: %w",
				bandConfig.IntraFlowDispatchPolicy, bandConfig.Priority, err)
		}

		s.priorityBands[bandConfig.Priority] = &priorityBand{
			config:                         bandConfig,
			queues:                         make(map[string]*managedQueue),
			interFlowDispatchPolicy:        interPolicy,
			defaultIntraFlowDispatchPolicy: intraPolicy,
		}
		s.orderedPriorityLevels = append(s.orderedPriorityLevels, bandConfig.Priority)
	}

	// Sort the priority levels to ensure deterministic iteration order.
	slices.Sort(s.orderedPriorityLevels)
	s.logger.V(logging.DEFAULT).Info("Registry shard initialized successfully",
		"priorityBandCount", len(s.priorityBands), "orderedPriorities", s.orderedPriorityLevels)
	return s, nil
}

// reconcileStats is the single point of entry for all statistics changes within the shard. It updates the relevant
// band's stats, the shard's total stats, and propagates the delta to the parent registry.
func (s *registryShard) reconcileStats(priority uint, lenDelta, byteSizeDelta int64) {
	s.totalLen.Add(uint64(lenDelta))
	s.totalByteSize.Add(uint64(byteSizeDelta))

	if band, ok := s.priorityBands[priority]; ok {
		band.len.Add(uint64(lenDelta))
		band.byteSize.Add(uint64(byteSizeDelta))
	}

	s.logger.V(logging.TRACE).Info("Reconciled shard stats", "priority", priority,
		"lenDelta", lenDelta, "byteSizeDelta", byteSizeDelta)

	if s.reconcileFun != nil {
		s.reconcileFun(lenDelta, byteSizeDelta)
	}
}

// ID returns the unique identifier for this shard.
func (s *registryShard) ID() string { return s.id }

// IsActive returns true if the shard is active and accepting new requests.
func (s *registryShard) IsActive() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.isActive
}

// ActiveManagedQueue returns the currently active `ManagedQueue` for a given flow.
func (s *registryShard) ActiveManagedQueue(flowID string) (contracts.ManagedQueue, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	mq, ok := s.activeFlows[flowID]
	if !ok {
		return nil, fmt.Errorf("failed to get active queue for flow %q: %w", flowID, contracts.ErrFlowInstanceNotFound)
	}
	return mq, nil
}

// ManagedQueue retrieves a specific (potentially draining) `ManagedQueue` instance from this shard.
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

// PriorityBandAccessor retrieves a read-only accessor for a given priority level.
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

// AllOrderedPriorityLevels returns all configured priority levels for this shard, sorted from highest to lowest
// priority (ascending numerical order).
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
			CapacityBytes: band.config.MaxBytes, // This is the partitioned capacity
			ByteSize:      band.byteSize.Load(),
			Len:           band.len.Load(),
		}
	}
	return stats
}

var _ contracts.RegistryShard = &registryShard{}

// --- priorityBandAccessor ---

// priorityBandAccessor implements `framework.PriorityBandAccessor`. It provides a read-only, concurrent-safe view of a
// single priority band within a shard.
type priorityBandAccessor struct {
	shard *registryShard
	band  *priorityBand
}

// Priority returns the numerical priority level of this band.
func (a *priorityBandAccessor) Priority() uint {
	return a.band.config.Priority
}

// PriorityName returns the human-readable name of this priority band.
func (a *priorityBandAccessor) PriorityName() string {
	return a.band.config.PriorityName
}

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

var _ framework.PriorityBandAccessor = &priorityBandAccessor{}
