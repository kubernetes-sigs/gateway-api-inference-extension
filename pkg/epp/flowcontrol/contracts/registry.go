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

package contracts

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// FlowRegistry is the complete interface for the global control plane, composed of administrative functions and the
// ability to provide shard accessors. A concrete implementation of this interface is the single source of truth for all
// flow control state and configuration.
//
// # Conformance
//
// All methods defined in this interface (including those embedded) MUST be goroutine-safe.
// Implementations are expected to perform complex updates (e.g., `RegisterOrUpdateFlow`, `UpdateShardCount`) atomically
// to preserve system invariants.
//
// # Invariants
//
// Concrete implementations of FlowRegistry MUST uphold the following invariants across all operations:
//  1. Shard Consistency: All configured priority bands and logical flows must be represented on every Active internal
//     shard. Plugin instance types (e.g., the specific `framework.SafeQueue` implementation or policy plugins) must be
//     consistent for a given flow or band across all shards.
//  2. Flow Instance Uniqueness per Band: For any given logical flow, there can be a maximum of one `ManagedQueue`
//     instance per priority band. An instance can be either Active or Draining.
//  3. Single Active Instance per Flow: For any given logical flow, there can be a maximum of one Active `ManagedQueue`
//     instance across all priority bands. All other instances for that flow must be in a Draining state.
//  4. Capacity Partitioning Consistency: Global and per-band capacity limits are uniformly partitioned across all
//     active shards. The sum of the capacity limits allocated to each shard must not exceed the globally configured
//     limits.
//
// # Flow Lifecycle States
//
//   - Registered: A logical flow is Registered when it is known to the `FlowRegistry`. It has exactly one Active
//     instance across all priority bands and zero or more Draining instances.
//   - Active: A specific instance of a flow within a priority band is Active if it is the designated target for all
//     new enqueues for that logical flow.
//   - Draining: A flow instance is Draining if it no longer accepts new enqueues but still contains items that are
//     eligible for dispatch. This occurs after a priority change.
//   - Garbage Collected (Unregistered): A logical flow is automatically unregistered and garbage collected by the
//     system when it has been 'idle' for a configurable period. A flow is considered idle if its active queue instance
//     has been empty on all active shards for the timeout duration. Once unregistered, it has no active instances,
//     though draining instances from previous priority levels may still exist until their queues are also empty.
//
// # Shard Garbage Collection
//
// When a shard is decommissioned via `UpdateShardCount`, the `FlowRegistry` must ensure a graceful shutdown. It must
// mark the shard as inactive to prevent new enqueues, allow the `FlowController` to continue draining its queues, and
// only delete the shard's state after the associated worker has fully terminated and all queues are empty.
type FlowRegistry interface {
	FlowRegistryAdmin
	ShardProvider
}

// FlowRegistryAdmin defines the administrative interface for the global control plane. This interface is intended for
// external systems to configure flows, manage system parallelism, and query aggregated statistics for observability.
//
// # Design Rationale for Dynamic Update Strategies
//
// The `FlowRegistryAdmin` contract specifies precise behaviors for handling dynamic updates. These strategies were
// chosen to prioritize system stability, correctness, and minimal disruption:
//
//   - Graceful Draining (for Priority/Shard Lifecycle Changes): For operations that change a flow's priority or
//     decommission a shard, the affected queue instances are marked as inactive but are not immediately deleted. They
//     enter a Draining state where they no longer accept new requests but are still processed for dispatch. This
//     ensures that requests already accepted by the system are processed to completion. Crucially, requests in a
//     draining queue continue to be dispatched according to the priority level and policies they were enqueued with,
//     ensuring consistency.
//
//   - Atomic Queue Migration (Future Design for Incompatible Intra-Flow Policy Changes): When an intra-flow policy is
//     updated to one that is incompatible with the existing queue data structure, the designed future behavior is a
//     full "drain and re-enqueue" migration. This more disruptive operation is necessary to guarantee correctness. A
//     simpler "graceful drain"—by creating a second instance of the same flow in the same priority band—is not used
//     because it would violate the system's "one flow instance per band" invariant. This invariant is critical because
//     it ensures that inter-flow policies operate on a clean set of distinct flows, stateful intra-flow policies have a
//     single authoritative view of their flow's state, and lookups are unambiguous. Note: This atomic migration is a
//     future design consideration and is not implemented in the current version.
//
//   - Self-Balancing on Shard Scale-Up: When new shards are added via `UpdateShardCount`, the framework relies on the
//     `FlowController`'s request distribution logic (e.g., a "Join the Shortest Queue by Bytes (JSQ-Bytes)" strategy)
//     to naturally funnel *new* requests to the less-loaded shards. This design choice strategically avoids the
//     complexity of actively migrating or rebalancing existing items that are already queued on other shards, promoting
//     system stability during scaling events.
type FlowRegistryAdmin interface {
	// RegisterOrUpdateFlow handles the registration of a new flow or the update of an existing flow's specification.
	// This method orchestrates complex state transitions atomically across all managed shards.
	//
	// # Dynamic Update Behaviors
	//
	//   - Priority Changes: If a flow's priority level changes, its current active `ManagedQueue` instance is marked
	//     as inactive to drain existing requests. A new instance is activated at the new priority level. If a flow is
	//     updated to a priority level where an instance is already draining (e.g., during a rapid rollback), that
	//     draining instance is re-activated.
	//
	// # Returns
	//
	//   - nil on success.
	//   - An error wrapping `ErrFlowIDEmpty` if `spec.ID` is empty.
	//   - An error wrapping`ErrPriorityBandNotFound` if `spec.Priority` refers to an unconfigured priority level.
	//   - Other errors if internal creation/activation of policy or queue instances fail.
	RegisterOrUpdateFlow(spec types.FlowSpecification) error

	// UpdateShardCount dynamically adjusts the number of internal state shards, triggering a state rebalance.
	//
	// # Dynamic Update Behaviors
	//
	//   - On Increase: New, empty state shards are initialized with all registered flows. The
	//     `controller.FlowController`'s request distribution logic will naturally balance load to these new shards over
	//     time.
	//   - On Decrease: A specified number of existing shards are marked as inactive. They stop accepting new requests
	//     but continue to drain existing items. They are fully removed only after their queues are empty.
	//
	// The implementation MUST atomically re-partition capacity allocations across all active shards when the count
	// changes.
	UpdateShardCount(n uint) error

	// Stats returns globally aggregated statistics for the entire `FlowRegistry`.
	Stats() AggregateStats

	// ShardStats returns a slice of statistics, one for each internal shard. This provides visibility for debugging and
	// monitoring per-shard behavior (e.g., identifying hot or stuck shards).
	ShardStats() []ShardStats
}

// ShardProvider defines a minimal interface for consumers that need to discover and iterate over available shards.
//
// A "shard" is an internal, parallel execution unit that allows the `FlowController`'s core dispatch logic to be
// parallelized. Consumers of this interface, such as a request distributor, MUST check `RegistryShard.IsActive()`
// before routing new work to a shard to ensure they do not send requests to a shard that is gracefully draining.
type ShardProvider interface {
	// Shards returns a slice of accessors, one for each internal state shard.
	//
	// A "shard" is an internal, parallel execution unit that allows the `controller.FlowController`'s core dispatch logic
	// to be parallelized, preventing a CPU bottleneck at high request rates. The `FlowRegistry`'s state is sharded to
	// support this parallelism by reducing lock contention.
	//
	// The returned slice includes accessors for both active and draining shards. Consumers MUST use `IsActive()` to
	// determine if new work should be routed to a shard. Callers should not modify the returned slice.
	Shards() []RegistryShard
}

// RegistryShard defines the read-oriented interface that a `controller.FlowController` worker uses to access its
// specific slice (shard) of the `FlowRegistry`'s state. It provides the necessary methods for a worker to perform its
// dispatch operations by accessing queues and policies in a concurrent-safe manner.
//
// # Conformance
//
// All methods MUST be goroutine-safe.
type RegistryShard interface {
	// ID returns a unique identifier for this shard, which must remain stable for the shard's lifetime.
	ID() string

	// IsActive returns true if the shard should accept new requests for enqueueing. A false value indicates the shard is
	// being gracefully drained and should not be given new work.
	IsActive() bool

	// ActiveManagedQueue returns the currently active `ManagedQueue` for a given flow on this shard. This is the queue to
	// which new requests for the flow should be enqueued.
	// Returns an error wrapping `ErrFlowInstanceNotFound` if no active instance exists for the given `flowID`.
	ActiveManagedQueue(flowID string) (ManagedQueue, error)

	// ManagedQueue retrieves a specific (potentially draining) `ManagedQueue` instance from this shard. This allows a
	// worker to continue dispatching items from queues that are draining as part of a flow update.
	// Returns an error wrapping `ErrFlowInstanceNotFound` if no instance for the given flowID and priority exists.
	ManagedQueue(flowID string, priority uint) (ManagedQueue, error)

	// IntraFlowDispatchPolicy retrieves a flow's configured `framework.IntraFlowDispatchPolicy` for this shard.
	// The registry guarantees that a non-nil default policy (as configured at the priority-band level) is returned if
	// none is specified on the flow itself.
	// Returns an error wrapping `ErrFlowInstanceNotFound` if the flow instance does not exist.
	IntraFlowDispatchPolicy(flowID string, priority uint) (framework.IntraFlowDispatchPolicy, error)

	// InterFlowDispatchPolicy retrieves a priority band's configured `framework.InterFlowDispatchPolicy` for this shard.
	// The registry guarantees that a non-nil default policy is returned if none is configured for the band.
	// Returns an error wrapping `ErrPriorityBandNotFound` if the priority level is not configured.
	InterFlowDispatchPolicy(priority uint) (framework.InterFlowDispatchPolicy, error)

	// PriorityBandAccessor retrieves a read-only accessor for a given priority level, providing a view of the band's
	// state as seen by this specific shard. This is the primary entry point for inter-flow dispatch policies that
	// need to inspect and compare multiple flow queues within the same priority band.
	// Returns an error wrapping `ErrPriorityBandNotFound` if the priority level is not configured.
	PriorityBandAccessor(priority uint) (framework.PriorityBandAccessor, error)

	// AllOrderedPriorityLevels returns all configured priority levels that this shard is aware of, sorted in ascending
	// numerical order. This order corresponds to highest priority (lowest numeric value) to lowest priority (highest
	// numeric value).
	// The returned slice provides a definitive, ordered list of priority levels for iteration, for example, by a
	// `controller.FlowController` worker's dispatch loop.
	AllOrderedPriorityLevels() []uint

	// Stats returns a snapshot of the statistics for this specific shard.
	Stats() ShardStats
}

// ManagedQueue defines the interface for a flow's queue instance on a specific shard.
// It acts as a stateful decorator around an underlying `framework.SafeQueue`, augmenting it with lifecycle validation
// against the `FlowRegistry` and integrating atomic statistics updates.
//
// # Conformance
//
//   - All methods defined by this interface and the `framework.SafeQueue` it wraps MUST be goroutine-safe.
//   - The `Add()` method MUST reject new items if the queue has been marked as Draining by the `FlowRegistry`, ensuring
//     that lifecycle changes are respected even by consumers holding a stale pointer to the queue.
//   - All mutating methods (`Add()`, `Remove()`, `Cleanup()`, `Drain()`) MUST atomically update relevant statistics
//     (e.g., queue length, byte size).
type ManagedQueue interface {
	framework.SafeQueue

	// FlowQueueAccessor returns a read-only, flow-aware accessor for this queue.
	// This accessor is primarily used by policy plugins to inspect the queue's state in a structured way.
	//
	// Conformance: This method MUST NOT return nil.
	FlowQueueAccessor() framework.FlowQueueAccessor
}

// AggregateStats holds globally aggregated statistics for the entire `FlowRegistry`.
type AggregateStats struct {
	// TotalCapacityBytes is the globally configured maximum total byte size limit across all priority bands and shards.
	TotalCapacityBytes uint64
	// TotalByteSize is the total byte size of all items currently queued across the entire system.
	TotalByteSize uint64
	// TotalLen is the total number of items currently queued across the entire system.
	TotalLen uint64
	// PerPriorityBandStats maps each configured priority level to its globally aggregated statistics.
	PerPriorityBandStats map[uint]PriorityBandStats
}

// ShardStats holds statistics for a single internal shard within the `FlowRegistry`.
type ShardStats struct {
	// TotalCapacityBytes is the optional, maximum total byte size limit aggregated across all priority bands within this
	// shard. Its value represents the globally configured limit for the `FlowRegistry` partitioned for this shard.
	// The `controller.FlowController` enforces this limit in addition to any per-band capacity limits.
	// A value of 0 signifies that this global limit is ignored, and only per-band limits apply.
	TotalCapacityBytes uint64
	// TotalByteSize is the total byte size of all items currently queued across all priority bands within this shard.
	TotalByteSize uint64
	// TotalLen is the total number of items currently queued across all priority bands within this shard.
	TotalLen uint64
	// PerPriorityBandStats maps each configured priority level to its statistics within this shard.
	// The capacity values within represent this shard's partition of the global band capacity.
	// The key is the numerical priority level.
	// All configured priority levels are guaranteed to be represented.
	PerPriorityBandStats map[uint]PriorityBandStats
}

// DeepCopy returns a deep copy of the `ShardStats`.
func (s *ShardStats) DeepCopy() ShardStats {
	if s == nil {
		return ShardStats{}
	}
	newStats := *s
	if s.PerPriorityBandStats != nil {
		newStats.PerPriorityBandStats = make(map[uint]PriorityBandStats, len(s.PerPriorityBandStats))
		for k, v := range s.PerPriorityBandStats {
			newStats.PerPriorityBandStats[k] = v.DeepCopy()
		}
	}
	return newStats
}

// PriorityBandStats holds aggregated statistics for a single priority band.
type PriorityBandStats struct {
	// Priority is the numerical priority level this struct describes.
	Priority uint
	// PriorityName is an optional, human-readable name for the priority level (e.g., "Critical", "Sheddable").
	PriorityName string
	// CapacityBytes is the configured maximum total byte size for this priority band.
	// When viewed via `AggregateStats`, this is the global limit. When viewed via `ShardStats`, this is the partitioned
	// value for that specific shard.
	// The `controller.FlowController` enforces this limit.
	// A default non-zero value is guaranteed if not configured.
	CapacityBytes uint64
	// ByteSize is the total byte size of items currently queued in this priority band.
	ByteSize uint64
	// Len is the total number of items currently queued in this priority band.
	Len uint64
}

// DeepCopy returns a deep copy of the `PriorityBandStats`.
func (s *PriorityBandStats) DeepCopy() PriorityBandStats {
	if s == nil {
		return PriorityBandStats{}
	}
	return *s
}
