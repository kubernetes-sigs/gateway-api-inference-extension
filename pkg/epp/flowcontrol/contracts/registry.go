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

// FlowRegistry is the complete interface for the global control plane. An implementation of this interface is the single
// source of truth for all flow control state and configuration.
//
// # Conformance
//
// All methods MUST be goroutine-safe. Implementations are expected to perform complex updates (e.g.,
// `RegisterOrUpdateFlow`) atomically.
//
// # System Invariants
//
// Concrete implementations MUST uphold the following invariants:
//  1. Shard Consistency: All configured priority bands and registered flow instances must exist on every Active shard.
//     Plugin instance types must be consistent for a given flow across all shards.
//  2. Flow Instance Uniqueness: Each unique `types.FlowKey` (`ID` + `Priority`) corresponds to exactly one managed flow
//     instance.
//  3. Capacity Partitioning: Global and per-band capacity limits must be uniformly partitioned across all Active
//     shards.
//
// # Flow Lifecycle
//
// A flow instance (identified by its immutable `FlowKey`) has a simple lifecycle:
//
//   - Registered: Known to the `FlowRegistry` via `RegisterOrUpdateFlow`.
//   - Idle: Queues are empty across all Active and Draining shards.
//   - Garbage Collected (Unregistered): The registry automatically garbage collects flows after they have remained Idle
//     for a configurable duration.
//
// # Shard Lifecycle
//
// When a shard is decommissioned, it is marked inactive (`IsActive() == false`) to prevent new enqueues. The shard
// continues to drain and is deleted only after it is empty.
type FlowRegistry interface {
	FlowRegistryAdmin
	ShardProvider
}

// FlowRegistryAdmin defines the administrative interface for the global control plane.
//
// # Dynamic Update Strategies
//
// The contract specifies behaviors for handling dynamic updates, prioritizing stability and correctness:
//
//   - Immutable Flow Identity (`types.FlowKey`): The system treats the `FlowKey` (`ID` + `Priority`) as the immutable
//     identifier. Changing the priority of traffic requires registering a new `FlowKey`. The old flow instance is
//     automatically garbage collected when Idle. This design eliminates complex priority migration logic.
//
//   - Graceful Draining (Shard Scale-Down): Decommissioned shards enter a Draining state. They stop accepting new
//     requests but continue to be processed for dispatch until empty.
//
//   - Self-Balancing (Shard Scale-Up): When new shards are added, the `controller.FlowController`'s distribution logic
//     naturally utilizes them, funneling new requests to the less-loaded shards. Existing queued items are not
//     migrated.
type FlowRegistryAdmin interface {
	// RegisterOrUpdateFlow handles the registration of a new flow instance or the update of an existing instance's
	// specification (for the same `types.FlowKey`). The operation is atomic across all shards.
	//
	// Since the `FlowKey` (including `Priority`) is immutable, this method cannot change a flow's priority.
	// To change priority, the caller should simply register the new `FlowKey`; the old flow instance will be
	// automatically garbage collected when it becomes Idle.
	//
	// Returns errors wrapping `ErrFlowIDEmpty`, `ErrPriorityBandNotFound`, or internal errors if plugin instantiation
	// fails.
	RegisterOrUpdateFlow(spec types.FlowSpecification) error

	// UpdateShardCount dynamically adjusts the number of internal state shards.
	//
	// The implementation MUST atomically re-partition capacity allocations across all active shards.
	// Returns an error wrapping `ErrInvalidShardCount` if `n` is not positive.
	UpdateShardCount(n int) error

	// Stats returns globally aggregated statistics for the entire `FlowRegistry`.
	Stats() AggregateStats

	// ShardStats returns a slice of statistics, one for each internal shard. This provides visibility for debugging and
	// monitoring per-shard behavior (e.g., identifying hot or stuck shards).
	ShardStats() []ShardStats
}

// ShardProvider defines the interface for discovering available shards.
//
// A "shard" is an internal, parallel execution unit that allows the `controller.FlowController`'s core dispatch logic
// to be parallelized, preventing a CPU bottleneck at high request rates. The `FlowRegistry`'s state is sharded to
// support this parallelism by reducing lock contention.
//
// Consumers MUST check `RegistryShard.IsActive()` before routing new work to a shard to avoid sending requests to a
// Draining shard.
type ShardProvider interface {
	// Shards returns a slice of accessors, one for each internal state shard (Active and Draining).
	// Callers should not modify the returned slice.
	Shards() []RegistryShard
}

// RegistryShard defines the interface for accessing a specific slice (shard) of the `FlowRegistry's` state.
// It provides a concurrent-safe view for `controller.FlowController` workers.
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

	// ManagedQueue retrieves the managed queue for the given, unique `types.FlowKey`. This is the primary method for
	// accessing a specific flow's queue for either enqueueing or dispatching requests.
	//
	// Returns an error wrapping `ErrPriorityBandNotFound` if the priority specified in the `key` is not configured, or
	// `ErrFlowInstanceNotFound` if no instance exists for the given `key`.
	ManagedQueue(key types.FlowKey) (ManagedQueue, error)

	// IntraFlowDispatchPolicy retrieves a flow's configured `framework.IntraFlowDispatchPolicy` for this shard,
	// identified by its unique `types.FlowKey`.
	// The registry guarantees that a non-nil default policy (as configured at the priority-band level) is returned if
	// none is specified for the flow.
	// Returns an error wrapping `ErrFlowInstanceNotFound` if the flow instance does not exist.
	IntraFlowDispatchPolicy(key types.FlowKey) (framework.IntraFlowDispatchPolicy, error)

	// InterFlowDispatchPolicy retrieves a priority band's configured `framework.InterFlowDispatchPolicy` for this shard.
	// The registry guarantees that a non-nil default policy is returned if none is configured for the band.
	// Returns an error wrapping `ErrPriorityBandNotFound` if the priority level is not configured.
	InterFlowDispatchPolicy(priority uint) (framework.InterFlowDispatchPolicy, error)

	// PriorityBandAccessor retrieves a read-only accessor for a given priority level, providing a view of the band's
	// state as seen by this specific shard. This is the primary entry point for inter-flow dispatch policies that need to
	// inspect and compare multiple flow queues within the same priority band.
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

// ManagedQueue defines the interface for a flow's queue on a specific shard.
// It acts as a stateful decorator around an underlying `framework.SafeQueue`, augmenting it with statistics tracking.
//
// # Conformance
//
//   - Implementations MUST be goroutine-safe.
//   - All mutating methods MUST ensure that the underlying queue state and the public statistics (`Len`, `ByteSize`)
//     are updated as a single atomic transaction.
//   - The `Add` method MUST return an error wrapping `ErrShardDraining` if the queue instance belongs to a parent shard
//     that is no longer Active.
type ManagedQueue interface {
	framework.SafeQueue

	// FlowQueueAccessor returns a read-only, flow-aware accessor for this queue, used by policy plugins.
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

// PriorityBandStats holds aggregated statistics for a single priority band.
type PriorityBandStats struct {
	// Priority is the numerical priority level this struct describes.
	Priority uint
	// PriorityName is a human-readable name for the priority band (e.g., "Critical", "Sheddable").
	// The registry configuration requires this field, so it is guaranteed to be non-empty.
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
