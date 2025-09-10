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

// FlowRegistry is the complete interface for the global flow control plane.
// It composes the client-facing data path interface and the administrative interface. A concrete implementation of this
// interface is the single source of truth for all flow control state.
//
// # Conformance: Implementations MUST be goroutine-safe.
//
// # Flow Lifecycle
//
// A flow instance, identified by its immutable `types.FlowKey`, has a lease-based lifecycle managed by this interface.
// Any implementation MUST adhere to this lifecycle:
//
//  1. Lease Acquisition: A client calls Connect to acquire a lease. This signals that the flow is in use and protects
//     it from garbage collection. If the flow does not exist, it is created Just-In-Time (JIT).
//  2. Active State: A flow is "Active" as long as its lease count is greater than zero.
//  3. Lease Release: The client MUST call `Close()` on the returned `FlowConnection` to release the lease.
//     When the lease count drops to zero, the flow becomes "Idle".
//  4. Garbage Collection: The implementation MUST automatically garbage collect a flow after it has remained
//     continuously Idle for a configurable duration.
//
// # System Invariants
//
// Concrete implementations MUST uphold the following invariants:
//
//  1. Shard Consistency: All configured priority bands and registered flow instances must exist on every Active shard.
//  2. Capacity Partitioning: Global and per-band capacity limits must be uniformly partitioned across all Active
//     shards.
type FlowRegistry interface {
	FlowRegistryClient
	FlowRegistryAdmin
}

// FlowRegistryAdmin defines the administrative interface for the global control plane.
type FlowRegistryAdmin interface {
	// Stats returns globally aggregated statistics for the entire `FlowRegistry`.
	Stats() AggregateStats

	// ShardStats returns a slice of statistics, one for each internal shard.
	ShardStats() []ShardStats
}

// FlowRegistryClient defines the primary, client-facing interface for the registry.
// This is the interface that the `controller.FlowController`'s data path depends upon.
type FlowRegistryClient interface {
	// WithConnection manages a scoped, leased session for a given flow.
	// It is the primary and sole entry point for interacting with the data path.
	//
	// This method handles the entire lifecycle of a flow connection:
	// 1. Just-In-Time (JIT) Registration: If the flow for the given `types.FlowKey` does not exist, it is created and
	//    registered automatically.
	// 2. Lease Acquisition: It acquires a lifecycle lease, protecting the flow from garbage collection.
	// 3. Callback Execution: It invokes the provided function `fn`, passing in a temporary `ActiveFlowConnection` handle.
	// 4. Guaranteed Lease Release: It ensures the lease is safely released when the callback function returns.
	//
	// This functional, callback-based approach makes resource leaks impossible, as the caller is not responsible for
	// manually closing the connection.
	//
	// Errors returned by the callback `fn` are propagated up.
	// Returns `ErrFlowIDEmpty` if the provided key has an empty ID.
	WithConnection(key types.FlowKey, fn func(conn ActiveFlowConnection) error) error
}

// ActiveFlowConnection represents a handle to a temporary, leased session on a flow.
// It provides a safe, scoped entry point to the registry's sharded data plane.
//
// An `ActiveFlowConnection` instance is only valid for the duration of the `WithConnection` callback from which it was
// received. Callers MUST NOT store a reference to this object or use it after the callback returns.
// Its purpose is to ensure that any interaction with the flow's state (e.g., accessing its shards and queues) occurs
// safely while the flow is guaranteed to be protected from garbage collection.
type ActiveFlowConnection interface {
	// Shards returns a stable snapshot of accessors for all internal state shards (both Active and Draining).
	// Consumers MUST check `RegistryShard.IsActive()` before routing new work to a shard from this slice.
	Shards() []RegistryShard
}

// RegistryShard defines the interface for a single slice (shard) of the `FlowRegistry`'s state.
// A shard acts as an independent, parallel execution unit, allowing the system's dispatch logic to scale horizontally.
//
// # Conformance: Implementations MUST be goroutine-safe.
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
	InterFlowDispatchPolicy(priority int) (framework.InterFlowDispatchPolicy, error)

	// PriorityBandAccessor retrieves a read-only accessor for a given priority level, providing a view of the band's
	// state as seen by this specific shard. This is the primary entry point for inter-flow dispatch policies that need to
	// inspect and compare multiple flow queues within the same priority band.
	// Returns an error wrapping `ErrPriorityBandNotFound` if the priority level is not configured.
	PriorityBandAccessor(priority int) (framework.PriorityBandAccessor, error)

	// AllOrderedPriorityLevels returns all configured priority levels that this shard is aware of, sorted in descending
	// numerical order. This order corresponds to highest priority (highest numeric value) to lowest priority (lowest
	// numeric value).
	// The returned slice provides a definitive, ordered list of priority levels for iteration, for example, by a
	// `controller.FlowController` worker's dispatch loop.
	AllOrderedPriorityLevels() []int

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
	PerPriorityBandStats map[int]PriorityBandStats
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
	PerPriorityBandStats map[int]PriorityBandStats
}

// PriorityBandStats holds aggregated statistics for a single priority band.
type PriorityBandStats struct {
	// Priority is the numerical priority level this struct describes.
	Priority int
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
