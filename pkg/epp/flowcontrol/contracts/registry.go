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

// Package contracts defines the service interfaces that decouple the core `controller.FlowController` engine from its
// primary dependencies. In alignment with a "Ports and Adapters" (or "Hexagonal") architectural style, these
// interfaces represent the "ports" through which the engine communicates.
//
// This package contains the primary service contracts for the Flow Registry, which acts as the control plane for all
// flow state and configuration.
package contracts

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
)

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
// It wraps an underlying `framework.SafeQueue`, augmenting it with lifecycle validation against the `FlowRegistry` and
// integrating atomic statistics updates.
//
// # Conformance
//
//   - All methods (including those embedded from `framework.SafeQueue`) MUST be goroutine-safe.
//   - The `Add()` method MUST reject new items if the queue has been marked as "draining" by the `FlowRegistry`,
//     ensuring that lifecycle changes are respected even by consumers holding a stale pointer to the queue.
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
	// CapacityBytes is the configured maximum total byte size for this priority band, aggregated across all items in
	// all flow queues within this band. If scoped to a shard, its value represents the configured band limit for the
	// `FlowRegistry` partitioned for this shard.
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
