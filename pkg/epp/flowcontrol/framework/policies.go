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

package framework

import "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"

// PriorityScoreType is a descriptor for the domain of a policy's item comparator.
type PriorityScoreType string

const (
	// EnqueueTimePriorityScoreType indicates that the priority is based on the item's enqueue time, with earlier times
	// being higher priority.
	EnqueueTimePriorityScoreType PriorityScoreType = "enqueue_time_ns_asc"
)

// ItemComparatorFunc defines the function signature for comparing two `types.QueueItemAccessor` instances to determine
// their relative dispatch priority.
//
// An implementation of this function determines if item 'a' should be dispatched before item 'b'. It returns true if
// 'a' is of higher priority, and false otherwise. The specific criteria for "higher priority" (e.g., earlier deadline,
// lower enqueue time) are defined by the `IntraFlowDispatchPolicy` that vends this function via an `ItemComparator`.
type ItemComparatorFunc func(a, b types.QueueItemAccessor) bool

// ItemComparator encapsulates the logic for comparing two `types.QueueItemAccessor` instances to determine their
// relative dispatch priority. It is the definitive source of ordering truth for a flow's queue.
//
// It is vended by an `IntraFlowDispatchPolicy` to make its internal ordering logic explicit and available to other
// components. It is used by `SafeQueue` implementations that support the `CapabilityPriorityConfigurable` capability
// and can also be used by inter-flow policies to compare items from different queues.
//
// Design Justification: This design treats item priority as a relational concept defined by a policy, rather than a
// static attribute on the item itself. This allows for sophisticated, dynamic priority evaluation (e.g., based on
// real-time SLO attainment), as the comparison logic can be stateful.
type ItemComparator interface {
	// Func returns the core comparison logic as an `ItemComparatorFunc`.
	//
	// This function is the single source of truth for determining the relative priority between two items. A `SafeQueue`
	// that declares `CapabilityPriorityConfigurable` MUST use this function for its internal ordering. Inter-flow
	// policies MAY use this function to compare items from different queues, but only after verifying that their
	// `ScoreType()` values are identical, ensuring the comparison is meaningful.
	//
	// Conformance: MUST NOT return nil.
	Func() ItemComparatorFunc

	// ScoreType returns a string descriptor that defines the semantic meaning and domain of the comparison logic.
	//
	// A non-empty, descriptive string is required for two primary reasons:
	//  1. Comparability Check: Inter-flow policies that compare items across different queues (e.g., a "BestHead" policy)
	//     MUST check for identical `ScoreType` strings before using the comparator functions. A comparison is only
	//     meaningful if the underlying scoring logic is the same.
	//  2. Introspectability: The string makes the priority scheme human-readable for debugging and observability.
	//
	// Examples: "enqueue_time_ns_asc", "slo_urgency_score_desc".
	//
	// Future Considerations: While currently a simple string for initial simplicity, a future enhancement could introduce
	// a more structured `ScoreType`. Such a structure might explicitly encode ordering (ascending/descending) and value
	// semantics (e.g., time, custom_metric), potentially enabling advanced features like cross-`ScoreType` normalization
	// plugins.
	//
	// Conformance:
	//   - MUST return a non-empty, meaningful string that describes the domain or unit of comparison.
	//   - For the present, policies MUST NOT assume any implicit cross-`ScoreType` normalization capabilities.
	ScoreType() string
}

// IntraFlowDispatchPolicy selects a specific request to dispatch next from a single flow's queue.
// Implementations define the dispatch ordering of requests within a single flow.
//
// For example, a "First-Come, First-Served" (FCFS) policy would typically inspect a queue and select the item that was
// enqueued the earliest.
type IntraFlowDispatchPolicy interface {
	// Name returns a string identifier for the concrete policy implementation type (e.g., "FCFS").
	Name() string

	// SelectItem inspects a flow's queue and returns the `types.QueueItemAccessor` of the item chosen for dispatch.
	//
	// For queues that inherently order items by dispatch preference, this method will typically just call
	// `queue.PeekHead()`.
	//
	// The `controller.FlowController` uses the handle from the returned item to instruct the `contracts.ManagedQueue` to
	// remove it.
	//
	// Returns:
	//   - `types.QueueItemAccessor`: The selected item, or nil if no item is chosen.
	//   - error: Non-nil if an unrecoverable error occurs. A nil error is returned if no item is selected (e.g., the
	//     queue is empty or the policy logic determines a pause is appropriate).
	//
	// Conformance: Implementations MUST be goroutine-safe if they maintain internal state.
	SelectItem(queue FlowQueueAccessor) (selectedItem types.QueueItemAccessor, err error)

	// Comparator returns the `ItemComparator` that defines this policy's item ordering logic. This is the definitive
	// source for how items within a flow governed by this policy should be prioritized.
	//
	// A policy MUST provide a meaningful comparator even if it relies on a queue's inherent ordering (e.g., an FCFS
	// policy using a `CapabilityFIFO` queue should return a comparator based on enqueue time). This makes the ordering
	// principle explicit and available to other components, like inter-flow policies.
	//
	// Conformance: MUST NOT return nil.
	Comparator() ItemComparator

	// RequiredQueueCapabilities returns a slice of capabilities that the `SafeQueue` used with this policy MUST support.
	// This policy is responsible for defining the ordering of items within a flow and so it must require the relevant
	// *behavioral* capability (e.g., `CapabilityPriorityConfigurable` or `CapabilityFIFO`). The `ItemComparator` vended
	// by this policy then defines that behavior.
	RequiredQueueCapabilities() []QueueCapability
}

// InterFlowDispatchPolicy selects which flow's queue to service next from a given priority band.
// Implementations define the fairness or dispatch ordering logic between different flows that share the same priority
// level.
type InterFlowDispatchPolicy interface {
	// Name returns a string identifier for the concrete policy implementation type (e.g., "RoundRobin").
	Name() string

	// SelectQueue inspects the flow queues within the provided `PriorityBandAccessor` and returns the `FlowQueueAccessor`
	// of the queue chosen for the next dispatch attempt.
	//
	// Returns:
	//   - `FlowQueueAccessor`: The selected queue, or nil if no queue is chosen.
	//   - error: Non-nil if an unrecoverable error occurs. A nil error is returned if no queue is selected (e.g., all
	//     queues in the band are empty or the policy logic determines a pause is appropriate).
	//
	// Policies should be resilient to transient issues (like a queue becoming empty during inspection) and select from
	// other available queues if possible, rather than returning an error for such conditions.
	//
	// Conformance: Implementations MUST be goroutine-safe if they maintain internal state.
	SelectQueue(band PriorityBandAccessor) (selectedQueue FlowQueueAccessor, err error)
}

// FlowQueueAccessor provides a policy-facing, read-only view of a single flow's queue.
// It combines general queue inspection methods (embedded via `QueueInspectionMethods`) with flow-specific metadata.
//
// Instances of `FlowQueueAccessor` are typically obtained via a `PriorityBandAccessor` and are the primary means by
// which policies inspect individual queue state.
//
// Conformance: Implementations MUST ensure all methods (including those embedded from `QueueInspectionMethods`) are
// goroutine-safe for concurrent access.
type FlowQueueAccessor interface {
	QueueInspectionMethods

	// Comparator returns the `ItemComparator` that defines the ordering logic of the items within this queue.
	// This is determined by the `IntraFlowDispatchPolicy` associated with this queue's flow.
	Comparator() ItemComparator

	// FlowKey returns the unique, immutable `types.FlowKey` of the flow instance this queue accessor is associated with.
	// This provides essential context (like the logical grouping `ID` and `Priority`) to policies.
	FlowKey() types.FlowKey
}

// PriorityBandAccessor provides a read-only view into a specific priority band within the `contracts.FlowRegistry`.
// It allows the `controller.FlowController` and inter-flow policies to inspect the state of all flow queues within that
// band.
//
// Conformance: Implementations MUST ensure all methods are goroutine-safe for concurrent access.
type PriorityBandAccessor interface {
	// Priority returns the numerical priority level of this band.
	Priority() int

	// PriorityName returns the human-readable name of this priority band.
	PriorityName() string

	// FlowKeys returns a slice of the composite `types.FlowKey`s for every flow instance currently active within this
	// priority band.
	// This method provides the complete, canonical identifiers for all flows in the band. The caller can use the `ID`
	// field from each key to look up a specific queue via the `Queue(id string)` method.
	// The order of keys in the returned slice is not guaranteed unless specified by the implementations (e.g., for
	// deterministic testing scenarios).
	FlowKeys() []types.FlowKey

	// Queue returns a `FlowQueueAccessor` for the specified logical grouping `ID` within this priority band.
	// Note: This uses the logical `ID` string (`types.FlowKey.ID`), not the full `types.FlowKey`, as the priority is
	// already defined by the accessor's scope.
	//
	// Conformance: Implementations MUST return nil if the `ID` is not found in this band.
	Queue(id string) FlowQueueAccessor

	// IterateQueues executes the given `callback` for each `FlowQueueAccessor` in this priority band.
	// Iteration stops if the `callback` returns false. The order of iteration is not guaranteed unless specified by the
	// implementation (e.g., for deterministic testing scenarios).
	IterateQueues(callback func(queue FlowQueueAccessor) (keepIterating bool))
}
