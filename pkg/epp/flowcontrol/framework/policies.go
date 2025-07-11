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
// It is vended by an `IntraFlowDispatchPolicy` and used by `SafeQueue` implementations that support the
// `CapabilityPriorityConfigurable` capability. It can also be used by inter-flow policies to compare items from
// different queues, provided their `ScoreType` is compatible.
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
