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

package flowcontrol

import (
	"context"
	"errors"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// EvictableQueue provides the minimal interface needed by evictors to remove items from queues.
// This avoids import cycles by not depending on the contracts package.
// Implementations of contracts.ManagedQueue satisfy this interface.
type EvictableQueue interface {
	// Remove atomically finds and removes an item from the queue using its handle.
	Remove(handle QueueItemHandle) (QueueItemAccessor, error)
}

var (
	// ErrIncompatiblePriorityType indicates that a FairnessPolicy attempted to compare items from two different flow
	// queues whose ItemComparators have different ScoreType values, making a meaningful comparison impossible.
	ErrIncompatiblePriorityType = errors.New("incompatible priority score type for comparison")
)

// FairnessPolicy governs the distribution of dispatch opportunities among competing Flows within the same Priority
// Band.
//
// In simple terms, this policy answers the question: "Which flow gets to dispatch a request next?"
//
// While "Priority" determines strictly which group of flows is serviced first, "Fairness" determines how resources are
// shared when multiple flows in that same group are fighting for capacity.
//
// Architecture (Flyweight Pattern):
// Fairness plugins are Singletons. A single instance of a FairnessPolicy handles the logic for potentially many
// different Priority Bands. To support this, the plugin must be purely functional, separating its Logic (methods) from
// its State (data).
//
//   - Logic: Defined here in the FairnessPolicy interface.
//   - State: Created via NewState() and stored on the PriorityBandAccessor.
//
// Conformance: Implementations MUST ensure all methods are goroutine-safe.
type FairnessPolicy interface {
	plugin.Plugin

	// NewState creates the scoped, mutable storage required by this policy for a single Priority Band.
	//
	// Because the plugin instance itself is shared globally, it cannot hold state like "current round-robin index" or
	// "accumulated deficits" inside struct fields. Instead, it creates this state object once per Band.
	//
	// The Flow Registry manages the lifecycle of this object, storing it on the Priority Band and passing it back to the
	// plugin via the PriorityBandAccessor during Pick.
	//
	// Returns:
	//   - any: The opaque state object (e.g., &roundRobinCursor{index: 0}).
	NewState(ctx context.Context) any

	// Pick inspects the active flows in the provided Flow Group (Priority Band) and selects the "winner" for the next
	// dispatch attempt.
	//
	// This is the core logic loop. The implementation should:
	//  1. Retrieve its scoped state from band.GetPolicyState().
	//  2. Cast the state to its concrete type (e.g., *roundRobinCursor).
	//  3. Apply its algorithm to select a FlowQueueAccessor.
	//  4. Update the state (e.g., increment the cursor) if necessary.
	//
	// State may also be updated out-of-band (e.g., from monitoring a metrics server, from integrating with request
	// lifeycle hooks, etc.).
	//
	// Returns:
	//   - flow: The Flow to service next. Returns nil if no valid candidate is found (e.g., all queues empty).
	//   - err: Only returned for unrecoverable internal errors. Policies should generally return (nil, nil) if simply
	//     nothing is eligible.
	Pick(ctx context.Context, flowGroup PriorityBandAccessor) (flow FlowQueueAccessor, err error)
}

// OrderingPolicy governs the strict sequence of service within a single Flow.
//
// In simple terms, this policy answers the question: "Which request in this specific queue should be processed next?"
//
// While "Fairness" governs the competition between flows, "Ordering" dictates the internal discipline of a single
// flow. This allows different flows to have different internal service objectives (e.g., FCFS vs. EDF).
//
// Architecture (Flyweight Pattern):
// Ordering policies are Singletons. A single instance handles the logic for all queues in a Priority Band.
// The policy is purely functional.
//
//   - Logic: Defined here as a Comparator-centric interface.
//   - State: Ordering policies are generally stateless, operating on the intrinsic properties of the items.
//
// Conformance: Implementations MUST ensure all methods are goroutine-safe.
type OrderingPolicy interface {
	plugin.Plugin

	// Less reports whether item 'a' should be dispatched before item 'b'.
	// This makes the policy act as a sort.Interface for the queue.
	//
	// Invariants:
	//   - Returning true means 'a' has higher priority than 'b'.
	//   - If the queue supports CapabilityPriorityConfigurable, this function determines the heap order.
	Less(a, b QueueItemAccessor) bool

	// RequiredQueueCapabilities returns the set of capabilities that a SafeQueue MUST support to effectively apply this
	// policy.
	//
	// For example:
	//   - "fcfs-ordering-policy" coupled with CapabilityFIFO is O(1).
	//   - "edf-ordering-policy" (Earliest Deadline First) REQUIRES CapabilityPriorityConfigurable (Heap) to function
	//     correctly.
	RequiredQueueCapabilities() []QueueCapability
}

// UsageLimitPolicy computes the usage limit of a priority band dynamically.
//
// The goal of this policy is to enable adaptive capacity management by gating low-priority traffic
// when a request target approaches saturation, reserving capacity for higher-priority requests.
//
// Saturation represents resource usage as a fraction of total capacity (0.0 = idle, 1.0 = fully saturated)
// as described in [/pkg/epp/flowcontrol/contracts.SaturationDetector]
//
// Architecture (Singleton with Internal State):
// UsageLimitPolicy plugins are Singletons. A single instance handles limit computation for all priority bands
// across all shards. The plugin maintains internal state (saturation history, trend derivatives, per-priority
// limits) to enable trend-based decisions and smooth limit adjustments over time.
//
// Integration:
// This policy is called during dispatch decision-making, before a request is allowed to proceed. The returned
// limit is compared against the request's projected saturation impact. If saturation + request > limit, the
// request is gated (not dispatched).
//
// Conformance: Implementations MUST ensure all methods are goroutine-safe.
type UsageLimitPolicy interface {
	plugin.Plugin

	// ComputeLimit calculates the dynamic usage limit for a given priority level based on current saturation
	// and historical trends.
	//
	// Parameters:
	//   - ctx: Request context for logging, tracing, etc.
	//   - priority: The priority level for which to compute the limit (higher numbers = higher priority)
	//   - saturation: Current resource saturation as a fraction [0.0, 1.0]
	//   - requestMetadata: Optional request-specific metadata (may include endpoint subset identifiers)
	//
	// Returns:
	//   - limit: The maximum saturation threshold at which this priority can dispatch
	//     - 0.0 = fully gated (cannot dispatch regardless of current saturation)
	//     - 1.0 = no gating (can dispatch until fully saturated)
	//     - Values between 0.0 and 1.0 reserve capacity headroom
	//
	ComputeLimit(ctx context.Context, priority int, saturation float64, requestMetadata map[string]any) (limit float64)
}

// Evictor handles capacity reclamation by removing queued low-priority requests when the system
// approaches saturation thresholds.
//
// Eviction Model (Two-Phase):
// The evictor uses a two-phase approach to separate eviction candidate selection from actual removal:
//
//  1. Selection Phase (during dispatch cycle):
//     When a request is gated by usage limits, it is scheduled as an eviction candidate via ScheduleEviction.
//     The framework passes the queue and item, allowing the evictor to track candidates for later removal.
//
//  2. Eviction Phase (end of dispatch cycle):
//     ProcessScheduled is called to actually remove scheduled candidates from their queues.
//     The evictor can apply its own logic to decide which candidates to evict based on current state.
//
// This two-phase design allows the evictor to:
//   - Accumulate a window of gated items before making eviction decisions
//   - Cancel scheduled evictions if saturation improves
//   - Apply custom policies (e.g., LIFO, priority-based, time-based)
//   - Batch removals for efficiency
//
// Usage Limit Integration:
// Evictors work in coordination with UsageLimitPolicy:
//   - Items gated by usage limits are scheduled for eviction
//   - The evictor can observe saturation trends to decide whether to proceed with eviction
//   - If saturation drops naturally, scheduled evictions can be cancelled
//
// Future: In-Flight Cancellation:
// Currently, evictors operate only on queued requests. Future enhancements may support canceling
// in-flight requests that are already executing. This would require additional infrastructure for
// request lifecycle tracking and cancellation propagation.
//
// Conformance: Implementations MUST ensure all methods are goroutine-safe.
type Evictor interface {
	plugin.Plugin

	// ScheduleEvictionCandidate registers a gated item as a candidate for eviction.
	//
	// This method is called during the dispatch cycle when an item cannot proceed due to usage limits.
	// The evictor stores the queue, item, and gating context for potential removal during ProcessScheduled.
	//
	// The evictor may apply its own logic to decide whether to actually evict the item later:
	//   - Track saturation trends (cancel evictions if saturation improves)
	//   - Apply priority-based policies (evict lower priorities first)
	//   - Use LIFO ordering (evict most recently scheduled first)
	//   - Implement time-based windows (only evict if gated for > N seconds)
	//   - Compare against usage limits (evict items further below their limit first)
	//
	// Parameters:
	//   - ctx: Request context for logging, tracing
	//   - candidate: The gated item to potentially evict
	//   - queue: The queue containing the item (provides Remove capability)
	//   - priority: The priority level of the item
	//   - usageLimit: The usage limit that gated this item
	ScheduleEvictionCandidate(ctx context.Context, candidate QueueItemAccessor, queue EvictableQueue, priority int, usageLimit float64)

	// ProcessScheduled removes scheduled eviction candidates from their queues.
	//
	// This method is called at the end of each dispatch cycle to process items scheduled via ScheduleEviction.
	// The evictor decides which candidates to actually remove based on its internal policy and current state.
	//
	// The implementation should:
	//  1. Evaluate scheduled candidates (check if eviction is still needed)
	//  2. Remove selected candidates from their queues using queue.Remove(handle) or queue.Cleanup(predicate)
	//  3. Clear the schedule for the next cycle
	//  4. Return the count of items actually evicted
	//
	// Parameters:
	//   - ctx: Request context for logging, tracing, cancellation
	//
	// Returns:
	//   - evicted: The number of items successfully removed from queues
	//   - err: Only returned for unrecoverable internal errors. Returning 0 evicted items is not an error.
	//
	// Example:
	//   During dispatch cycle: 50 items at priority -10 are gated, scheduled via ScheduleEvictionCandidate.
	//   At end of cycle: ProcessScheduled checks saturation. Still high, so evicts 30 items (LIFO).
	//   Returns evicted=30.
	ProcessScheduled(ctx context.Context) (evicted int, err error)
}
