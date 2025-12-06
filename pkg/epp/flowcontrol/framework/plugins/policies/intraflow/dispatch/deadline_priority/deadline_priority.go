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

package deadlinepriority

import (
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// DeadlinePriorityPolicyName is the name of the deadline-based priority policy implementation.
//
// This policy implements a deadline-urgency scheduling strategy by selecting the request with the earliest absolute
// deadline, computed as `EnqueueTime() + EffectiveTTL()`. Requests without a valid TTL (i.e., EffectiveTTL <= 0) are
// treated as having no deadline and are scheduled after all time-bound requests, using FCFS as a tie-breaker for fairness.
//
// # Behavior and Queue Pairing
//
// The correctness and performance of this policy are tightly coupled to the capabilities of the underlying
// `framework.SafeQueue`:
//   - When paired with a `CapabilityPriorityConfigurable` queue (e.g., a heap-based priority queue), the policy provides
//     strict deadline-ordered dispatch. The queue uses the policy's vended `ItemComparator` to maintain items in
//     urgency-sorted order, ensuring that `PeekHead()` always returns the most urgent request.
//   - This policy **MUST NOT** be used with a `CapabilityFIFO` queue (e.g., "ListQueue"), as such queues do not respect
//     custom comparators. In that case, `PeekHead()` would return the physically first enqueued item, completely
//     ignoring deadlines and violating the policy's semantics.
//
// To enforce correct behavior, this policy explicitly requires `CapabilityPriorityConfigurable` via its
// `RequiredQueueCapabilities()` method. The system will reject any configuration that attempts to bind this policy to
// an incompatible queue.
const DeadlinePriorityPolicyName = "DeadlinePriority"

func init() {
	dispatch.MustRegisterPolicy(dispatch.RegisteredPolicyName(DeadlinePriorityPolicyName),
		func() (framework.IntraFlowDispatchPolicy, error) {
			return newDeadlinePriorityPolicy(), nil
		})
}

// DeadlinePriorityPolicy implements an intra-flow dispatch policy that prioritizes
// requests based on their deadline urgency: the closer the absolute deadline, the higher the priority.
// See the documentation for the exported `DeadlinePriorityPolicyName` constant for detailed behavioral guarantees.
type DeadlinePriorityPolicy struct {
	comparator framework.ItemComparator
}

var _ framework.IntraFlowDispatchPolicy = &DeadlinePriorityPolicy{}

func newDeadlinePriorityPolicy() framework.IntraFlowDispatchPolicy {
	return &DeadlinePriorityPolicy{
		comparator: &deadlinePriorityComparator{},
	}
}

func (p *DeadlinePriorityPolicy) Name() string {
	return DeadlinePriorityPolicyName
}

// RequiredQueueCapabilities returns an empty slice, indicating that this policy can operate with any queue.
func (p *DeadlinePriorityPolicy) RequiredQueueCapabilities() []framework.QueueCapability {
	return []framework.QueueCapability{framework.CapabilityPriorityConfigurable}
}

func (p *DeadlinePriorityPolicy) Comparator() framework.ItemComparator {
	return p.comparator
}

// SelectItem selects the next item to dispatch by returning the head of the queue.
// This implementation assumes the underlying queue is ordered according to the policy's comparator
// (enforced by RequiredQueueCapabilities). Therefore, the most urgent request is always at the head.
// Returns (nil, nil) if the queue is empty or nil.
func (p *DeadlinePriorityPolicy) SelectItem(queue framework.FlowQueueAccessor) (selectedItem types.QueueItemAccessor, err error) {
	if queue == nil {
		return nil, nil
	}
	return queue.PeekHead(), nil
}

var maxDeadlineTime = time.Unix(1<<63-60, 0)

// calculateDeadline computes the absolute deadline for a request.
// The deadline is defined as the logical enqueue time plus the effective time-to-live (TTL).
// If EffectiveTTL is zero or negative, the request is considered non-time-sensitive and assigned a
// far-future deadline so it sorts after all SLO-bound requests.
func calculateDeadline(item types.QueueItemAccessor) time.Time {
	ttl := item.EffectiveTTL()
	if ttl <= 0 {
		// No TTL, Treat as "never expire", but still respect enqueue time for fairness.
		// Return max time so it sorts last.
		return maxDeadlineTime
	}
	return item.EnqueueTime().Add(ttl)
}

type deadlinePriorityComparator struct{}

func (d *deadlinePriorityComparator) Func() framework.ItemComparatorFunc {
	return func(a, b types.QueueItemAccessor) bool {
		deadlineA := calculateDeadline(a)
		deadlineB := calculateDeadline(b)

		if !deadlineA.Equal(deadlineB) {
			return deadlineA.Before(deadlineB) // earlier deadline = higher priority
		}

		// Same deadline: FCFS (earlier enqueue time = higher priority)
		return a.EnqueueTime().Before(b.EnqueueTime())
	}
}

// ScoreType indicates this policy uses deadline-based scoring.
func (d *deadlinePriorityComparator) ScoreType() string {
	return string(framework.DeadlineUrgencyPriorityScoreType)
}
