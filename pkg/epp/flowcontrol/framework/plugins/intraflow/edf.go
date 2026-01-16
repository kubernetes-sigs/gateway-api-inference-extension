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

package intraflow

import (
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// EDFPolicyName is the name of the Earliest Deadline First (EDF) intra-flow dispatch policy.
//
// This policy implements a deadline-urgency scheduling strategy by selecting the request with the earliest absolute
// deadline, computed as `EnqueueTime() + EffectiveTTL()`. Requests without a valid TTL (i.e., EffectiveTTL <= 0) are
// treated as having no deadline and are scheduled after all time-bound requests, using FCFS as a tie-breaker for fairness.
const EDFPolicyName = "EDF"

func init() {
	MustRegisterPolicy(RegisteredPolicyName(EDFPolicyName),
		func() (framework.IntraFlowDispatchPolicy, error) {
			return newEDFPolicy(), nil
		})
}

// EDFPolicy implements an intra-flow dispatch policy based on the Earliest Deadline First (EDF) scheduling algorithm.
// Requests with earlier absolute deadlines (EnqueueTime + EffectiveTTL) are dispatched first.
// See the documentation for the exported `EDFPolicyName` constant for detailed behavioral guarantees.
type EDFPolicy struct {
	comparator framework.ItemComparator
}

var _ framework.IntraFlowDispatchPolicy = &EDFPolicy{}

func newEDFPolicy() framework.IntraFlowDispatchPolicy {
	return &EDFPolicy{
		comparator: &edfComparator{},
	}
}

func (p *EDFPolicy) Name() string {
	return EDFPolicyName
}

// RequiredQueueCapabilities returns the queue capabilities required by this policy.
// It requires a priority-configurable queue (e.g., heap-based) to maintain items in deadline-sorted order.
func (p *EDFPolicy) RequiredQueueCapabilities() []framework.QueueCapability {
	return []framework.QueueCapability{framework.CapabilityPriorityConfigurable}
}

func (p *EDFPolicy) Comparator() framework.ItemComparator {
	return p.comparator
}

// SelectItem selects the next item to dispatch by returning the head of the queue.
// This assumes the underlying queue is ordered according to the policy's comparator
// (enforced via RequiredQueueCapabilities). Thus, the most urgent request is always at the head.
// Returns (nil, nil) if the queue is empty or nil.
func (p *EDFPolicy) SelectItem(queue framework.FlowQueueAccessor) (selectedItem types.QueueItemAccessor, err error) {
	if queue == nil {
		return nil, nil
	}
	return queue.PeekHead(), nil
}

var maxDeadlineTime = time.Unix(0, 1<<63-1)

// calculateDeadline computes the absolute deadline for a request.
// The deadline is defined as the logical enqueue time plus the effective time-to-live (TTL).
// If EffectiveTTL is zero or negative, the request is considered non-time-sensitive and assigned a
// far-future deadline so it sorts after all SLO-bound requests.
func calculateDeadline(item types.QueueItemAccessor) time.Time {
	ttl := item.EffectiveTTL()
	if ttl <= 0 {
		// No TTL: treat as "never expire", but still respect enqueue time for fairness.
		return maxDeadlineTime
	}
	return item.EnqueueTime().Add(ttl)
}

type edfComparator struct{}

func (d *edfComparator) Func() framework.ItemComparatorFunc {
	return func(a, b types.QueueItemAccessor) bool {
		if a == nil && b == nil {
			return false
		}
		if a == nil { // Treat nil as lowest priority
			return false
		}
		if b == nil { // Treat non-nil 'a' as higher priority than nil 'b'
			return true
		}
		deadlineA := calculateDeadline(a)
		deadlineB := calculateDeadline(b)

		if !deadlineA.Equal(deadlineB) {
			return deadlineA.Before(deadlineB) // earlier deadline = higher priority
		}

		// Same deadline: FCFS (earlier enqueue time = higher priority)
		return a.EnqueueTime().Before(b.EnqueueTime())
	}
}

// ScoreType indicates this policy uses EDF-based scoring.
func (d *edfComparator) ScoreType() string {
	return string(framework.EDFPriorityScoreType)
}
