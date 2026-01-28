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

package ordering

import (
	"encoding/json"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// EDFOrderingPolicyType represents an ordering policy that implements a Earliest Deadline First (EDF) strategy.
//
// It selects the request with the earliest absolute deadline, computed as `EnqueueTime() + EffectiveTTL()`.
// Requests without a valid TTL (i.e., EffectiveTTL <= 0) are treated as having no deadline and are scheduled after all
// time-bound requests, using FCFS as a tie-breaker for fairness.
const EDFOrderingPolicyType = "edf-ordering-policy"

func init() {
	plugin.Register(EDFOrderingPolicyType, func(string, json.RawMessage, plugin.Handle) (plugin.Plugin, error) {
		return newEDFPolicy(), nil
	})
}

// EDFPolicy implements an OrderingPolicy based on the Earliest Deadline First (EDF) scheduling algorithm.
// Requests with earlier absolute deadlines (EnqueueTime + EffectiveTTL) are dispatched first.
// See the documentation for the exported EDFOrderingPolicyType constant for detailed behavioral guarantees.
type EDFPolicy struct{}

var _ flowcontrol.OrderingPolicy = &EDFPolicy{}

func newEDFPolicy() *EDFPolicy {
	return &EDFPolicy{}
}

func (p *EDFPolicy) Name() string {
	return EDFOrderingPolicyType
}

// RequiredQueueCapabilities returns the queue capabilities required by this policy.
// It requires a priority-configurable queue (e.g., heap-based) to maintain items in deadline-sorted order.
func (p *EDFPolicy) RequiredQueueCapabilities() []flowcontrol.QueueCapability {
	return []flowcontrol.QueueCapability{flowcontrol.CapabilityPriorityConfigurable}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *EDFPolicy) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: EDFOrderingPolicyType,
		Name: EDFOrderingPolicyType,
	}
}

var maxDeadlineTime = time.Unix(0, 1<<63-1)

// calculateDeadline computes the absolute deadline for a request.
// The deadline is defined as the logical enqueue time plus the effective time-to-live (TTL).
// If EffectiveTTL is zero or negative, the request is considered non-time-sensitive and assigned a
// far-future deadline so it sorts after all SLO-bound requests.
func calculateDeadline(item flowcontrol.QueueItemAccessor) time.Time {
	ttl := item.EffectiveTTL()
	if ttl <= 0 {
		// No TTL: treat as "never expire", but still respect enqueue time for fairness.
		return maxDeadlineTime
	}
	return item.EnqueueTime().Add(ttl)
}

// Less returns true if item 'a' should be dispatched before item 'b'.
// EDF orders by deadline (earliest first), using FCFS as a tie-breaker.
func (p *EDFPolicy) Less(a, b flowcontrol.QueueItemAccessor) bool {
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
