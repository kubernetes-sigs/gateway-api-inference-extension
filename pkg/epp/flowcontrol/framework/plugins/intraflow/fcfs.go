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

// Package fcfs provides a First-Come, First-Served implementation of the `framework.IntraFlowDispatchPolicy`.
package intraflow

import (
	"encoding/json"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// FCFSOrderingPolicyType represents an ordering policy that implements a First-Come, First-Served (FCFS) strategy.
//
// It selects the item with the earliest logical enqueue time.
//
// # Behavior and Queue Pairing
//
// The behavioral guarantees of this policy are critically dependent on the capabilities of the `framework.SafeQueue` it
// is paired with. The system distinguishes between:
//   - "Logical Enqueue Time": The timestamp when a request first arrives at the `controller.FlowController`.
//   - "Physical Enqueue Time": The timestamp when a request is added to a specific shard's queue, which happens later.
//
// This policy's behavior changes accordingly:
//   - Paired with a `CapabilityPriorityConfigurable` queue, it provides strict FCFS ordering based on logical enqueue
//     time, aligning with this policy's vended `framework.ItemComparator`.
//     This configuration ensures that requests are processed in the order they arrived at the controller, providing the
//     most intuitive behavior.
//   - Paired with a `CapabilityFIFO` queue, it provides approximate FCFS ordering based on physical arrival order at
//     the `framework.SafeQueue`.
//     This configuration offers higher performance at the cost of strict logical-time ordering, as the
//     `controller.FlowController`'s "bounce-and-retry" mechanic for Draining shards means a bounced request may be
//     processed after a request that logically darrived later.
//
// Given that true end-to-end ordering is non-deterministic in a distributd system, this policy defaults to pairing with
// a CapabilityFIFO` queue (like "ListQueue") to prioritize performance and high throughput. For users who require the
// strictest possible logical-time ordering that this layer can provide, explicitly pairing this policy with a
// `CapabilityPriorityConfigurable` queue is recommended.
const FCFSOrderingPolicyType = "fcfs-ordering-policy"

func init() {
	// Legacy Registration for IntraFlowDispatchPolicy factory
	// TODO(kubernetes-sigs/gateway-api-inference-extension#1405): Remove once migration to EPP plugin model is complete.
	MustRegisterPolicy(RegisteredPolicyName(FCFSOrderingPolicyType),
		func() (framework.IntraFlowDispatchPolicy, error) {
			return newFCFS(), nil
		})

	plugin.Register(FCFSOrderingPolicyType, func(string, json.RawMessage, plugin.Handle) (plugin.Plugin, error) {
		return newFCFS(), nil
	})
}

// fcfs is the internal implementation of the FCFS policy.
// See the documentation for the exported `FCFSPolicyName` constant for detailed user-facing information about its
// behavior.
type fcfs struct {
	comparator framework.ItemComparator
}

// newFCFS creates a new `fcfs` policy instance.
func newFCFS() *fcfs {
	p := &fcfs{}
	p.comparator = &enqueueTimeComparator{policy: p}
	return p
}

// Name returns the name of the policy.
func (p *fcfs) Name() string {
	return FCFSOrderingPolicyType
}

// SelectItem selects the next item from the queue by peeking its head. This implementation relies on the queue being
// ordered by dispatch preference, as indicated by its `RequiredQueueCapabilities`.
func (p *fcfs) SelectItem(queue framework.FlowQueueAccessor) (types.QueueItemAccessor, error) {
	if queue == nil {
		return nil, nil
	}
	return queue.PeekHead(), nil
}

// Comparator returns a `framework.ItemComparator` based on enqueue time.
// TODO(kubernetes-sigs/gateway-api-inference-extension#1405): Remove once migration to EPP plugin model is complete.
func (p *fcfs) Comparator() framework.ItemComparator {
	return p.comparator
}

// RequiredQueueCapabilities returns an empty slice, indicating that this policy can operate with any queue.
// See the `FCFSPolicyName` constant's documentation for details on the behavioral trade-offs.
func (p *fcfs) RequiredQueueCapabilities() []framework.QueueCapability {
	return []framework.QueueCapability{}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *fcfs) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: FCFSOrderingPolicyType,
		Name: FCFSOrderingPolicyType,
	}
}

// --- enqueueTimeComparator ---

// enqueueTimeComparator implements `framework.ItemComparator` for FCFS logic.
// It prioritizes items with earlier enqueue times.
// TODO(kubernetes-sigs/gateway-api-inference-extension#1405): Remove once migration to EPP plugin model is complete.
type enqueueTimeComparator struct {
	policy framework.OrderingPolicy
}

// Func returns the comparison logic.
// It delegates to the OrderingPolicy.Less method.
// TODO(kubernetes-sigs/gateway-api-inference-extension#1405): Remove once migration to EPP plugin model is complete.
func (c *enqueueTimeComparator) Func() framework.ItemComparatorFunc {
	return func(a, b types.QueueItemAccessor) bool {
		return c.policy.Less(a, b)
	}
}

// Less returns true if item 'a' should be dispatched before item 'b'.
// FCFS orders by logical enqueue time (earliest first).
func (p *fcfs) Less(a, b types.QueueItemAccessor) bool {
	if a == nil && b == nil {
		return false
	}
	if a == nil { // Treat nil as lowest priority
		return false
	}
	if b == nil { // Treat non-nil 'a' as higher priority
		return true
	}
	return a.EnqueueTime().Before(b.EnqueueTime())
}

// ScoreType returns a string descriptor for the comparison logic.
// TODO(kubernetes-sigs/gateway-api-inference-extension#1405): Remove once migration to EPP plugin model is complete.
func (c *enqueueTimeComparator) ScoreType() string {
	return string(framework.EnqueueTimePriorityScoreType)
}
