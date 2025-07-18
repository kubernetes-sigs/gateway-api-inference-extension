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
package fcfs

import (
	"errors"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// FCFSPolicyName is the name of the FCFS policy implementation.
const FCFSPolicyName = "FCFS"

func init() {
	dispatch.MustRegisterPolicy(dispatch.RegisteredPolicyName(FCFSPolicyName),
		func() (framework.IntraFlowDispatchPolicy, error) {
			return newFCFS(), nil
		})
}

// fcfs (First-Come, First-Served) implements the `framework.IntraFlowDispatchPolicy` interface.
type fcfs struct {
	comparator framework.ItemComparator
}

// newFCFS creates a new `fcfs` policy instance.
func newFCFS() *fcfs {
	return &fcfs{
		comparator: &enqueueTimeComparator{},
	}
}

// Name returns the name of the policy.
func (p *fcfs) Name() string {
	return FCFSPolicyName
}

// SelectItem selects the next item from the queue by peeking its head. This implementation relies on the queue being
// ordered by dispatch preference, as indicated by its `RequiredQueueCapabilities`.
func (p *fcfs) SelectItem(queue framework.FlowQueueAccessor) (types.QueueItemAccessor, error) {
	if queue == nil {
		return nil, nil
	}
	item, err := queue.PeekHead()
	if errors.Is(err, framework.ErrQueueEmpty) {
		return nil, nil
	}
	return item, err
}

// Comparator returns a `framework.ItemComparator` based on enqueue time.
func (p *fcfs) Comparator() framework.ItemComparator {
	return p.comparator
}

// RequiredQueueCapabilities specifies that this policy needs a queue that supports FIFO operations.
func (p *fcfs) RequiredQueueCapabilities() []framework.QueueCapability {
	return []framework.QueueCapability{framework.CapabilityFIFO}
}

// --- enqueueTimeComparator ---

// enqueueTimeComparator implements `framework.ItemComparator` for FCFS logic.
// It prioritizes items with earlier enqueue times.
type enqueueTimeComparator struct{}

// Func returns the comparison logic.
// It returns true if item 'a' should be dispatched before item 'b'.
func (c *enqueueTimeComparator) Func() framework.ItemComparatorFunc {
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
		return a.EnqueueTime().Before(b.EnqueueTime())
	}
}

// ScoreType returns a string descriptor for the comparison logic.
func (c *enqueueTimeComparator) ScoreType() string {
	return string(framework.EnqueueTimePriorityScoreType)
}
