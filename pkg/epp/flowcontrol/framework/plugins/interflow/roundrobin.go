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

// Package roundrobin provides a `framework.InterFlowDispatchPolicy` that selects a queue from a priority band using a
// simple round-robin strategy.
package interflow

import (
	"slices"
	"sync"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// RoundRobinPolicyName is the name of the Round Robin policy implementation.
const RoundRobinPolicyName = "RoundRobin"

func init() {
	MustRegisterPolicy(RegisteredPolicyName(RoundRobinPolicyName),
		func() (framework.InterFlowDispatchPolicy, error) {
			return newRoundRobin(), nil
		})
}

// roundRobin implements the `framework.InterFlowDispatchPolicy` interface using a round-robin strategy.
type roundRobin struct {
	iterator *iterator
}

func newRoundRobin() framework.InterFlowDispatchPolicy {
	return &roundRobin{
		iterator: newIterator(),
	}
}

// Name returns the name of the policy.
func (p *roundRobin) Name() string {
	return RoundRobinPolicyName
}

// SelectQueue selects the next flow queue in a round-robin fashion from the given priority band.
// It returns nil if all queues in the band are empty or if an error occurs.
func (p *roundRobin) SelectQueue(band framework.PriorityBandAccessor) (framework.FlowQueueAccessor, error) {
	if band == nil {
		return nil, nil
	}
	selectedQueue := p.iterator.selectNextQueue(band)
	return selectedQueue, nil
}

// iterator implements a thread-safe round-robin selection logic. It maintains the ID of the last selected flow to
// ensure the selection order is correct even when the set of available flows changes dynamically.
//
// This is kept as a private, nested type as its logic is specific to this policy. This structure is a deliberate
// choice for future refactoring; the iterator logic can be easily extracted into a shared internal package if a
// "RoundRobin" displacement policy is introduced, while keeping the dispatch policy's public API stable.
type iterator struct {
	mu           sync.Mutex
	lastSelected *types.FlowKey
}

// newIterator creates a new round-robin Iterator.
func newIterator() *iterator {
	return &iterator{}
}

// selectNextQueue iterates through the flow queues in a round-robin fashion, starting from the flow after the one
// selected in the previous call. It sorts the flow IDs to ensure a deterministic ordering. If no non-empty queue is
// found after a full cycle, it returns nil.
func (r *iterator) selectNextQueue(band framework.PriorityBandAccessor) framework.FlowQueueAccessor {
	r.mu.Lock()
	defer r.mu.Unlock()

	keys := band.FlowKeys()
	if len(keys) == 0 {
		r.lastSelected = nil // Reset state if no flows are present
		return nil
	}
	// Sort for deterministic ordering.
	slices.SortFunc(keys, func(a, b types.FlowKey) int { return a.Compare(b) })

	startIndex := 0
	if r.lastSelected != nil {
		// Find the index of the last selected flow.
		// If it's not found (e.g., the flow was removed), we'll start from the beginning.
		if idx := slices.Index(keys, *r.lastSelected); idx != -1 {
			startIndex = (idx + 1) % len(keys)
		}
	}

	numFlows := len(keys)
	for i := range numFlows {
		currentIdx := (startIndex + i) % numFlows
		currentKey := keys[currentIdx]
		queue := band.Queue(currentKey.ID)
		if queue != nil && queue.Len() > 0 {
			r.lastSelected = &currentKey
			return queue
		}
	}

	// No non-empty queue was found.
	r.lastSelected = nil
	return nil
}
