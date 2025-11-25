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

// Package besthead provides a `framework.InterFlowDispatchPolicy` that selects the queue containing the single "best"
// item from across all queues in a priority band.
package interflow

import (
	"fmt"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// BestHeadPolicyName is the name of the Best Head policy implementation.
const BestHeadPolicyName = "BestHead"

func init() {
	MustRegisterPolicy(RegisteredPolicyName(BestHeadPolicyName),
		func() (framework.InterFlowDispatchPolicy, error) {
			return newBestHead(), nil
		})
}

type bestHead struct{}

func newBestHead() *bestHead {
	return &bestHead{}
}

// Name returns the name of the policy.
func (p *bestHead) Name() string {
	return BestHeadPolicyName
}

// SelectQueue implements a greedy strategy that bypasses fairness concerns to select the queue containing the single
// "best" item from across all queues in the priority band. It iterates through all non-empty queues, peeks at their
// head items, and uses the `framework.ItemComparator` from each queue to find the highest-priority item overall.
//
// This policy is useful for maximizing utilization when fairness between flows is not a concern. It requires that all
// queues being compared have a compatible `framework.ScoreType` to ensure the comparison is meaningful. If an
// incompatible comparator is found, the selection fails with an error.
func (p *bestHead) SelectQueue(band framework.PriorityBandAccessor) (framework.FlowQueueAccessor, error) {
	if band == nil {
		return nil, nil
	}

	var bestQueue framework.FlowQueueAccessor
	var bestItem types.QueueItemAccessor

	var iterationErr error
	band.IterateQueues(func(queue framework.FlowQueueAccessor) (keepIterating bool) {
		if queue == nil || queue.Len() == 0 {
			return true
		}

		item := queue.PeekHead()
		if item == nil {
			return true
		}

		if bestQueue == nil {
			bestQueue = queue
			bestItem = item
			return true
		}

		if queue.Comparator().ScoreType() != bestQueue.Comparator().ScoreType() {
			iterationErr = fmt.Errorf("%w: expected %q, got %q", framework.ErrIncompatiblePriorityType,
				bestQueue.Comparator().ScoreType(), queue.Comparator().ScoreType())
			return false
		}

		if bestQueue.Comparator().Func()(item, bestItem) {
			bestQueue = queue
			bestItem = item
		}
		return true
	})

	if iterationErr != nil {
		return nil, iterationErr
	}
	return bestQueue, nil
}
