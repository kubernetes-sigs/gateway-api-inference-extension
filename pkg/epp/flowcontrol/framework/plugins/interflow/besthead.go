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

package interflow

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// GlobalStrictFairnessPolicyType represents a fairness policy that enforces a strict global ordering across all flows.
// By ignoring flow boundaries and always selecting the absolute "best" item from any available queue, this policy
// effectively transforms the multi-queue Priority Band into a single logical Priority Queue.
// TODO: rename files to `global_strict.go` and `global_strict_test.go`.
const GlobalStrictFairnessPolicyType = "global-strict-fairness-policy"

func init() {
	fwkplugin.Register(GlobalStrictFairnessPolicyType, GlobalStrictFairnessPolicyFactory)
}

func GlobalStrictFairnessPolicyFactory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return newGlobalStrict(name), nil
}

// globalStrict implements FairnessPolicy.
// It selects the queue containing the single "best" item from across all queues in a priority band.
type globalStrict struct {
	name string
}

func newGlobalStrict(name string) *globalStrict {
	if name == "" {
		name = GlobalStrictFairnessPolicyType
	}
	return &globalStrict{name}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *globalStrict) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{
		Type: GlobalStrictFairnessPolicyType,
		Name: p.name,
	}
}

// NewState initializes the policy state for a specific priority band.
// globalStrict is purely greedy and stateless, so it returns nil.
func (p *globalStrict) NewState(_ context.Context) any {
	return nil
}

// Pick executes the global strict strategy.
//
// It iterates over every active flow in the band, inspecting the head of each queue to find the single highest-priority
// item currently waiting in the entire pool.
//
// Behavior:
//   - Fairness: Ignored. A single flow with a burst of high-priority traffic can starve others.
//   - Ordering: Strict. The system behaves as if all requests were in one global queue.
//
// Requirements:
// All flows in the band MUST use compatible ItemComparators (i.e., identical ScoreTypes).
// If incompatible comparators are detected (e.g., comparing "Timestamp" vs "UrgencyScore"), the method returns an error
// as a strict comparison is impossible.
func (p *globalStrict) Pick(
	_ context.Context,
	flowGroup framework.PriorityBandAccessor,
) (framework.FlowQueueAccessor, error) {
	if flowGroup == nil {
		return nil, nil
	}

	var bestQueue framework.FlowQueueAccessor
	var bestItem types.QueueItemAccessor
	var iterationErr error

	flowGroup.IterateQueues(func(queue framework.FlowQueueAccessor) (keepIterating bool) {
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

		if queue.OrderingPolicy().TypedName().Type != bestQueue.OrderingPolicy().TypedName().Type {
			iterationErr = fmt.Errorf("%w: expected %q, got %q", framework.ErrIncompatiblePriorityType,
				bestQueue.OrderingPolicy().TypedName().Type, queue.OrderingPolicy().TypedName().Type)
			return false
		}

		if bestQueue.OrderingPolicy().Less(item, bestItem) {
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
