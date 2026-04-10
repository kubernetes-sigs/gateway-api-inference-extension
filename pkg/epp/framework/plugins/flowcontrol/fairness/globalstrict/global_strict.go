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

// Package globalstrict implements a greedy fairness policy that enforces a strict global ordering
// across all flows within a priority band, ignoring flow boundaries.
//
// For detailed documentation, see README.md.
package globalstrict

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// GlobalStrictFairnessPolicyType is the registration type for the global strict fairness policy.
const GlobalStrictFairnessPolicyType = "global-strict-fairness-policy"

// GlobalStrictFairnessPolicyFactory is the factory function for the global strict fairness policy.
func GlobalStrictFairnessPolicyFactory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return newGlobalStrict(name), nil
}

// globalStrict implements the FairnessPolicy interface, selecting the queue
// containing the single "best" item from across all queues in a priority band.
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

// Pick executes the global strict strategy by iterating over every active flow in the band
// and inspecting the head of each queue to find the single highest-priority item.
//
// Requirements:
// All flows in the band MUST use compatible OrderingPolicy types (i.e., identical score types).
// If incompatible policies are detected, an error is returned.
func (p *globalStrict) Pick(
	_ context.Context,
	flowGroup flowcontrol.PriorityBandAccessor,
) (flowcontrol.FlowQueueAccessor, error) {
	if flowGroup == nil {
		return nil, nil
	}

	var bestQueue flowcontrol.FlowQueueAccessor
	var bestItem flowcontrol.QueueItemAccessor
	var iterationErr error

	flowGroup.IterateQueues(func(queue flowcontrol.FlowQueueAccessor) (keepIterating bool) {
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
			iterationErr = fmt.Errorf("%w: expected %q, got %q", flowcontrol.ErrIncompatiblePriorityType,
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
