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
	"slices"
	"sync"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// RoundRobinFairnessPolicyType represents a fairness policy that selects a queue from a priority band using a simple
// round-robin strategy.
const RoundRobinFairnessPolicyType = "round-robin-fairness-policy"

func init() {
	fwkplugin.Register(
		RoundRobinFairnessPolicyType,
		func(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
			return newRoundRobin(name), nil
		},
	)
}

// roundRobin implements FairnessPolicy.
type roundRobin struct {
	name string
}

func newRoundRobin(name string) *roundRobin {
	if name == "" {
		name = RoundRobinFairnessPolicyType
	}
	return &roundRobin{name}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *roundRobin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{
		Type: RoundRobinFairnessPolicyType,
		Name: p.name,
	}
}

// roundRobinCursor holds the mutable cursor for a specific priority band.
// It is initialized via NewState and stored on the PriorityBandAccessor.
type roundRobinCursor struct {
	mu           sync.Mutex
	lastSelected *types.FlowKey
}

// NewState initializes the policy state for a specific priority band.
func (p *roundRobin) NewState(_ context.Context) any {
	return &roundRobinCursor{}
}

// Pick selects the next flow queue in a round-robin fashion from the given priority band.
// It retrieves the band-specific state, locks it, and advances the cursor.
func (p *roundRobin) Pick(
	_ context.Context,
	flowGroup framework.PriorityBandAccessor,
) (framework.FlowQueueAccessor, error) {
	if flowGroup == nil {
		return nil, nil
	}

	v := flowGroup.PolicyState()
	c, ok := v.(*roundRobinCursor)
	if !ok {
		return nil, fmt.Errorf("invalid state type for RoundRobin policy: expected *roundRobinCursor, got %T", v)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	keys := flowGroup.FlowKeys()
	if len(keys) == 0 {
		c.lastSelected = nil // Reset cursor if no flows are present.
		return nil, nil
	}

	// Sort for deterministic ordering.
	slices.SortFunc(keys, func(a, b types.FlowKey) int { return a.Compare(b) })

	startIndex := 0
	if c.lastSelected != nil {
		// Find the index of the last selected flow.
		// If it's not found (e.g., the flow was removed), we'll start from the beginning (index 0).
		if idx := slices.Index(keys, *c.lastSelected); idx != -1 {
			startIndex = (idx + 1) % len(keys)
		}
	}

	numFlows := len(keys)
	for i := range numFlows {
		currentIdx := (startIndex + i) % numFlows
		currentKey := keys[currentIdx]
		queue := flowGroup.Queue(currentKey.ID)
		if queue != nil && queue.Len() > 0 {
			c.lastSelected = &currentKey
			return queue, nil
		}
	}

	// No non-empty queue was found.
	c.lastSelected = nil
	return nil, nil
}
