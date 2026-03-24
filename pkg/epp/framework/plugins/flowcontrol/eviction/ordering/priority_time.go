/*
Copyright 2026 The Kubernetes Authors.

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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// PriorityThenTimeOrderingType evicts the lowest priority request first,
// breaking ties by newest dispatch time (least KV-cache investment).
const PriorityThenTimeOrderingType = "eviction-priority-then-time-ordering"

func init() {
	plugin.Register(PriorityThenTimeOrderingType, PriorityThenTimeOrderingFactory)
}

// PriorityThenTimeOrderingFactory creates a PriorityThenTimeOrdering plugin.
func PriorityThenTimeOrderingFactory(name string, _ json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	p := &PriorityThenTimeOrdering{name: PriorityThenTimeOrderingType}
	if name != "" {
		p.name = name
	}
	return p, nil
}

// PriorityThenTimeOrdering evicts the lowest priority request first.
// When priorities are equal, the newest dispatched request is evicted first
// to minimize wasted KV-cache investment.
type PriorityThenTimeOrdering struct {
	name string
}

var _ flowcontrol.EvictionOrderingPolicy = &PriorityThenTimeOrdering{}

func (p *PriorityThenTimeOrdering) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: PriorityThenTimeOrderingType, Name: p.name}
}

func (p *PriorityThenTimeOrdering) Less(a, b *flowcontrol.EvictionItem) bool {
	if a.Priority != b.Priority {
		return a.Priority < b.Priority
	}
	return a.DispatchTime.After(b.DispatchTime)
}
