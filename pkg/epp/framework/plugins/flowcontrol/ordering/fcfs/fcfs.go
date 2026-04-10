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

// Package fcfs implements an ordering policy that selects requests based on their arrival order
// at the Flow Control layer (First-Come, First-Served).
//
// For detailed documentation, see README.md.
package fcfs

import (
	"encoding/json"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// FCFSOrderingPolicyType is the registration type for the FCFS ordering policy.
//
// It selects the item with the earliest logical enqueue time.
// For detailed documentation on behavior and queue pairing, see README.md.
const FCFSOrderingPolicyType = "fcfs-ordering-policy"

func FCFSOrderingPolicyFactory(name string, _ json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	return newFCFS().withName(name), nil
}

// fcfs is the internal implementation of the FCFS policy.
// See the documentation for the exported `FCFSPolicyName` constant for detailed user-facing information about its
// behavior.
type fcfs struct {
	name string
}

var _ flowcontrol.OrderingPolicy = &fcfs{}

// newFCFS creates a new `fcfs` policy instance.
func newFCFS() *fcfs {
	return &fcfs{
		name: FCFSOrderingPolicyType,
	}
}

func (p *fcfs) withName(name string) *fcfs {
	if name != "" {
		p.name = name
	}
	return p
}

// RequiredQueueCapabilities returns an empty slice, indicating that this policy can operate with any queue.
// See the `FCFSPolicyName` constant's documentation for details on the behavioral trade-offs.
func (p *fcfs) RequiredQueueCapabilities() []flowcontrol.QueueCapability {
	return []flowcontrol.QueueCapability{}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *fcfs) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: FCFSOrderingPolicyType,
		Name: p.name,
	}
}

// Less returns true if item 'a' should be dispatched before item 'b'.
// FCFS orders by logical enqueue time (earliest first).
func (p *fcfs) Less(a, b flowcontrol.QueueItemAccessor) bool {
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
