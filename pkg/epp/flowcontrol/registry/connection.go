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

package registry

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// connection is the concrete, un-exported implementation of the `contracts.ActiveFlowConnection` interface.
// It is a temporary handle created for the duration of a single `WithConnection` call.
type connection struct {
	registry *FlowRegistry
	key      types.FlowKey
}

var _ contracts.ActiveFlowConnection = &connection{}

// Shards returns a stable snapshot of accessors for all internal state shards.
func (c *connection) ActiveShards() []contracts.RegistryShard {
	c.registry.mu.RLock()
	defer c.registry.mu.RUnlock()

	// Return a copy to ensure the caller cannot modify the registry's internal slice.
	shardsCopy := make([]contracts.RegistryShard, len(c.registry.activeShards))
	for i, s := range c.registry.activeShards {
		shardsCopy[i] = s
	}
	return shardsCopy
}
