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
	"fmt"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// flowState holds all tracking state for a single flow instance within the registry.
//
// # Role: The Eventually Consistent Cache for GC
//
// This structure is central to the GC logic. It acts as an eventually consistent, cached view of a flow's emptiness
// status across all shards. It is updated asynchronously via events from the data path.
//
// # Concurrency and Consistency Model
//
// `flowState` is a passive, non-thread-safe data structure. Access to `flowState` is serialized by `FlowRegistry.mu`.
//
// Invariant: Because this state is eventually consistent, it must not be the sole source of truth for destructive
// operations (like garbage collection). All destructive actions must first verify the live state of the system by
// consulting the atomic counters on the `managedQueue` instances directly (the "Trust but Verify" pattern).
type flowState struct {
	// spec is the desired state of the flow instance, uniquely identified by its immutable `spec.Key`.
	spec types.FlowSpecification
	// emptyOnShards tracks the empty status of the flow's queue across all shards.
	// The key is the shard ID.
	emptyOnShards map[string]bool
	// lastActiveGeneration tracks the GC generation number in which this flow was last observed to be Active.
	// This is used by the periodic GC scanner to identify flows that have been Idle for at least one GC cycle.
	lastActiveGeneration uint64
}

// newFlowState creates the initial state for a newly registered flow instance.
// It initializes the state based on the current set of all shards.
func newFlowState(spec types.FlowSpecification, allShards []*registryShard) *flowState {
	s := &flowState{
		spec:          spec,
		emptyOnShards: make(map[string]bool, len(allShards)),
		// A new flow starts at generation 0. This sentinel value indicates to the GC "Mark" phase that the flow is brand
		// new and should be granted a grace period of one cycle.
		lastActiveGeneration: gcGenerationNewFlow,
	}
	// A new flow instance starts with an empty queue on all shards.
	for _, shard := range allShards {
		s.emptyOnShards[shard.id] = true
	}
	return s
}

// update synchronizes the flow's specification. This is a forward-looking method for when non-key fields on the spec
// become mutable (e.g., a per-flow policy override). It enforces the invariant that the `FlowKey` cannot change.
func (s *flowState) update(newSpec types.FlowSpecification) {
	if s.spec.Key != newSpec.Key {
		// This should be impossible if the FlowRegistry's logic is correct, as flowStates are keyed by FlowKey.
		panic(fmt.Sprintf("invariant violation: attempted to update flowState for key %v with a new key %v",
			s.spec.Key, newSpec.Key))
	}
	s.spec = newSpec
}

// handleQueueSignal updates the flow's internal emptiness state based on a signal from one of its queues.
func (s *flowState) handleQueueSignal(shardID string, signal queueStateSignal) {
	switch signal {
	case queueStateSignalBecameEmpty:
		s.emptyOnShards[shardID] = true
	case queueStateSignalBecameNonEmpty:
		s.emptyOnShards[shardID] = false
	}
}

// isIdle checks if the flow's queues are empty across the provided set of shards (active and draining).
//
// The `shards` parameter defines the scope of the check (the current ground truth). We only verify the state against
// this list because the internal `emptyOnShards` map might contain stale entries for shards that have been garbage
// collected by the registry but not yet purged from this specific `flowState`.
func (s *flowState) isIdle(shards []*registryShard) bool {
	for _, shard := range shards {
		if !s.emptyOnShards[shard.id] {
			return false
		}
	}
	return true
}

// purgeShard removes a decommissioned shard's ID from the tracking map to prevent memory leaks.
func (s *flowState) purgeShard(shardID string) {
	delete(s.emptyOnShards, shardID)
}
