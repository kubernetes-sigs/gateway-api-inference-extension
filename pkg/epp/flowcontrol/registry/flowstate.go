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

import "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"

// flowState holds all tracking state for a single logical flow within the registry.
//
// # Role: The Eventually Consistent Cache for GC
//
// This structure is central to the garbage collection (GC) logic. It acts as an eventually consistent, cached view of a
// flow's emptiness status across all shards. It is updated asynchronously via events from the data path and is used by
// the `FlowRegistry`'s control plane to make preliminary, non-destructive decisions about a flow's lifecycle (e.g.,
// "the flow appears to be idle, start a GC timer").
//
// # Concurrency and Consistency Model
//
// `flowState` is a passive, non-thread-safe data structure. It is owned and managed exclusively by the `FlowRegistry`'s
// single-threaded event loop.
//
// CRITICAL: Because this state is eventually consistent, it MUST NOT be the sole source of truth for any destructive
// operation (like garbage collection). All destructive actions must first verify the live state of the system by
// consulting the atomic counters on the `managedQueue` instances directly. This is part of the "Trust but Verify"
// pattern.
type flowState struct {
	// spec is the desired state of the flow.
	spec types.FlowSpecification
	// generation is an internal counter to resolve races between GC timers and flow re-registration.
	// This ensures that a stale timer event from a previous configuration does not incorrectly garbage collect the flow.
	generation uint64
	// activeQueueEmptyOnShards tracks the empty status of the flow's single active queue across all shards.
	// The key is the shard ID.
	activeQueueEmptyOnShards map[string]bool
	// drainingQueuesEmptyOnShards tracks the empty status of any draining queues for this flow.
	// The key is the priority level of the draining queue, then the shard ID.
	drainingQueuesEmptyOnShards map[uint]map[string]bool
}

// newFlowState creates the initial state for a newly registered flow.
// It initializes the state based on the current set of active and draining shards.
func newFlowState(spec types.FlowSpecification, allShards []*registryShard) *flowState {
	s := &flowState{
		spec:                     spec,
		generation:               1,
		activeQueueEmptyOnShards: make(map[string]bool, len(allShards)),
	}
	// New queues start empty on all shards (active and draining).
	for _, shard := range allShards {
		s.activeQueueEmptyOnShards[shard.id] = true
	}
	return s
}

// update applies a new specification to the flow, incrementing its generation and handling priority change logic.
func (s *flowState) update(spec types.FlowSpecification, allShards []*registryShard) {
	oldPriority := s.spec.Priority
	s.spec = spec
	s.generation++ // Invalidate any pending GC timers for the old generation.

	// If priority did not change, there's nothing more to do.
	if oldPriority == spec.Priority {
		return
	}

	// Priority changed. The old active queue is now a draining queue.
	// We transfer its current emptiness state from the active map to the draining map.
	if s.drainingQueuesEmptyOnShards == nil {
		s.drainingQueuesEmptyOnShards = make(map[uint]map[string]bool)
	}
	s.drainingQueuesEmptyOnShards[oldPriority] = s.activeQueueEmptyOnShards

	// After an update, the new active queue must be re-evaluated.
	// Check if we are reactivating a previously draining queue.
	if drainingState, ok := s.drainingQueuesEmptyOnShards[spec.Priority]; ok {
		// Yes, we are reactivating. The draining state becomes the new active state.
		s.activeQueueEmptyOnShards = drainingState
		delete(s.drainingQueuesEmptyOnShards, spec.Priority)
	} else {
		// No, this is a new priority. It starts as empty on all shards.
		s.activeQueueEmptyOnShards = make(map[string]bool, len(allShards))
		for _, shard := range allShards {
			s.activeQueueEmptyOnShards[shard.id] = true
		}
	}
}

// handleQueueSignal updates the flow's internal state based on a signal from one of its queues.
func (s *flowState) handleQueueSignal(shardID string, priority uint, signal queueStateSignal) {
	switch signal {
	case queueStateSignalBecameDrained:
		if priorityState, ok := s.drainingQueuesEmptyOnShards[priority]; ok {
			priorityState[shardID] = true
		}
	case queueStateSignalBecameEmpty:
		s.activeQueueEmptyOnShards[shardID] = true
	case queueStateSignalBecameNonEmpty:
		s.activeQueueEmptyOnShards[shardID] = false
	}
}

// isIdle checks if the flow's active queues are empty across all active shards.
// A flow is considered idle even if it has items remaining in a shard that is itself draining.
func (s *flowState) isIdle(activeShards []*registryShard) bool {
	for _, shard := range activeShards {
		// We rely on the caller (FlowRegistry) to provide only the active shards.
		if !s.activeQueueEmptyOnShards[shard.id] {
			return false
		}
	}
	return true
}

// isDrained checks if a specific draining queue is now empty on all shards (active and draining).
func (s *flowState) isDrained(priority uint, allShards []*registryShard) bool {
	priorityState, ok := s.drainingQueuesEmptyOnShards[priority]
	if !ok {
		// If the priority isn't in the map, it's not currently draining, so it cannot be complete.
		return false
	}

	// We must check both active and draining shards, as the queue instance exists on both until GC'd.
	for _, shard := range allShards {
		if !priorityState[shard.id] {
			return false
		}
	}
	return true
}

// purgeShard removes a decommissioned shard's ID from all tracking maps to prevent memory leaks.
func (s *flowState) purgeShard(shardID string) {
	delete(s.activeQueueEmptyOnShards, shardID)
	for _, priorityState := range s.drainingQueuesEmptyOnShards {
		delete(priorityState, shardID)
	}
}
