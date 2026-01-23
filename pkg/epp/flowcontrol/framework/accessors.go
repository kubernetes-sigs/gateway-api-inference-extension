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

package framework

import "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"

// FlowQueueAccessor represents the runtime state of a single active Flow.
//
// Role in Fairness:
// To a Fairness Policy, this interface represents a "Candidate": a distinct stream of requests that is competing for
// dispatch. The policy may inspect the state of this object (e.g., how many requests are waiting, how long they have
// been waiting, etc.) to decide if it should be the "Winner". Alternatively, it may operate on orthogonal state tracked
// for each FlowKey.
//
// Conformance: Implementations MUST ensure all methods are goroutine-safe.
type FlowQueueAccessor interface {
	QueueInspectionMethods

	// Comparator returns the ItemComparator that defines the ordering logic of the items within this queue.
	// This is determined by the IntraFlowDispatchPolicy associated with this queue's flow.
	Comparator() ItemComparator

	// FlowKey returns the unique, immutable identity of the flow instance this queue belongs to.
	// This provides essential context (like the logical grouping ID and Priority) to policies.
	FlowKey() types.FlowKey
}

// PriorityBandAccessor represents a Priority Band (conceptually, a 'Flow Group')—a collection of flows contending for
// resources at the same priority level.
//
// In the Endpoint Picker's hierarchy:
//  1. Incoming requests are mapped to a FlowKey.
//  2. All active flows sharing the same FlowKey.Priority are managed within this specific "Flow Group".
//  3. A FairnessPolicy is attached to this group to decide which flow within it gets the next dispatch attempt.
//
// This interface provides the read-only view required by the FairnessPolicy to inspect candidates and the state access
// required to implement the Flyweight pattern.
//
// Conformance: Implementations MUST ensure all methods are goroutine-safe.
type PriorityBandAccessor interface {
	// Priority returns the numerical priority level of this group.
	Priority() int

	// FlowKeys returns the list of identities for every flow currently active within this group.
	// The caller can use the ID field from each key to look up a specific queue via the Queue(id) method.
	// The order of keys is not guaranteed unless specified by the implementation.
	FlowKeys() []types.FlowKey

	// Queue returns the specific Flow (queue) associated with the given logical ID within this group.
	// Returns nil if the ID is not found in this group.
	Queue(id string) FlowQueueAccessor

	// IterateQueues executes the given callback for each active Flow in this group.
	// Iteration stops if the callback returns false. The order of iteration is not guaranteed unless specified by
	// the implementation.
	IterateQueues(callback func(flow FlowQueueAccessor) (keepIterating bool))

	// PolicyState retrieves the opaque, mutable state object associated with the active FairnessPolicy for this
	// group.
	//
	// This method allows the Stateless FairnessPolicy plugin (Singleton) to access its Stateful data (Scoped).
	// For example, a Round Robin policy uses this to retrieve the cursor indicating which flow was served last
	// in this specific group.
	//
	// Returns:
	//   - any: The opaque state object. This is guaranteed to be the exact object instance returned by the
	//     policy's NewState() method during initialization.
	PolicyState() any
}
