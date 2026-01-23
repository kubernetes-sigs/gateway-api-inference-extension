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

import (
	"context"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	// FairnessPolicyExtensionPoint identifies the plugin type responsible for managing contention between Flows.
	FairnessPolicyExtensionPoint = "FairnessPolicy"
)

// FairnessPolicy governs the distribution of dispatch opportunities among competing Flows within the same Priority
// Band.
//
// In simple terms, this policy answers the question: "Which flow gets to dispatch a request next?"
//
// While "Priority" determines strictly which group of flows is serviced first, "Fairness" determines how resources are
// shared when multiple flows in that same group are fighting for capacity.
//
// Architecture (Flyweight Pattern):
// Fairness plugins are Singletons. A single instance of a FairnessPolicy handles the logic for potentially many
// different Priority Bands. To support this, the plugin must be purely functional, separating its Logic (methods) from
// its State (data).
//
//   - Logic: Defined here in the FairnessPolicy interface.
//   - State: Created via NewState() and stored on the PriorityBandAccessor.
//
// Conformance: Implementations MUST ensure all methods are goroutine-safe.
type FairnessPolicy interface {
	plugin.Plugin

	// NewState creates the scoped, mutable storage required by this policy for a single Priority Band.
	//
	// Because the plugin instance itself is shared globally, it cannot hold state like "current round-robin index" or
	// "accumulated deficits" inside struct fields. Instead, it creates this state object once per Band.
	//
	// The Flow Registry manages the lifecycle of this object, storing it on the Priority Band and passing it back to the
	// plugin via the PriorityBandAccessor during Pick.
	//
	// Returns:
	//   - any: The opaque state object (e.g., &roundRobinCursor{index: 0}).
	NewState(ctx context.Context) any

	// Pick inspects the active flows in the provided Flow Group (Priority Band) and selects the "winner" for the next
	// dispatch attempt.
	//
	// This is the core logic loop. The implementation should:
	//  1. Retrieve its scoped state from band.GetPolicyState().
	//  2. Cast the state to its concrete type (e.g., *roundRobinCursor).
	//  3. Apply its algorithm to select a FlowQueueAccessor.
	//  4. Update the state (e.g., increment the cursor) if necessary.
	//
	// State may also be updated out-of-band (e.g., from monitoring a metrics server, from integrating with request
	// lifeycle hooks, etc.).
	//
	// Returns:
	//   - flow: The Flow to service next. Returns nil if no valid candidate is found (e.g., all queues empty).
	//   - err: Only returned for unrecoverable internal errors. Policies should generally return (nil, nil) if simply
	//     nothing is eligible.
	Pick(ctx context.Context, flowGroup PriorityBandAccessor) (flow FlowQueueAccessor, err error)
}
