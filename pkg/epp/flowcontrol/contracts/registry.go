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

// Package contracts defines the service interfaces that decouple the core `controller.FlowController` engine from its
// primary dependencies. In alignment with a "Ports and Adapters" (or "Hexagonal") architectural style, these
// interfaces represent the "ports" through which the engine communicates.
//
// This package contains the primary service contracts for the Flow Registry and Saturation Detector.
package contracts

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
)

// ManagedQueue defines the interface for a flow's queue instance on a specific shard.
// It wraps an underlying `framework.SafeQueue`, augmenting it with lifecycle validation against the `FlowRegistry` and
// integrating atomic statistics updates.
//
// Conformance:
//   - All methods (including those embedded from `framework.SafeQueue`) MUST be goroutine-safe.
//   - Mutating methods (`Add()`, `Remove()`, `CleanupExpired()`, `Drain()`) MUST ensure the flow instance still exists
//     and is valid within the `FlowRegistry` before proceeding. They MUST also atomically update relevant statistics
//     (e.g., queue length, byte size) at both the queue and priority-band levels.
type ManagedQueue interface {
	framework.SafeQueue

	// FlowQueueAccessor returns a read-only, flow-aware accessor for this queue.
	// This accessor is primarily used by policy plugins to inspect the queue's state in a structured way.
	//
	// Conformance: This method MUST NOT return nil.
	FlowQueueAccessor() framework.FlowQueueAccessor
}
