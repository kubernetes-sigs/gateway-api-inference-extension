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

package contracts

import "context"

// SaturationDetector defines the contract for a component that provides real-time load signals to the
// `controller.FlowController`.
//
// This interface abstracts away the complexity of determining system load. An implementation would consume various
// backend metrics (e.g., queue depths, KV cache utilization, observed latencies) and translate them into a simple
// boolean signal.
//
// This decoupling is important because it allows the saturation detection logic to evolve independently of the core
// `controller.FlowController` engine, which is only concerned with the final true/false signal.
//
// # Conformance
//
// Implementations MUST be goroutine-safe.
type SaturationDetector interface {
	// IsSaturated returns true if the system's backend resources are considered saturated.
	// `controller.FlowController`'s dispatch workers call this method to decide whether to pause or throttle dispatch
	// operations to prevent overwhelming the backends.
	IsSaturated(ctx context.Context) bool
}
