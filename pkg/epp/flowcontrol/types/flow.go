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

// Package types defines the core data structures and service contracts for the Flow Controller system. It establishes
// the "vocabulary" of the system, including the request lifecycle interfaces, final outcomes, and standard error types.
package types

// FlowSpecification defines the configuration of a logical flow, encapsulating its identity and registered priority.
//
// A FlowSpecification acts as the registration key for a flow within the Flow Registry.
type FlowSpecification interface {
	// ID returns the unique name or identifier for this flow (e.g., model name, tenant ID), corresponding to the value
	// from `FlowControlRequest.FlowID()`.
	ID() string

	// Priority returns the numerical priority level currently associated with this flow within the Flow Registry.
	//
	// Convention: Lower numerical values indicate higher priority.
	Priority() uint
}
