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

package types

// FlowSpecification defines the configuration of a logical flow, encapsulating its identity and registered priority.
//
// It acts as the registration key for a flow within the `ports.FlowRegistry`.
type FlowSpecification struct {
	// ID returns the unique name or identifier for this logical flow, corresponding to the value from
	// `FlowControlRequest.FlowID()`.
	ID string

	// Priority returns the numerical priority level currently associated with this flow within the `ports.FlowRegistry`.
	//
	// Convention: Lower numerical values indicate higher priority.
	Priority uint
}
