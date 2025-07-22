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

// FlowSpecification defines the complete configuration for a single logical flow.
// It is the data contract used by the `contracts.FlowRegistry` to create and manage the lifecycle of queues and
// policies.
type FlowSpecification struct {
	// ID is the unique identifier for this flow (e.g., model name, tenant ID).
	ID string

	// Priority is the numerical priority level for this flow. Lower values indicate higher priority.
	Priority uint
}
