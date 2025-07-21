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

import "errors"

// Registry Errors
var (
	// ErrFlowInstanceNotFound indicates that a requested flow instance (a `ManagedQueue`) does not exist in the registry
	// shard, either because the flow is not registered or the specific instance (e.g., a draining queue at a particular
	// priority) is not present.
	ErrFlowInstanceNotFound = errors.New("flow instance not found")

	// ErrPriorityBandNotFound indicates that a requested priority band does not exist in the registry because it was not
	// part of the initial configuration.
	ErrPriorityBandNotFound = errors.New("priority band not found")

	// ErrPolicyQueueIncompatible indicates that a selected policy is not compatible with the capabilities of the queue it
	// is intended to operate on. For example, a policy requiring priority-based peeking is used with a simple FIFO queue.
	ErrPolicyQueueIncompatible = errors.New("policy is not compatible with queue capabilities")
)
