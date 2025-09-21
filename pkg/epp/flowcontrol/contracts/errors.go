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

var (
	// ErrFlowInstanceNotFound indicates that a requested flow instance (a `ManagedQueue`) does not exist.
	ErrFlowInstanceNotFound = errors.New("flow instance not found")

	// ErrFlowIDEmpty indicates that a flow specification was provided with an empty flow ID.
	ErrFlowIDEmpty = errors.New("flow ID cannot be empty")

	// ErrPriorityBandNotFound indicates that a requested priority band does not exist in the registry configuration.
	ErrPriorityBandNotFound = errors.New("priority band not found")

	// ErrPolicyQueueIncompatible indicates that a selected policy is not compatible with the capabilities of the queue.
	ErrPolicyQueueIncompatible = errors.New("policy is not compatible with queue capabilities")

	// ErrInvalidShardCount indicates that an invalid shard count was provided (e.g., zero or negative).
	ErrInvalidShardCount = errors.New("invalid shard count")

	// ErrShardDraining indicates that an operation could not be completed because the target shard is in the process of
	// being gracefully drained. The caller should retry the operation on a different, Active shard.
	ErrShardDraining = errors.New("shard is draining")
)
