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

import (
	"strconv"
	"strings"
)

// FlowKey is the unique, immutable identifier for a single, independently managed flow instance within the
// `contracts.FlowRegistry`.
// It combines a logical grouping `ID` with a specific `Priority` level to form a composite primary key.
//
// The core architectural principle of this model is that each unique, immutable `FlowKey` represents a completely
// separate stream of traffic with its own queue, lifecycle and statistics.
type FlowKey struct {
	// ID is the logical grouping identifier for a flow, such as a tenant ID or a model name. This ID can be shared across
	// multiple `FlowKey`s to represent different classes of traffic for the same logical entity. It provides a way to
	// group related traffic but does not uniquely identify a manageable flow instance on its own.
	ID string

	// Priority is the numerical priority level that defines the priority band for this specific flow instance.
	//
	// A different Priority value for the same ID creates a distinct flow instance. For example, the keys
	// `{ID: "TenantA", Priority: 10}` and `{ID: "TenantA", Priority: 100}` are treated as two completely separate,
	// concurrently active flows. This allows a single tenant to serve traffic at multiple priority levels simultaneously.
	//
	// Because the `FlowKey` is immutable, changing the priority of traffic requires using a new `FlowKey`; the old flow
	// instance will be automatically garbage collected by the registry when it becomes idle.
	Priority int
}

func (k FlowKey) String() string {
	return k.ID + ":" + strconv.FormatUint(uint64(k.Priority), 10)
}

// Compare provides a stable comparison function for two FlowKey instances, suitable for use with sorting algorithms.
// It returns 1 if the key is less than the other, 0 if they are equal, and -1 if the key is greater than the other.
// The comparison is performed first by `Priority` (ascending, higher priority first) and then by `ID` (ascending).
func (k FlowKey) Compare(other FlowKey) int {
	if k.Priority > other.Priority { // Higher number means higher priority
		return -1
	}
	if k.Priority < other.Priority {
		return 1
	}
	return strings.Compare(k.ID, other.ID)
}

// FlowSpecification defines the complete configuration for a single logical flow instance, which is uniquely identified
// by its immutable `Key`.
//
// It is the primary data contract used by the `contracts.FlowRegistry` to register or update a flow instance. The
// registry manages one flow instance for each unique `FlowKey`. While the `Key` itself is immutable, the specification
// can be updated via `RegisterOrUpdateFlow` to modify configurable parameters (e.g., policies, capacity limits) for the
// existing flow instance (when these update stories are supported).
type FlowSpecification struct {
	// Key is the unique, immutable identifier for the flow instance this specification describes.
	Key FlowKey

	// TODO: Add other flow-scoped configuration fields here, such as:
	// - IntraFlowDispatchPolicy intra.RegisteredPolicyName
	// - CapacityBytes uint64
}
