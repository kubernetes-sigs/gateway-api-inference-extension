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

package registry

import (
	"fmt"
)

// =============================================================================
// Component Lifecycle State Machine
// =============================================================================

// componentStatus represents the lifecycle state of a `registryShard`.
// It is manipulated using atomic operations (e.g., `atomic.Int32`) to ensure robust, atomic state transitions.
type componentStatus int32

const (
	// componentStatusActive indicates the component is fully operational and accepting new work.
	componentStatusActive componentStatus = iota

	// componentStatusDraining indicates the component is shutting down. It is not accepting new work, but is still
	// processing existing work.
	componentStatusDraining

	// componentStatusDrained indicates the component has finished draining and is empty.
	// The transition into this state (from `Draining`) occurs exactly once via `CompareAndSwap` and triggers the
	// `BecameDrained` signal. This acts as an atomic latch for GC.
	componentStatusDrained
)

func (s componentStatus) String() string {
	switch s {
	case componentStatusActive:
		return "Active"
	case componentStatusDraining:
		return "Draining"
	case componentStatusDrained:
		return "Drained"
	default:
		return fmt.Sprintf("Unknown(%d)", s)
	}
}
