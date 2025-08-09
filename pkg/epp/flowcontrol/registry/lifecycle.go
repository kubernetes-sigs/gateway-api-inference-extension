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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// =============================================================================
// Component Lifecycle State Machine
//
// The registry manages stateful components (`managedQueue`, `registryShard`).
// These components follow a well-defined lifecycle. The system is an
// event-driven state machine where changes in state trigger signals, which are
// processed by the central `FlowRegistry` control plane.
// =============================================================================

// componentStatus represents the lifecycle state of a managed component (Queue or Shard).
// It is intended to be stored and manipulated using atomic operations (e.g., `atomic.Int32`) to ensure robust, atomic
// state transitions and eliminate inconsistent intermediate states.
type componentStatus int32

const (
	// componentStatusActive indicates the component is fully operational and accepting new work.
	componentStatusActive componentStatus = iota

	// componentStatusDraining indicates the component is shutting down. It is not accepting new work, but is still
	// processing existing work.
	componentStatusDraining

	// componentStatusDrained indicates the component has finished draining and is now empty.
	// The transition into this state (from `componentStatusDraining`) occurs exactly once via `CompareAndSwap` and
	// triggers the corresponding `BecameDrained` signal. This acts as an atomic latch for GC.
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

// =============================================================================
// Control Plane Signals and Callbacks
//
// These definitions establish the communication protocol from the data plane components (Queues, Shards) up to the
// control plane (`FlowRegistry`).
// =============================================================================

// --- Queue Signals ---

// queueStateSignal is an enum describing the edge-triggered state change events emitted by a `managedQueue`.
type queueStateSignal int

const (
	// queueStateSignalBecameEmpty is sent when an Active queue transitions from non-empty to empty.
	// Trigger: Len > 0 -> Len == 0. Used for inactivity GC tracking.
	queueStateSignalBecameEmpty queueStateSignal = iota

	// queueStateSignalBecameNonEmpty is sent when an Active queue transitions from empty to non-empty.
	// Trigger: Len == 0 -> Len > 0. Used for inactivity GC tracking.
	queueStateSignalBecameNonEmpty

	// queueStateSignalBecameDrained is sent when a Draining queue transitions to empty.
	// Trigger: Transition from `componentStatusDraining` -> `componentStatusDrained`. Used for final GC of the queue
	// instance.
	queueStateSignalBecameDrained
)

func (s queueStateSignal) String() string {
	switch s {
	case queueStateSignalBecameEmpty:
		return "QueueBecameEmpty"
	case queueStateSignalBecameNonEmpty:
		return "QueueBecameNonEmpty"
	case queueStateSignalBecameDrained:
		return "QueueBecameDrained"
	default:
		return fmt.Sprintf("Unknown(%d)", s)
	}
}

// signalQueueStateFunc defines the callback function that a `managedQueue` uses to signal lifecycle events to its
// parent shard.
// Implementations MUST NOT block on internal locks or I/O. However, they are expected to block if the `FlowRegistry`'s
// event channel is full; this is required behavior to apply necessary backpressure and ensure reliable event delivery.
type signalQueueStateFunc func(spec types.FlowSpecification, signal queueStateSignal)

// --- Shard Signals ---

// shardStateSignal is an enum describing the edge-triggered state change events emitted by a `registryShard`.
type shardStateSignal int

const (
	// shardStateSignalBecameDrained is sent when a Draining shard transitions to empty.
	// Trigger: Transition from `componentStatusDraining` -> componentStatusDrained`. Used for final GC of the shard.
	shardStateSignalBecameDrained shardStateSignal = iota
)

func (s shardStateSignal) String() string {
	switch s {
	case shardStateSignalBecameDrained:
		return "ShardBecameDrained"
	default:
		return fmt.Sprintf("Unknown(%d)", s)
	}
}

// signalShardStateFunc defines the callback function that a `registryShard` uses to signal its own state changes to the
// parent registry.
// Implementations MUST NOT block on internal locks or I/O. However, they are expected to block if the `FlowRegistry`'s
// event channel is full; this is required behavior to apply necessary backpressure and ensure reliable event delivery.
// Note: The `registryShard` pointer is passed here to facilitate efficient identification and removal during GC.
type signalShardStateFunc func(shard *registryShard, signal shardStateSignal)

// --- Statistics Propagation ---

// propagateStatsDeltaFunc defines the callback function that a component uses to propagate its statistics changes
// (deltas) up to its parent (Queue -> Shard -> Registry).
// Implementations MUST be non-blocking (relying on atomics) and must not acquire any locks held by the caller.
type propagateStatsDeltaFunc func(priority uint, lenDelta, byteSizeDelta int64)

// =============================================================================
// Control Plane Events (Transport)
//
// These structures define the data transported over the `FlowRegistry`'s event
// channel, carrying the signals defined above or timer expirations to the
// centralized event loop.
// =============================================================================

// event is a marker interface for internal state machine events processed by the `FlowRegistry`'s event loop.
type event interface {
	isEvent()
}

// gcTimerFiredEvent is sent when a flow's garbage collection timer expires.
type gcTimerFiredEvent struct {
	flowID     string
	generation uint64
}

func (gcTimerFiredEvent) isEvent() {}

// queueStateChangedEvent is sent when a `managedQueue`'s state changes, carrying a `queueStateSignal`.
type queueStateChangedEvent struct {
	shardID string
	spec    types.FlowSpecification
	signal  queueStateSignal
}

func (queueStateChangedEvent) isEvent() {}

// shardStateChangedEvent is sent when a `registryShard`'s state changes, carrying a `shardStateSignal`.
type shardStateChangedEvent struct {
	shard  *registryShard
	signal shardStateSignal
}

func (shardStateChangedEvent) isEvent() {}

// syncEvent is a special event used exclusively for testing to synchronize the test execution with the `FlowRegistry`'s
// event loop. It allows tests to wait until all preceding events have been processed, eliminating the need for polling.
type syncEvent struct {
	// doneCh is used by the event loop to signal back to the sender that the sync event (and thus all preceding events)
	// has been processed.
	doneCh chan struct{}
}

func (syncEvent) isEvent() {}
