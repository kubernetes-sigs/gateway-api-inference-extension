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
// Control Plane Events (Transport)
// =============================================================================

// event is a marker interface for internal state machine events processed by the `FlowRegistry`'s event loop.
type event interface {
	isEvent()
}

// queueStateChangedEvent is sent when a `managedQueue`'s state changes, carrying a `queueStateSignal`.
type queueStateChangedEvent struct {
	shardID string
	key     types.FlowKey
	signal  queueStateSignal
}

func (queueStateChangedEvent) isEvent() {}

// shardStateChangedEvent is sent when a `registryShard`'s state changes, carrying a `shardStateSignal`.
type shardStateChangedEvent struct {
	shardID string
	signal  shardStateSignal
}

func (shardStateChangedEvent) isEvent() {}

// syncEvent is used exclusively for testing to synchronize test execution with the `FlowRegistry`'s event loop.
// It allows tests to wait until all preceding events have been processed, eliminating the need for polling.
// test-only
type syncEvent struct {
	// doneCh is used by the event loop to signal back to the sender that the sync event has been processed.
	doneCh chan struct{}
}

func (syncEvent) isEvent() {}

// =============================================================================
// Control Plane Signals and Callbacks
// =============================================================================

// --- Queue Signals ---

// queueStateSignal is an enum describing the edge-triggered state change events emitted by a `managedQueue`.
type queueStateSignal int

const (
	// queueStateSignalBecameEmpty is sent when a queue transitions from non-empty to empty.
	// Trigger: Len > 0 -> Len == 0. Used for inactivity GC tracking.
	queueStateSignalBecameEmpty queueStateSignal = iota

	// queueStateSignalBecameNonEmpty is sent when a queue transitions from empty to non-empty.
	// Trigger: Len == 0 -> Len > 0. Used for inactivity GC tracking.
	queueStateSignalBecameNonEmpty
)

func (s queueStateSignal) String() string {
	switch s {
	case queueStateSignalBecameEmpty:
		return "QueueBecameEmpty"
	case queueStateSignalBecameNonEmpty:
		return "QueueBecameNonEmpty"
	default:
		return fmt.Sprintf("Unknown(%d)", s)
	}
}

// signalQueueStateFunc defines the callback function that a `managedQueue` uses to signal lifecycle events.
// Implementations should avoid blocking on internal locks or I/O. However, they are expected to block if the
// `FlowRegistry`'s event channel is full; this behavior is required to apply backpressure and ensure reliable event
// delivery for the GC system.
type signalQueueStateFunc func(key types.FlowKey, signal queueStateSignal)

// --- Shard Signals ---

// shardStateSignal is an enum describing the edge-triggered state change events emitted by a `registryShard`.
type shardStateSignal int

const (
	// shardStateSignalBecameDrained is sent when a Draining shard transitions to empty.
	// Trigger: Transition from `componentStatusDraining` -> `componentStatusDrained`. Used for final GC of the shard.
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

// signalShardStateFunc defines the callback function that a `registryShard` uses to signal its own state changes.
// Implementations should avoid blocking on internal locks or I/O. However, they are expected to block if the
// `FlowRegistry`'s event channel is full (see `signalQueueStateFunc` for rationale).
type signalShardStateFunc func(shardID string, signal shardStateSignal)

// --- Statistics Propagation ---

// propagateStatsDeltaFunc defines the callback function used to propagate statistics changes (deltas) up the hierarchy
// (Queue -> Shard -> Registry).
// Implementations MUST be non-blocking (relying on atomics).
type propagateStatsDeltaFunc func(priority uint, lenDelta, byteSizeDelta int64)
