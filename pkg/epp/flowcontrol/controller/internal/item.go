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

package internal

import (
	"sync"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// FinalState encapsulates the terminal outcome of a `FlowItem`'s lifecycle.
// It is sent over the item's `Done()` channel exactly once.
type FinalState struct {
	Outcome types.QueueOutcome
	Err     error
}

// FlowItem is the internal representation of a request managed by the `FlowController`.
type FlowItem struct {
	// --- Immutable fields (set at creation) ---

	enqueueTime     time.Time
	effectiveTTL    time.Duration
	originalRequest types.FlowControlRequest
	handle          types.QueueItemHandle

	// --- Finalization state (protected by onceFinalize) ---

	// done is closed exactly once when the item is finalized.
	// The closing of this channel establishes a "happens-before" memory barrier, guaranteeing that writes to `outcome`
	// and `err` are visible to any goroutine that has successfully read from `done`.
	done chan FinalState

	// finalState  is safely visible to any goroutine after it has confirmed the channel is closed.
	finalState FinalState

	// onceFinalize ensures the `finalize()` logic is idempotent.
	onceFinalize sync.Once
}

// ensure FlowItem implements the interface.
var _ types.QueueItemAccessor = &FlowItem{}

// NewItem creates a new `FlowItem`.
func NewItem(req types.FlowControlRequest, effectiveTTL time.Duration, enqueueTime time.Time) *FlowItem {
	return &FlowItem{
		enqueueTime:     enqueueTime,
		effectiveTTL:    effectiveTTL,
		originalRequest: req,
		// Buffer to size one, preventing finalizing goroutine (e.g., the dispatcher) from blocking if the waiting
		// goroutine has already timed out and is no longer reading.
		done: make(chan FinalState, 1),
	}
}

// EnqueueTime returns the time the item was logically accepted by the `FlowController` for queuing. This is used as the
// basis for TTL calculations.
func (fi *FlowItem) EnqueueTime() time.Time { return fi.enqueueTime }

// EffectiveTTL returns the actual time-to-live assigned to this item by the `FlowController`.
func (fi *FlowItem) EffectiveTTL() time.Duration { return fi.effectiveTTL }

// OriginalRequest returns the original, underlying `types.FlowControlRequest` object.
func (fi *FlowItem) OriginalRequest() types.FlowControlRequest { return fi.originalRequest }

// Handle returns the `types.QueueItemHandle` that uniquely identifies this item within a specific queue instance. It
// returns nil if the item has not yet been added to a queue.
func (fi *FlowItem) Handle() types.QueueItemHandle { return fi.handle }

// SetHandle associates a `types.QueueItemHandle` with this item. This method is called by a `framework.SafeQueue`
// implementation immediately after the item is added to the queue.
func (fi *FlowItem) SetHandle(handle types.QueueItemHandle) { fi.handle = handle }

// Done returns a channel that is closed when the item has been finalized (e.g., dispatched, rejected, or evicted).
func (fi *FlowItem) Done() <-chan FinalState {
	return fi.done
}

// Finalize sets the item's terminal state and signals the waiting goroutine by closing its `done` channel idempotently.
// This method is idempotent and is the single point where an item's lifecycle concludes.
// It is intended to be called only by the component that owns the item's lifecycle, such as a `ShardProcessor`.
func (fi *FlowItem) Finalize(outcome types.QueueOutcome, err error) {
	fi.onceFinalize.Do(func() {
		finalState := FinalState{Outcome: outcome, Err: err}
		fi.finalState = finalState
		fi.done <- finalState
		close(fi.done)
	})
}

// isFinalized checks if the item has been finalized without blocking or consuming the final state.
// It is a side-effect-free check used by the `ShardProcessor` as a defensive measure to avoid operating on
// already-completed items.
func (fi *FlowItem) isFinalized() bool {
	// A buffered channel of size 1 can be safely and non-blockingly checked by its length.
	// If the finalize function has run, it will have sent a value, and the length will be 1.
	return len(fi.done) > 0
}
