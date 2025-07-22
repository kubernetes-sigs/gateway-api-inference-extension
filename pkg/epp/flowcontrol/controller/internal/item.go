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
	"sync/atomic"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// flowItem is the internal representation of a request managed by the `FlowController`. It implements the
// `types.QueueItemAccessor` interface, which is the primary view of the item used by queue and policy implementations.
// It wraps the original `types.FlowControlRequest` and adds metadata for queuing, lifecycle management, and policy
// interaction.
//
// # Concurrency
//
// The `finalize` method is the primary point of concurrency concern. It is designed to be atomic and idempotent through
// the use of `sync.Once`. This guarantees that an item's final outcome can be set exactly once, even if multiple
// goroutines (e.g., the main dispatch loop and the expiry cleanup loop) race to finalize it. All other fields are set
// at creation time and are not modified thereafter, making them safe for concurrent access.
type flowItem struct {
	// enqueueTime is the timestamp when the item was logically accepted by the `FlowController`.
	enqueueTime time.Time
	// effectiveTTL is the actual time-to-live assigned to this item.
	effectiveTTL time.Duration
	// originalRequest is the underlying request object.
	originalRequest types.FlowControlRequest
	// handle is the unique identifier for this item within a specific queue instance.
	handle types.QueueItemHandle

	// done is closed exactly once when the item is finalized (dispatched or evicted/rejected).
	done chan struct{}
	// err stores the final error state if the item was not successfully dispatched.
	// It is written to exactly once, protected by `onceFinalize`.
	err atomic.Value // Stores error
	// outcome stores the final `types.QueueOutcome` of the item's lifecycle.
	// It is written to exactly once, protected by `onceFinalize`.
	outcome atomic.Value // Stores `types.QueueOutcome`
	// onceFinalize ensures the `finalize()` logic is idempotent.
	onceFinalize sync.Once
}

// ensure flowItem implements the interface.
var _ types.QueueItemAccessor = &flowItem{}

// NewItem creates a new `flowItem`, which is the internal representation of a request inside the `FlowController`.
// This constructor is exported so that the parent `controller` package can create items to be passed into the
// `internal` package's processors. It initializes the item with a "NotYetFinalized" outcome and an open `done` channel.
func NewItem(req types.FlowControlRequest, effectiveTTL time.Duration, enqueueTime time.Time) *flowItem {
	fi := &flowItem{
		enqueueTime:     enqueueTime,
		effectiveTTL:    effectiveTTL,
		originalRequest: req,
		done:            make(chan struct{}),
	}
	// Initialize the outcome to its zero state.
	fi.outcome.Store(types.QueueOutcomeNotYetFinalized)
	return fi
}

// EnqueueTime returns the time the item was logically accepted by the `FlowController` for queuing. This is used as the
// basis for TTL calculations.
func (fi *flowItem) EnqueueTime() time.Time { return fi.enqueueTime }

// EffectiveTTL returns the actual time-to-live assigned to this item by the `FlowController`.
func (fi *flowItem) EffectiveTTL() time.Duration { return fi.effectiveTTL }

// OriginalRequest returns the original, underlying `types.FlowControlRequest` object.
func (fi *flowItem) OriginalRequest() types.FlowControlRequest { return fi.originalRequest }

// Handle returns the `types.QueueItemHandle` that uniquely identifies this item within a specific queue instance. It
// returns nil if the item has not yet been added to a queue.
func (fi *flowItem) Handle() types.QueueItemHandle { return fi.handle }

// SetHandle associates a `types.QueueItemHandle` with this item. This method is called by a `framework.SafeQueue`
// implementation immediately after the item is added to the queue.
func (fi *flowItem) SetHandle(handle types.QueueItemHandle) { fi.handle = handle }

// Done returns a channel that is closed when the item has been finalized (e.g., dispatched or evicted).
// This is the primary mechanism for consumers to wait for an item's outcome. It is designed to be used in a `select`
// statement, allowing the caller to simultaneously wait for other events, such as context cancellation.
//
// # Example Usage
//
//	select {
//	case <-item.Done():
//	    outcome, err := item.FinalState()
//	    // ... handle outcome
//	case <-ctx.Done():
//	    // ... handle cancellation
//	}
func (fi *flowItem) Done() <-chan struct{} {
	return fi.done
}

// FinalState returns the terminal outcome and error for the item.
//
// CRITICAL: This method must only be called after the channel returned by `Done()` has been closed. Calling it before
// the item is finalized may result in a race condition where the final state has not yet been written.
func (fi *flowItem) FinalState() (types.QueueOutcome, error) {
	outcomeVal := fi.outcome.Load()
	errVal := fi.err.Load()

	var finalOutcome types.QueueOutcome
	if oc, ok := outcomeVal.(types.QueueOutcome); ok {
		finalOutcome = oc
	} else {
		// This case should not happen if finalize is always called correctly, but we default to a safe value.
		finalOutcome = types.QueueOutcomeNotYetFinalized
	}

	var finalErr error
	if e, ok := errVal.(error); ok {
		finalErr = e
	}
	return finalOutcome, finalErr
}

// finalize sets the item's terminal state (`outcome`, `error`) and closes its `done` channel idempotently using
// `sync.Once`. This is the single, internal point where an item's lifecycle within the `FlowController` concludes.
func (fi *flowItem) finalize(outcome types.QueueOutcome, err error) {
	fi.onceFinalize.Do(func() {
		if err != nil {
			fi.err.Store(err)
		}
		fi.outcome.Store(outcome)
		close(fi.done)
	})
}

// isFinalized checks if the item has been finalized without blocking. It is used internally by the `ShardProcessor` as
// a defensive check to avoid operating on items that have already been completed.
func (fi *flowItem) isFinalized() bool {
	select {
	case <-fi.done:
		return true
	default:
		return false
	}
}
