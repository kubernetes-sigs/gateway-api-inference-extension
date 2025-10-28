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
	"context"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
)

// FinalState encapsulates the terminal outcome of a FlowItem's lifecycle.
type FinalState struct {
	Outcome types.QueueOutcome
	Err     error
}

// FlowItem is the internal representation of a request managed by the Flow Controller.
//
// # Lifecycle Management
//
// Finalization (determining outcome) can be initiated by the Controller (e.g., Context expiry) or the Processor (e.g.,
// Dispatch/Reject). It sets the outcome and signals the waiting goroutine.
//
// # Synchronization
//
// Atomic operations synchronize state across the Controller and Processor goroutines:
//   - finalState (atomic.Pointer): Safely publishes the outcome.
//   - handle (atomic.Pointer): Safely publishes the queue admission status.
type FlowItem struct {
	// --- Immutable fields during a single lifecycle ---

	enqueueTime     time.Time
	effectiveTTL    time.Duration
	originalRequest types.FlowControlRequest

	// --- Synchronized State ---

	// handle stores the types.QueueItemHandle atomically.
	// Written by the Processor (SetHandle) when admitted.
	// Read by inferOutcome (called by Finalize) to infer the outcome (Rejected vs. Evicted).
	// Distinguishing between pre-admission (Rejection) and post-admission (Eviction) during asynchronous finalization
	// relies on whether this handle is nil or non-nil.
	handle atomic.Pointer[types.QueueItemHandle]

	// finalState holds the result of the finalization. Stored atomically once.
	// Use FinalState() for safe access.
	finalState atomic.Pointer[FinalState]

	// --- Finalization Signaling ---

	// done is the channel used to signal the completion of the item's lifecycle.
	// Buffered to size 1 to prevent Finalize from blocking.
	done chan *FinalState

	// onceFinalize ensures the finalization logic runs exactly once per lifecycle.
	onceFinalize sync.Once
}

var _ types.QueueItemAccessor = &FlowItem{}

// NewItem allocates and initializes a new FlowItem for a request lifecycle.
func NewItem(req types.FlowControlRequest, effectiveTTL time.Duration, enqueueTime time.Time) *FlowItem {
	return &FlowItem{
		enqueueTime:     enqueueTime,
		effectiveTTL:    effectiveTTL,
		originalRequest: req,
		done:            make(chan *FinalState, 1),
	}
}

// EnqueueTime returns the time the item was logically accepted by the FlowController.
func (fi *FlowItem) EnqueueTime() time.Time { return fi.enqueueTime }

// EffectiveTTL returns the actual time-to-live assigned to this item.
func (fi *FlowItem) EffectiveTTL() time.Duration { return fi.effectiveTTL }

// OriginalRequest returns the original types.FlowControlRequest object.
func (fi *FlowItem) OriginalRequest() types.FlowControlRequest { return fi.originalRequest }

// Done returns a read-only channel that will receive the FinalState pointer exactly once.
func (fi *FlowItem) Done() <-chan *FinalState { return fi.done }

// FinalState returns the FinalState if the item has been finalized, or nil otherwise.
// Safe for concurrent access.
func (fi *FlowItem) FinalState() *FinalState { return fi.finalState.Load() }

// Handle returns the types.QueueItemHandle for this item within a queue.
// Returns nil if the item is not in a queue. Safe for concurrent access.
func (fi *FlowItem) Handle() types.QueueItemHandle {
	ptr := fi.handle.Load()
	if ptr == nil {
		return nil
	}
	return *ptr
}

// SetHandle associates a types.QueueItemHandle with this item. Called by the queue implementation (via Processor).
// Safe for concurrent access.
func (fi *FlowItem) SetHandle(handle types.QueueItemHandle) { fi.handle.Store(&handle) }

// Finalize determines the item's terminal state based on the provided cause (e.g., Context error) and the item's
// current admission status (queued or not).
//
// This method is intended for asynchronous finalization initiated by the Controller (e.g., TTL expiry).
// It is idempotent.
func (fi *FlowItem) Finalize(cause error) {
	fi.onceFinalize.Do(func() {
		// Atomically load the handle to determine if the item was admitted to a queue.
		// This synchronization is critical for correctly inferring the outcome across goroutines.
		isQueued := fi.Handle() != nil
		outcome, finalErr := inferOutcome(cause, isQueued)
		fi.finalizeInternal(outcome, finalErr)
	})
}

// FinalizeWithOutcome sets the item's terminal state explicitly.
//
// This method is intended for synchronous finalization by the Processor (Dispatch, Reject) or the Controller
// (Distribution failure).
// It is idempotent.
func (fi *FlowItem) FinalizeWithOutcome(outcome types.QueueOutcome, err error) {
	fi.onceFinalize.Do(func() {
		fi.finalizeInternal(outcome, err)
	})
}

// finalizeInternal is the core finalization logic. It must be called within the sync.Once.Do block.
// It captures the state, stores it atomically, and signals the Done channel.
func (fi *FlowItem) finalizeInternal(outcome types.QueueOutcome, err error) {
	finalState := &FinalState{
		Outcome: outcome,
		Err:     err,
	}

	// Atomically store the pointer. This is the critical memory barrier that publishes the state safely.
	fi.finalState.Store(finalState)

	duration := time.Since(fi.enqueueTime)
	flowKey := fi.originalRequest.FlowKey()
	metrics.RecordFlowControlRequestQueueDuration(flowKey.ID, strconv.Itoa(flowKey.Priority), outcome.String(), duration)

	fi.done <- finalState
	close(fi.done)
}

// inferOutcome determines the correct QueueOutcome and Error based on the cause of finalization and whether the item
// was already admitted to a queue.
func inferOutcome(cause error, isQueued bool) (types.QueueOutcome, error) {
	var specificErr error
	var outcomeIfEvicted types.QueueOutcome
	switch {
	case errors.Is(cause, types.ErrTTLExpired) || errors.Is(cause, context.DeadlineExceeded):
		specificErr = types.ErrTTLExpired
		outcomeIfEvicted = types.QueueOutcomeEvictedTTL
	case errors.Is(cause, context.Canceled):
		specificErr = fmt.Errorf("%w: %w", types.ErrContextCancelled, cause)
		outcomeIfEvicted = types.QueueOutcomeEvictedContextCancelled
	default:
		// Handle other potential causes (e.g., custom context errors).
		specificErr = cause
		outcomeIfEvicted = types.QueueOutcomeEvictedOther
	}

	if isQueued {
		// The item was in the queue when it expired/cancelled.
		return outcomeIfEvicted, fmt.Errorf("%w: %w", types.ErrEvicted, specificErr)
	}

	// The item was not yet in the queue (e.g., buffered in enqueueChan).
	// We treat this as a rejection, as it never formally consumed queue capacity.
	return types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, specificErr)
}
