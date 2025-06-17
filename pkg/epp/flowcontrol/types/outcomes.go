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

import "strconv"

// QueueOutcome represents the high-level final state of a request's lifecycle within the Flow Controller.
//
// It is returned by `FlowController.EnqueueAndWait()` along with a corresponding error. This enum is designed to be a
// low-cardinality label ideal for metrics, while the error provides fine-grained details for non-dispatched outcomes.
type QueueOutcome int

const (
	// QueueOutcomeDispatched indicates the request was successfully processed by the Flow Controller and unblocked for
	// the caller to proceed.
	// The associated error from `FlowController.EnqueueAndWait()` will be nil.
	QueueOutcomeDispatched QueueOutcome = iota

	// --- Pre-Enqueue Rejection Outcomes (request never entered a `framework.SafeQueue`) ---
	// For these outcomes, the error from `FlowController.EnqueueAndWait()` will wrap `ErrRejected`.

	// QueueOutcomeRejectedCapacity indicates rejection because queue capacity limits were met and displacement (if
	// applicable) failed to make space.
	// The associated error will wrap `ErrQueueAtCapacity` (and `ErrRejected`).
	QueueOutcomeRejectedCapacity

	// QueueOutcomeRejectedOther indicates rejection for reasons other than capacity before the request was formally
	// enqueued.
	// The specific underlying cause can be determined from the associated error (e.g., a nil request, an unregistered
	// flow ID, or a controller shutdown), which will be wrapped by `ErrRejected`.
	QueueOutcomeRejectedOther

	// --- Post-Enqueue Eviction Outcomes (request was in a `framework.SafeQueue` but not dispatched) ---
	// For these outcomes, the error from `FlowController.EnqueueAndWait()` will wrap `ErrEvicted`.

	// QueueOutcomeEvictedTTL indicates eviction from a queue because the request's effective Time-To-Live expired.
	// The associated error will wrap `ErrTTLExpired` (and `ErrEvicted`).
	QueueOutcomeEvictedTTL

	// QueueOutcomeEvictedContextCancelled indicates eviction from a queue because the request's own context (from
	// `FlowControlRequest.Context()`) was cancelled.
	// The associated error will wrap `ErrContextCancelled` (which may further wrap the underlying `context.Canceled` or
	// `context.DeadlineExceeded` error) (and `ErrEvicted`).
	QueueOutcomeEvictedContextCancelled

	// QueueOutcomeEvictedDisplaced indicates eviction from a queue to make space for another request due to a
	// displacement policy.
	// The associated error will wrap `ErrDisplaced` (and `ErrEvicted`).
	QueueOutcomeEvictedDisplaced

	// QueueOutcomeEvictedOther indicates eviction from a queue for reasons not covered by more specific eviction
	// outcomes.
	// The specific underlying cause can be determined from the associated error (e.g., a controller shutdown while the
	// item was queued), which will be wrapped by `ErrEvicted`.
	QueueOutcomeEvictedOther
)

// String returns a human-readable string representation of the QueueOutcome.
func (o QueueOutcome) String() string {
	switch o {
	case QueueOutcomeDispatched:
		return "Dispatched"
	case QueueOutcomeRejectedCapacity:
		return "RejectedCapacity"
	case QueueOutcomeRejectedOther:
		return "RejectedOther"
	case QueueOutcomeEvictedTTL:
		return "EvictedTTL"
	case QueueOutcomeEvictedContextCancelled:
		return "EvictedContextCancelled"
	case QueueOutcomeEvictedDisplaced:
		return "EvictedDisplaced"
	case QueueOutcomeEvictedOther:
		return "EvictedOther"
	default:
		// Return the integer value for unknown outcomes to aid in debugging.
		return "UnknownOutcome(" + strconv.Itoa(int(o)) + ")"
	}
}
