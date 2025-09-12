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
	"errors"
)

// --- High-Level Outcome Errors ---

var (
	// ErrRejected is a sentinel error indicating a request was rejected by the `controller.FlowController` *before* being
	// formally enqueued. Errors returned by `FlowController.EnqueueAndWait()` that signify pre-queue rejection will wrap
	// this error.
	//
	// Callers should use `errors.Is(err, ErrRejected)` to check for this general class of failure.
	ErrRejected = errors.New("request rejected pre-queue")

	// ErrEvicted is a sentinel error indicating a request was removed from a queue *after* being successfully enqueued,
	// but for reasons other than successful dispatch (e.g., TTL expiry, displacement). Errors returned by
	// `FlowController.EnqueueAndWait()` that signify post-queue eviction will wrap this error.
	//
	// Callers should use `errors.Is(err, ErrEvicted)` to check for this general class of failure.
	ErrEvicted = errors.New("request evicted from queue")
)

// --- Pre-Enqueue Rejection Errors ---

// The following errors can occur before a request is formally added to a `framework.SafeQueue`. When returned by
// `FlowController.EnqueueAndWait()`, these specific errors will typically be wrapped by `ErrRejected`.
var (
	// ErrQueueAtCapacity indicates that a request could not be enqueued because queue capacity limits were met.
	ErrQueueAtCapacity = errors.New("queue at capacity")
)

// --- Post-Enqueue Eviction Errors ---

// The following errors occur when a request, already in a `framework.SafeQueue`, is removed for reasons other than
// dispatch. When returned by `FlowController.EnqueueAndWait()`, these specific errors will typically be wrapped by
// `ErrEvicted`.
var (
	// ErrTTLExpired indicates a request was evicted from a queue because its effective Time-To-Live expired.
	ErrTTLExpired = errors.New("request TTL expired")

	// ErrContextCancelled indicates a request was evicted because its associated context (from
	// `FlowControlRequest.Context()`) was cancelled. This error typically wraps the underlying `context.Canceled` or
	// `context.DeadlineExceeded` error.
	ErrContextCancelled = errors.New("request context cancelled")
)

// --- General `controller.FlowController` Errors ---

var (
	// ErrFlowControllerNotRunning indicates that an operation could not complete or an item was evicted because the
	// `controller.FlowController` is not running or is in the process of shutting down.
	//
	// When returned by `FlowController.EnqueueAndWait()`, this will be wrapped by `ErrRejected` (if rejection happens
	// before internal queuing) or `ErrEvicted` (if eviction happens after internal queuing).
	ErrFlowControllerNotRunning = errors.New("flow controller is not running")
)
