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

// Package types defines the fundamental data structures, interfaces, and errors that form the vocabulary of the Flow
// Control system. It establishes the core data contracts for the request lifecycle, from initial submission to final,
// reportable outcome.
//
// # The Request Lifecycle
//
// The primary entry point to the `controller.FlowController` is the synchronous `EnqueueAndWait` method. The types in
// this package are designed to model a request's journey through this blocking call.
//
//  1. A client first constructs an object that implements the `FlowControlRequest` interface. This is the "raw" input,
//     containing the essential data for the request, such as its `FlowID` and `ByteSize`. This object is passed to
//     `EnqueueAndWait`.
//
//  2. Internally, the `controller.FlowController` wraps the `FlowControlRequest` in an object that implements the
//     `QueueItemAccessor` interface. This is an enriched, read-only view used by policies and queues. It adds internal
//     metadata like `EnqueueTime` and `EffectiveTTL`.
//
//  3. If the request is accepted and added to a `framework.SafeQueue`, the queue creates a `QueueItemHandle`. This is
//     an opaque, queue-specific handle that the controller uses to perform targeted operations (like removal) without
//     needing to know the queue's internal implementation details.
//
//  4. The `EnqueueAndWait` method blocks until the request reaches a terminal state. This final state is reported using
//     a `QueueOutcome` enum and a corresponding `error`.
//
// # Final State Reporting: Outcomes and Errors
//
// This combination of a concise enum and a detailed error provides a clear, machine-inspectable result.
//
//   - `QueueOutcome`: A low-cardinality enum summarizing the final result (e.g., `QueueOutcomeDispatched`,
//     `QueueOutcomeRejectedCapacity`). This is ideal for metrics.
//
//   - `error`: For any non-dispatch outcome, a specific sentinel error is returned. These are nested to provide rich
//     context. Callers can use `errors.Is()` to check for the general class of failure (`ErrRejected` or `ErrEvicted`),
//     and then unwrap the error to find the specific cause (e.g., `ErrQueueAtCapacity` or `ErrTTLExpired`).
package types
