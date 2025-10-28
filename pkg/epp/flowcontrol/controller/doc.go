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

// Package controller contains the implementation of the FlowController engine.
//
// # Overview
//
// The FlowController is the central processing engine of the Flow Control layer. It acts as a stateless supervisor that
// orchestrates a pool of stateful workers (internal.ShardProcessor), managing the lifecycle of all incoming requests
// from initial submission to a terminal outcome (dispatch, rejection, or eviction).
//
// # Architecture: Supervisor-Worker Pattern
//
// This package implements a supervisor-worker pattern to achieve high throughput and dynamic scalability.
//
//   - The FlowController (Supervisor): The public-facing API of the system. Its primary responsibilities are to execute
//     a distribution algorithm to select the optimal worker for a new request and to manage the lifecycle of the worker
//     pool, ensuring it stays synchronized with the underlying shard topology defined by the contracts.FlowRegistry.
//   - The internal.ShardProcessor (Worker): A stateful, single-goroutine actor responsible for the entire lifecycle of
//     requests on a single shard. The supervisor manages a pool of these workers, one for each contracts.RegistryShard.
//
// # Concurrency Model
//
// The FlowController is designed to be highly concurrent and thread-safe. It acts primarily as a stateless distributor.
//
//   - EnqueueAndWait: Can be called concurrently by many goroutines.
//   - Worker Management: Uses a sync.Map (workers) for concurrent access and lazy initialization of workers.
//   - Supervision: A single background goroutine (run) manages the worker pool lifecycle (garbage collection).
//
// It achieves high throughput by minimizing shared state and relying on the internal ShardProcessors to handle state
// mutations serially (using an actor model).
//
// # Request Lifecycle and Ownership
//
// A request (represented internally as a FlowItem) has a lifecycle managed cooperatively by the Controller and a
// Processor. Defining ownership is critical for ensuring an item is finalized exactly once.
//
//  1. Submission (Controller): The Controller attempts to hand off the item to a Processor.
//  2. Handoff:
//     - Success: Ownership transfers to the Processor, which is now responsible for Finalization.
//     - Failure: Ownership remains with the Controller, which must Finalize the item.
//  3. Processing (Processor): The Processor enqueues, manages, and eventually dispatches or rejects the item.
//  4. Finalization: The terminal outcome is set. This can happen:
//     - Synchronously: The Processor determines the outcome (e.g., Dispatch, Capacity Rejection).
//     - Asynchronously: The Controller observes the request's Context expiry (TTL/Cancellation) and calls Finalize.
//
// The FlowItem uses atomic operations to safely coordinate the Finalization state across goroutines.
package controller
