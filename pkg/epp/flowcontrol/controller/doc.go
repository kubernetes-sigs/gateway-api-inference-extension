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

// Package controller contains the implementation of the `FlowController` engine.
//
// # Overview
//
// The `FlowController` is the central processing engine of the Flow Control system. It acts as a stateless supervisor
// that orchestrates a pool of stateful workers (`internal.ShardProcessor`), managing the lifecycle of all incoming
// requests from initial submission to a terminal outcome (dispatch, rejection, or eviction).
//
// # Architecture: Supervisor-Worker Pattern
//
// This package implements a classic "supervisor-worker" pattern to achieve high throughput and dynamic scalability.
//
//   - The `FlowController` (Supervisor): The public-facing API of the system. Its primary responsibilities are to
//     execute a distribution algorithm to select the optimal worker for a new request and to manage the lifecycle of
//     the worker pool, ensuring it stays synchronized with the underlying shard topology defined by the
//     `contracts.FlowRegistry`.
//   - The `internal.ShardProcessor` (Worker): A stateful, single-goroutine actor responsible for the entire lifecycle
//     of requests on a single shard. The supervisor manages a pool of these workers, one for each
//     `contracts.RegistryShard`.
package controller
