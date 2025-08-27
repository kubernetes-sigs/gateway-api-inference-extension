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

// Package registry provides the concrete implementation of the `contracts.FlowRegistry` interface.
//
// # Architecture: A Sharded, Concurrent Control Plane
//
// This package implements the flow control state machine using a sharded architecture to enable scalable, parallel
// request processing. It separates the orchestration control plane from the request-processing data plane.
//
//   - `FlowRegistry`: The top-level orchestrator (Control Plane). It manages the lifecycle of all flows and shards,
//     handling registration, garbage collection, and scaling operations.
//   - `registryShard`: A slice of the data plane. It holds a partition of the total state and provides a
//     read-optimized, concurrent-safe view for a single `controller.FlowController` worker.
//   - `managedQueue`: A stateful decorator around a `framework.SafeQueue`. It is the fundamental unit of state,
//     responsible for atomically tracking statistics (e.g., length and byte size) and ensuring data consistency.
//
// # Concurrency Model
//
// The registry uses a multi-layered strategy to maximize performance on the hot path while ensuring correctness for
// administrative tasks.
// (See the `FlowRegistry` struct documentation for detailed locking rules).
package registry
