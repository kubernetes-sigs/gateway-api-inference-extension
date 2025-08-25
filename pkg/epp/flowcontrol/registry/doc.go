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

// Package registry provides the concrete implementation of the `contracts.FlowRegistry`.
//
// As the stateful control plane, this package manages the lifecycle of all flows, queues, and policies. It provides a
// sharded, concurrent-safe view of its state to the `controller.FlowController` workers, enabling scalable, parallel
// request processing.
//
// # Architecture: Composite, Sharded, and Separated Concerns
//
// The registry separates the control plane (orchestration) from the data plane (request processing state).
//
//   - `FlowRegistry`: The Control Plane. The top-level orchestrator and single source of truth. It centralizes complex
//     operations: flow registration, garbage collection (GC) coordination, and shard scaling.
//
//   - `registryShard`: The Data Plane Slice. A concurrent-safe "slice" of the registry's total state. It provides a
//     read-optimized view for `FlowController` workers.
//
//   - `managedQueue`: The Stateful Decorator. A wrapper around a `framework.SafeQueue`. It augments the queue with
//     atomic statistics tracking and signaling state transitions to the control plane.
//
// # Data Flow and Interaction Model
//
// The data path (Enqueue/Dispatch) is optimized for minimal latency and maximum concurrency.
//
// Enqueue Path:
//  1. The `FlowController`'s distributor selects an active `registryShard`.
//  2. The distributor calls `shard.ManagedQueue(flowKey)` (acquires `RLock`).
//  3. The distributor calls `managedQueue.Add(item)`.
//  4. `managedQueue` atomically updates the queue and its stats and signals the control plane.
//
// Dispatch Path:
//  1. A `FlowController` worker iterates over its assigned `registryShard`.
//  2. The worker uses policies and accessors to select the next item (acquires `RLock`).
//  3. The worker calls `managedQueue.Remove(handle)`.
//  4. `managedQueue` atomically updates the queue and its stats and signals the control plane.
//
// # Concurrency Strategy: Multi-Tiered and Optimized
//
// The registry maximizes performance on the hot path while ensuring strict correctness for complex state transitions:
//
//   - Serialized Control Plane (Actor Model): The `FlowRegistry` uses a single background goroutine to process all
//     state change events serially, eliminating race conditions in the control plane.
//
//   - Sharding (Data Plane Parallelism): State is partitioned across multiple `registryShard` instances, allowing the
//     data path to scale linearly.
//
//   - Lock-Free Data Path (Atomics): Statistics aggregation (Shard/Registry level) uses lock-free atomics.
//
//   - Strict Consistency (Hybrid Locking): `managedQueue` uses a hybrid locking model (Mutex for writes, Atomics for
//     reads) to guarantee strict consistency between queue contents and statistics, which is required for GC
//     correctness.
//
// (See the `FlowRegistry` struct documentation for detailed locking rules).
//
// # Garbage Collection: The "Trust but Verify" Pattern
//
// The registry handles the race condition between asynchronous data path activity and synchronous GC. The control plane
// maintains an eventually consistent cache (`flowState`).
//
// The registry uses a periodic, generational "Trust but Verify" pattern. It identifies candidate flows using the cache.
// Before deletion, it performs a "Verify" step: it synchronously acquires write locks on ALL shards and queries the
// ground truth (live queue counters). This provides strong consistency when needed.
//
// (See the `garbageCollectFlowLocked` function documentation for detailed steps).
//
// # Scalability Characteristics and Trade-offs
//
// The architecture prioritizes data path throughput and correctness, introducing specific trade-offs:
//
//   - Data Path Throughput (Excellent): Scales linearly with the number of shards and benefits from lock-free
//     statistics updates.
//
//   - GC Latency Impact (Trade-off): The GC "Verify" step requires locking all shards (O(N)). This briefly pauses the
//     data path. As the shard count (N) increases, this may impact P99 latency. This trade-off guarantees correctness.
//
//   - Control Plane Responsiveness during Scale-Up (Trade-off): Scaling up requires synchronizing all existing flows
//     (M) onto the new shards (K). This O(M*K) operation occurs under the main control plane lock. If M is large, this
//     operation may block the control plane.
//
// # Event-Driven State Machine and Lifecycle Scenarios
//
// The system relies on atomic state transitions to generate reliable, edge-triggered signals. These signals are sent
// reliably; if the event channel is full, the sender blocks, applying necessary backpressure to ensure no events are
// lost, preventing state divergence.
//
// The following scenarios detail how the registry handles lifecycle events:
//
// New Flow Registration: A new flow instance `F1` (`FlowKey{ID: "A", Priority: 10}`) is registered.
//
//  1. `managedQueue` instances are created for `F1` on all shards.
//  2. The `flowState` cache marks `F1` as Idle. If it remains Idle, it will eventually be garbage collected.
//
// Flow Activity/Inactivity:
//
//  1. When the first request for `F1` is enqueued, the queue signals `BecameNonEmpty`. The control plane marks `F1`
//     as Active, protecting it from GC.
//  2. When the last request is dispatched globally, the queues signal `BecameEmpty`. The control plane updates the
//     cache, and `F1` is now considered Idle by the GC scanner.
//
// "Changing" Flow Priority: Traffic for `ID: "A"` needs to shift from `Priority: 10` to `Priority: 20`.
//
//  1. The caller registers a new flow instance, `F2` (`FlowKey{ID: "A", Priority: 20}`).
//  2. The system treats `F1` and `F2` as independent entities (Immutable `FlowKey` design).
//  3. As `F1` becomes Idle, it is automatically garbage collected. This achieves the outcome gracefully without complex
//     state migration logic.
//
// Shard Scaling:
//
//   - Scale-Up: New shards are created and marked Active. Existing flows are synchronized onto the new shards.
//     Configuration is re-partitioned.
//   - Scale-Down: Targeted shards transition to Draining (stop accepting new work). Configuration is re-partitioned
//     across remaining active shards. When a Draining shard is empty, it signals `BecameDrained` and is removed by the
//     control plane.
package registry
