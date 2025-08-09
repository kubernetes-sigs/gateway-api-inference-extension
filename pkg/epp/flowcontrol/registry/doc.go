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
// As the stateful control plane for the entire Flow Control system, this package is responsible for managing the
// lifecycle of all flows, queues, and policies. It provides a sharded, concurrent-safe view of its state to the
// `controller.FlowController` workers, enabling scalable, parallel request processing.
//
// # Architecture: Composite, Sharded, and Separated Concerns
//
// The registry employs a composite architecture designed to separate the control plane (orchestration) from the data
// plane (request processing state).
//
//   - `FlowRegistry`: The Control Plane. The top-level orchestrator and single source of truth. It manages the
//     lifecycle of its child shards and centralizes all complex administrative operations, such as flow registration,
//     garbage collection coordination, and shard scaling.
//
//   - `registryShard`: The Data Plane Slice. A single, concurrent-safe "slice" of the registry's total state. It is the
//     primary object that a `controller.FlowController` worker interacts with, providing a simplified, read-optimized
//     view of the queues and policies it needs to operate.
//
//   - `managedQueue`: The Stateful Decorator. A wrapper around a generic `framework.SafeQueue`. It augments the queue
//     with registry-specific concerns: atomic statistics tracking and lifecycle enforcement (active vs. draining).
//
// # Concurrency Model: Multi-Tiered Locking and the Actor Pattern
//
// The registry employs a multi-tiered concurrency strategy to maximize performance on the hot path while ensuring
// strict correctness for complex state transitions:
//
//  1. Serialized Control Plane (Actor Model): The `FlowRegistry` utilizes an Actor-like pattern for its control plane.
//     A single background goroutine processes all state change events (e.g., GC timers, queue emptiness signals) from a
//     channel. This serializes all mutations to the registry's core state, eliminating a significant class of race
//     conditions and simplifying the complex logic of distributed garbage collection and flow lifecycles.
//
//  2. Coarse-Grained Admin Lock: A single `sync.Mutex` on the `FlowRegistry` protects administrative operations (e.g.,
//     `RegisterOrUpdateFlow`). This lock is acquired both by external callers and the internal event loop, ensuring
//     that these complex, multi-step state changes appear atomic.
//
//  3. Shard-Level R/W Lock: Each `registryShard` uses a `sync.RWMutex` to protect its internal maps. This allows
//     multiple `controller.FlowController` workers to read from the same shard in parallel.
//
//  4. Lock-Free Data Path (Atomics): All statistics (queue length, byte size) at all layers are implemented using
//     `atomic.Uint64`. This allows for high-performance, lock-free updates on the "data plane" hot path, where every
//     request modifies these counters.
//
// # The "Trust but Verify" Garbage Collection Pattern
//
// A critical aspect of the registry's design is the race condition between the asynchronous data path (e.g., a queue
// becoming non-empty) and the control plane's destructive operations (garbage collection). The `flowState` object in
// the control plane is an eventually consistent cache of the system's state. Relying on this cached view for a
// destructive action could lead to incorrect behavior, for instance, if a GC timer fires and is processed before the
// activity event that should have cancelled it.
//
// To solve this without sacrificing performance via synchronous locking on the hot path, the registry employs a
// "Trust but Verify" pattern for all garbage collection decisions:
//
//  1. Trust: The control plane first "trusts" its cached `flowState` to make a preliminary, non-destructive decision
//     (e.g., "the flow appears to be idle").
//
//  2. Verify: Before committing to the destructive action (deleting the flow), the control plane performs a "verify"
//     step. It synchronously queries the ground truth—the atomic counters on the live `managedQueue` instances across
//     all relevant shards.
//
// This pattern provides the necessary strong consistency for critical operations precisely when needed, incurring the
// overhead of the live check only during the GC process (which is off the hot path), thereby maintaining high
// performance on the request path.
//
// # Event-Driven State Machine and Dynamic Updates
//
// The registry is designed to handle dynamic configuration changes gracefully and correctly. The interplay between
// state transitions, the event-driven control plane, and the garbage collection (GC) system is critical to its
// robustness.
//
// The system relies on atomic state transitions to generate reliable, edge-triggered signals. Components (queues,
// shards) use atomic state transitions (e.g., transitioning from Draining to Drained) to signal the control plane
// exactly once when a critical event occurs. These signals are sent reliably over the event channel; if the channel is
// full, the sender blocks, applying necessary backpressure to ensure no events are lost, which is vital for preventing
// state divergence and memory leaks.
//
// The following scenarios detail how the registry handles various lifecycle events:
//
// New Flow Registration: A new flow `F` is registered at priority `P1`.
//
//  1. A new `managedQueue` (`Q1`) is created on all shards and marked Active.
//  2. The control plane (`FlowRegistry`) starts inactivity GC tracking. If `Q1` remains empty globally for
//     `FlowGCTimeout`, flow `F` is automatically unregistered.
//
// Flow Activity/Inactivity: A flow transitions between having requests and being empty.
//
//  1. When the first request is enqueued, `Q1` signals `QueueBecameNonEmpty`. The control plane stops the GC timer.
//  2. When the last request is dispatched globally, `Q1` signals `QueueBecameEmpty`. The control plane starts the GC
//     timer.
//
// Flow Priority Change: Flow `F` changes from priority `P1` to `P2`.
//
//  1. The existing queue (`Q1`) at `P1` transitions to Draining (stops accepting new requests).
//  2. A new queue (`Q2`) at `P2` is created and marked Active.
//  3. Inactivity GC tracking starts for `Q2`.
//  4. When `Q1` becomes empty globally, it transitions to Drained and signals `QueueBecameDrainedAndEmpty`. The control
//     plane garbage collects Q1 instances.
//
// Draining Reactivation: Flow `F` changes `P1` -> `P2`, then quickly back `P2` -> `P1` before `Q1` is empty.
//
//  1. (`P1`->`P2`): `Q1` is Draining, `Q2` is Active.
//  2. (`P2`->`P1`): The system finds `Q1` and atomically transitions it back to Active. `Q2` transitions to Draining.
//  3. This optimization avoids unnecessary object churn. GC tracking is correctly updated.
//
// Shard Scale-Up: New shards are added.
//
//  1. New shards are created.
//  2. The `FlowRegistry` synchronizes all existing flows onto the new shards.
//  3. GC tracking state is initialized for these new queue instances.
//  4. Configuration (e.g., capacity limits) is re-partitioned across all active shards.
//
// Shard Scale-Down: The shard count is reduced.
//
//  1. Targeted shards transition to Draining.
//  2. Configuration is re-partitioned across the remaining active shards.
//  3. When a draining shard becomes completely empty, it transitions to Drained and signals `ShardDrainedAndEmpty`.
//  4. The control plane removes the shard and purges its ID from all flow tracking maps to prevent memory leaks.
package registry
