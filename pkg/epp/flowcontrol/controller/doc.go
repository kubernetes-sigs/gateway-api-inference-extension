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
// The `FlowController` is the central processing engine of the flow control system. It is a sharded, high-throughput
// component responsible for managing the lifecycle of all incoming requests—from initial submission via the synchronous
// `EnqueueAndWait` method to a terminal outcome (dispatch, rejection, or eviction). It achieves this by orchestrating
// its dependencies—the `contracts.FlowRegistry`, the pluggable `Policy` framework, and the
// `contracts.SaturationDetector`—to make continuous, state-aware decisions.
//
// # Architecture: The Processor-Shard Relationship
//
// The `FlowController` engine is designed around a clear separation of state and execution. This "control plane vs.
// data plane" separation is key to enabling dynamic, concurrent-safe configuration updates.
//
//   - The `contracts.FlowRegistry` is the **control plane**. It is the single source of truth for all configuration.
//     When an administrative action occurs (e.g., `RegisterOrUpdateFlow`), the `contracts.FlowRegistry` is responsible
//     for safely applying that change to each of its managed `contracts.RegistryShard` instances.
//
//   - The `contracts.RegistryShard` is the **concurrent-safe state port**. It defines the contract for a state store
//     that holds the `contracts.ManagedQueue` and framework `Policy` instances for a single shard.
//
//   - The `internal.ShardProcessor` is the **data plane worker**. It is given a single `contracts.RegistryShard` to
//     operate on. Its main `dispatchCycle` continuously acquires a read lock on the shard to get a consistent view of
//     the active queues and policies, and then executes its dispatch logic.
//
// This separation is what enables dynamic updates. The `internal.ShardProcessor` is stateless; it simply executes
// against the state presented by its `contracts.RegistryShard` on each cycle. This allows the control plane
// (`contracts.FlowRegistry`) to safely change that state in the background.
//
// # Architectural Deep Dive: The `EnqueueAndWait` Model
//
// A fundamental design choice is the synchronous, blocking `EnqueueAndWait` method. In the context of the Gateway API
// Inference ExtensionRef's Endpoint Picker (EPP), which operates as an Envoy External Processing (`ext_proc`) server, this
// model is deliberately chosen for its simplicity and robustness.
//
//   - Alignment with `ext_proc`: The `ext_proc` protocol is stream-based. A single goroutine within the EPP manages the
//     stream for a given HTTP request. `EnqueueAndWait` fits this perfectly: the request-handling goroutine calls it,
//     blocks, and upon return, has the definitive outcome. It can then immediately act on that outcome, maintaining
//     clear request-goroutine affinity.
//
//   - Simplified State Management: The state of a "waiting" request is implicitly managed by the blocked goroutine's
//     stack and its `context.Context`. The `FlowController` only needs to signal this specific goroutine to unblock it.
//
//   - Direct Backpressure: If queues are full, `EnqueueAndWait` returns `types.ErrQueueAtCapacity`. This provides
//     immediate, direct backpressure to the earliest point of contact.
//
// # Architectural Deep Dive: The Sharded Model & JSQ-Bytes
//
// The `FlowController` is built on a sharded architecture to enable parallel processing and prevent a central dispatch
// loop from becoming a bottleneck. The `FlowController` consists of a top-level manager and a pool of independent
// `internal.ShardProcessor` workers. The `contracts.FlowRegistry` guarantees that every logical flow is represented by
// a distinct queue instance on every active shard.
//
// This architecture trades deterministic global state for high throughput and scalability. The key challenge, and the
// system's most critical assumption, revolves around ensuring this distributed model can still achieve global fairness
// objectives.
//
// ## The Critical Assumption: Homogeneity Within Flows
//
// The effectiveness of the sharded model hinges on a critical assumption: while the system as a whole manages a
// heterogeneous set of flows, the traffic *within a single logical flow* is assumed to be roughly homogeneous in its
// characteristics. A logical flow is intended to represent a single workload or tenant; therefore, the most
// unpredictable variables (effecting decode behavior) are expected to be statistically similar *within* that flow.
//
// ## The Hedge: Join the Shortest Queue by Bytes (JSQ-Bytes)
//
// To make this assumption as robust as possible, the `FlowController` uses a "Join the Shortest Queue by Bytes
// (JSQ-Bytes)" algorithm. `ByteSize` is an excellent proxy for the resources the `FlowController` explicitly manages
// (host memory pressure and queuing capacity) and is also a reasonable proxy for prefill compute time.
//
// Crucially, the goal of the distributor is not to perfectly predict backend compute time, but to intelligently balance
// the load at the controller level. JSQ-Bytes achieves this by:
//
//  1. Reflecting True Load: It distributes work based on each shard's current queue size in bytes—a direct measure of
//     its memory and capacity congestion.
//
//  2. Adapting to Congestion: The byte-size of a queue is a real-time signal of a shard's overall congestion. If a
//     shard is slow (e.g., due to long-decoding downstream requests), its queues will remain full, and JSQ-Bytes will
//     adaptively steer new work away.
//
//  3. Hedging Against Assumption Violations: This adaptive, self-correcting nature makes it a powerful hedge. It
//     doesn't just distribute; it actively *load balances* based on the most relevant feedback available.
//
// # Architectural Deep Dive: Pre-Policy Gating
//
// Before policies are invoked, the `internal.ShardProcessor` applies an `internal.BandFilter`. This function determines
// which flows within a priority band are eligible for a given operation (e.g., dispatch). This pattern is a deliberate
// architectural choice to decouple the logic of *viability* from the logic of *selection*.
//
//   - An `internal.BandFilter` (e.g., the `internal.NewSaturationFilter`) determines if a flow is viable based on
//     external signals like backend load.
//   - The `framework.InterFlowDispatchPolicy` then selects from among the viable candidates based on its own fairness
//     logic.
//
// This abstraction provides two major benefits:
//
//  1. Low Contributor Burden: It makes the mental model for policy contributors significantly simpler. An author of a
//     new fairness policy does not need to be concerned with the complexities of saturation detection or other gating
//     concerns. They are given a simple, pre-filtered view of the world and can focus solely on their selection logic.
//
//  2. Correctness by Construction: The `internal.subsetPriorityBandAccessor` wrapper guarantees that a policy operates
//     on a consistent, filtered view, regardless of which accessor method it calls (`FlowIDs`, `Queue`, etc.). This
//     prevents an entire class of subtle bugs where a policy might otherwise see a stale or unfiltered view of the
//     system state.
package controller
