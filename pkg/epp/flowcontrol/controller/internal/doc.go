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

// Package internal provides the core worker implementation for the `controller.FlowController`.
//
// The components in this package are the private, internal building blocks of the `controller` package. This separation
// enforces a clean public API at the `controller` level and allows the internal mechanics of the engine to evolve
// independently.
//
// # Design Philosophy: A Single-Writer Actor Model
//
// The concurrency model for this package is deliberately built around a single-writer, channel-based actor pattern, as
// implemented in the `ShardProcessor`. While a simple lock-based approach might seem easier, it is insufficient for the
// system's requirements. The "enqueue" operation is a complex, stateful transaction that requires a **hierarchical
// capacity check** against both the overall shard and a request's specific priority band.
//
// A coarse, shard-wide lock would be required to make this transaction atomic, creating a major performance bottleneck
// and causing head-of-line blocking at the top-level `controller.FlowController`. The single-writer model, where all
// state mutations are funneled through a single goroutine, makes this transaction atomic *without locks*.
//
// This design provides two critical benefits:
//  1. **Decoupling:** The `controller.FlowController` is decoupled via a non-blocking channel send, allowing for much
//     higher throughput.
//  2. **Backpressure:** The state of the channel buffer serves as a high-fidelity, real-time backpressure signal,
//     enabling more intelligent load balancing.
//
// # Future-Proofing for Complex Transactions
//
// This model's true power is that it provides a robust foundation for future features like **displacement** (a
// high-priority item evicting lower-priority ones). This is an "all-or-nothing" atomic transaction that is
// exceptionally difficult to implement correctly in a lock-free or coarse-grained locking model without significant
// performance penalties. The single-writer model contains the performance cost of such a potentially long transaction
// to the single `ShardProcessor`, preventing it from blocking the entire `controller.FlowController`.
package internal
