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

// Package internal provides the core worker implementation for the controller.FlowController.
//
// The components in this package are the private, internal building blocks of the controller. This separation enforces
// a clean public API at the `controller` level and allows the internal mechanics of the engine to evolve independently.
//
// # Design Philosophy: The Single-Writer Actor Model
//
// The concurrency model for this package is built around a single-writer, channel-based actor pattern, as implemented
// in the ShardProcessor. All state-mutating operations for a given shard (primarily enqueuing new requests) are
// funneled through a single Run goroutine.
//
// This design makes complex, multi-step transactions (like a hierarchical capacity check against both a shard's total
// limit and a priority band's limit) inherently atomic without locks. This avoids the performance bottleneck of a
// coarse, shard-wide lock and allows the top-level Controller to remain decoupled and highly concurrent.
//
// # Key Components
//
//   - ShardProcessor: The implementation of the worker actor. Manages the lifecycle of requests for a single shard.
//   - FlowItem: The internal representation of a request, managing its state and synchronization across goroutines.
package internal
