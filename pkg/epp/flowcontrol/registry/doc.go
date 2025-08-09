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

// Package registry provides the concrete implementation of the Flow Registry.
//
// As the stateful control plane for the entire Flow Control system, this package is responsible for managing the
// lifecycle of all flows, queues, and policies. It serves as the "adapter" that implements the service "ports"
// (interfaces) defined in the `contracts` package. It provides a sharded, concurrent-safe view of its state to the
// `controller.FlowController` workers, enabling scalable, parallel request processing.
//
// # Key Components
//
//   - `FlowRegistry`: The future top-level administrative object that will manage the entire system, including shard
//     lifecycles and flow registration. (Not yet implemented).
//
//   - `registryShard`: A concrete implementation of the `contracts.RegistryShard` interface. It represents a single,
//     concurrent-safe slice of the registry's state, containing a set of priority bands and the flow queues within
//     them. This is the primary object a `controller.FlowController` worker interacts with.
//
//   - `managedQueue`: A concrete implementation of the `contracts.ManagedQueue` interface. It acts as a stateful
//     decorator around a `framework.SafeQueue`, adding critical registry-level functionality such as atomic statistics
//     tracking.
//
//   - `Config`: The top-level configuration object that defines the structure and default behaviors of the registry,
//     including the definition of priority bands and default policy selections. This configuration is partitioned and
//     distributed to each `registryShard`.
package registry
