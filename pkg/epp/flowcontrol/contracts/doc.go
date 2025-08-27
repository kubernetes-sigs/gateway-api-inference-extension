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

// Package contracts defines the boundaries and service interfaces for the Flow Control system.
//
// Adhering to a "Ports and Adapters" (Hexagonal) architectural style, these interfaces decouple the core
// `controller.FlowController` engine from its dependencies. They establish the required behaviors and system invariants
// that concrete implementations must uphold.
//
// The primary contracts are:
//
//   - `FlowRegistry`: The interface for the stateful control plane that manages the lifecycle of flows, queues, and
//     policies.
//
//   - `SaturationDetector`: The interface for a component that provides real-time load signals.
package contracts
