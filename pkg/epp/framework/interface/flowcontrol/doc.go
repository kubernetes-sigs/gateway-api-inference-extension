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

// Package flowcontrol defines the core plugin interfaces for extending the Flow Control layer.
//
// It establishes the contracts that custom logic, such as queueing disciplines and dispatching policies, must adhere
// to. By building on these interfaces, the Flow Control layer can be extended and customized without modifying the
// core controller logic.
//
// The primary contracts are:
//   - SafeQueue: An interface for concurrent-safe queue implementations.
//   - FairnessPolicy: The interface for policies that govern the competition between flows.
//   - OrderingPolicy: The interface for policies that decide the strict sequence of service within a flow.
//   - IntraFlowDispatchPolicy: (Deprecated) Legacy interface for intra-flow ordering. Replaced by OrderingPolicy.
//   - ItemComparator: (Deprecated) Legacy interface for exposing ordering logic. Replaced by OrderingPolicy.
//
// These components are linked by QueueCapability, which allows policies to declare their queue requirements (e.g.,
// FIFO or priority-based ordering).
package flowcontrol
