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

// Package framework defines the core plugin interfaces for extending the `controller.FlowController`.
//
// It establishes the contracts that custom logic, such as queueing disciplines and dispatching policies, must adhere
// to. By building on these interfaces, the Flow Control system can be extended and customized without modifying the
// core controller logic.
//
// The primary contracts are:
//   - `SafeQueue`: An interface for concurrent-safe queue implementations.
//   - `IntraFlowDispatchPolicy`: An interface for policies that decide which item to select from within a single flow's
//     queue.
//   - `ItemComparator`: An interface vended by policies to make their internal item-ordering logic explicit and
//     available to other components.
//
// These components are linked by `QueueCapability`, which allows policies to declare their queue requirements (e.g.,
// FIFO or priority-based ordering).
package framework
