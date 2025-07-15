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
// The primary interfaces defined here are:
//   - `SafeQueue`: The contract for concurrent-safe queue implementations.
//   - `ItemComparator`: The contract for policy-driven logic that defines the relative priority of items within a
//     queue.
package framework
