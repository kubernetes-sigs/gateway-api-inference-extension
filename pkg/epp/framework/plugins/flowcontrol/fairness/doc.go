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

// Package fairness provides the standard implementations of the FairnessPolicy interface.
//
// # Context: The 3-Tier Dispatch Hierarchy
//
// The Flow Control system manages traffic using a strict three-tier decision hierarchy.
// This package implements Tier 2.
//
//  1. Priority (Band Selection):
//     The system first strictly selects the highest-priority Band that has pending work.
//     Higher priority bands always starve lower priority bands.
//
//  2. Fairness (Flow Selection) - [THIS PACKAGE]:
//     Once a Band is selected, the FairnessPolicy determines which Flow within that band gets the next dispatch
//     opportunity.
//     This manages contention between tenants or models sharing the same priority level.
//
//  3. Ordering (Item Selection):
//     Once a Flow is selected, the IntraFlowDispatchPolicy (e.g., FCFS) determines which Request from that specific
//     flow's queue is dispatched.
//
// # Architecture: The Flyweight Pattern
//
// Fairness Policies must often maintain state (e.g., Round Robin cursors) for each Priority Band they govern.
// To support this efficiently without creating a new plugin instance for every Band, this package uses the Flyweight
// pattern:
//
//  1. The Plugin Instance (e.g., RoundRobin) is a Singleton. It contains only immutable configuration.
//  2. The State (e.g., the cursor) is created via the NewState() method and stored on the Band.
//  3. The Logic (Pick method) accepts the State as an argument during execution.
//
// # Standard Implementations
//
// This package includes the following core strategies:
//
//   - Round Robin ("round-robin-fairness-policy"): Cycles through active flows one by one.
//     This guarantees that no single flow can starve others, regardless of its volume.
//     It is "Work Conserving" (it skips empty queues).
//
//   - Global Strict ("global-strict-fairness-policy"): A greedy strategy that ignores Flow boundaries.
//     It scans all queues in the band and picks the absolute "best" request (e.g., oldest timestamp) globally.
//     This maximizes strict adherence to global ordering but offers no isolation; a noisy neighbor can starve other
//     flows.
package fairness
