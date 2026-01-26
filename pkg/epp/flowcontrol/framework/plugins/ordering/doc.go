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

// Package ordering provides the standard implementations of the OrderingPolicy interface.
//
// # Context: The 3-Tier Dispatch Hierarchy
//
// The Flow Control system manages traffic using a strict three-tier decision hierarchy.
// This package implements Tier 3.
//
//  1. Priority (Band Selection):
//     The system first strictly selects the highest-priority Band that has pending work.
//
//  2. Fairness (Flow Selection):
//     Once a Band is selected, the FairnessPolicy (interflow) determines which Flow within that band gets the next
//     dispatch opportunity.
//
//  3. Ordering (Item Selection) - [THIS PACKAGE]:
//     Once a Flow is selected, the OrderingPolicy determines which Request from that specific flow's queue is
//     dispatched. This governs the "internal discipline" of the flow (e.g., whether to serve oldest requests first or
//     most urgent).
//
// # Architecture: The Flyweight Pattern
//
// Ordering Policies are Singletons. A single instance handles the ordering logic for all queues in a Priority Band.
// To support this efficiently, the policy follows the Flyweight pattern:
//
//  1. The Plugin Instance (e.g., FCFS) is a Singleton. It defines the Logic (Less).
//  2. The Logic (Less) acts as a pure function (or comparator) that operates on the queue's items.
//
// # Standard Implementations
//
// This package includes the following core strategies:
//
//   - FCFS ("First-Come, First-Served") ("fcfs-ordering-policy"): Orders requests by their logical arrival time.
//     This is the default and most "intuitive" ordering.
//
//   - EDF ("Earliest Deadline First") ("edf-ordering-policy"): Orders requests by their absolute deadline
//     (EnqueueTime + TTL).
//     This maximizes the number of requests served before their deadlines expire.
package ordering
