/*
Copyright 2026 The Kubernetes Authors.

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

// Package utilizationdetector implements a reactive saturation detection and scheduling filter
// mechanism based on telemetry from LLM serving backends.
//
// # Role in Flow Control (The Gatekeeper)
//
// The detector implements the SaturationDetector interface to provide a utilization gradient,
// allowing the Flow Controller to apply proportional backpressure when the system is overloaded.
// It relies on a "roofline model", evaluating both the queue depth and the KV cache utilization
// to find the most constrained resource for a given endpoint:
//
//	EndpointScore = max(QueueDepth / QueueThreshold, KVCacheUsage / KVCacheThreshold)
//
// The global pool saturation is then evaluated across all candidate endpoints as a gradient:
//
//	PoolSaturation = Average(EndpointScore)
//
// # Heterogeneous Deployments
//
// Because this detector calculates saturation as an unweighted average of individual endpoint
// scores, it treats all endpoints equally regardless of their physical capacity. In deployments
// with heterogeneous compute, a small, saturated endpoint has the exact same impact on global
// backpressure as a massive, saturated endpoint. Contrast this with the Concurrency Detector, which
// evaluates saturation as a single aggregate fraction, biasing toward larger endpoints.
//
// # Role in Scheduling (The Traffic Shaper)
//
// The detector implements the Filter interface to protect individual endpoints.
// It removes endpoints from candidate lists if their telemetry exceeds the specific safety limits:
//
//	MaxQueueLimit = QueueThreshold * (1 + Headroom)
//	MaxKVCacheLimit = min(1.0, KVCacheThreshold * (1 + Headroom))
//
// This two-tier approach allows the Flow Controller to manage average pool load, while the
// Scheduler retains the flexibility to burst above ideal targets (the "Headroom") to satisfy
// affinity or scoring objectives.
//
// # Trade-offs
//
// Unlike the Concurrency Detector, this approach operates as a closed-loop controller. It is
// immune to estimation divergence and reflects the actual performance limits of the continuous
// batching engine's memory manager. However, it suffers from two fundamental flaws of reactive
// systems:
//
//  1. Telemetry Staleness (The Thundering Herd): Because it relies on asynchronous polling, the
//     view of the endpoint state is perpetually delayed. A sudden burst of traffic can create a
//     severe "thundering herd" condition, where the scheduler routes massive request volumes to
//     a seemingly healthy endpoint before the next metric interval reveals it is completely
//     saturated.
//  2. Reactive Backpressure: By definition, this detector only signals saturation after the
//     inference engine is already under physical duress. It cannot preemptively shield an
//     endpoint from an initial queue buildup; it can only throttle traffic after the physical
//     limits have been breached and latency (TTFT/TPOT) has already degraded.
package utilizationdetector
