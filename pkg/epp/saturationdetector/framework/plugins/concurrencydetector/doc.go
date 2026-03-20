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

// Package concurrencydetector implements a synchronous saturation detection and scheduling filter
// mechanism based on active in-flight request accounting.
//
// # Role in Flow Control (The Gatekeeper)
//
// The detector implements the SaturationDetector interface to provide a utilization gradient,
// allowing the Flow Controller to apply proportional backpressure.
//
//	PoolSaturation = Aggregate Inflight Load / Aggregate Pool Capacity
//
// In token mode, both numerator and denominator are evaluated in tokens: the aggregate inflight
// token count divided by the sum of all endpoints' MaxTokenConcurrency.
//
// # Heterogeneous Deployments
//
// Because this detector calculates saturation globally as a single aggregate fraction, it utilizes
// an aggregate queueing model. In deployments with heterogeneous compute (e.g., mixing H100 and
// L4 nodes), this heavily biases the pool saturation metric toward the state of the larger nodes.
// Contrast this with the Utilization Detector, which evaluates saturation as an unweighted average
// of individual endpoint scores.
//
// # Role in Scheduling (The Traffic Shaper)
//
// The detector implements the Filter interface to protect individual endpoints.
// It removes endpoints from candidate lists if their local inflight count exceeds the safety limit:
//
//	EndpointLimit = Capacity * (1 + Headroom)
//
// This two-tier approach allows the Flow Controller to manage average pool load, while the
// Scheduler retains the flexibility to burst above ideal targets (the "Headroom") to satisfy
// affinity or scoring objectives.
//
// # Trade-offs
//
// Unlike the Utilization Detector, this approach reacts instantaneously to new requests, preventing
// sudden bursts from overwhelming an endpoint before telemetry updates. However, it suffers from
// two critical flaws:
//
//  1. Open-Loop Divergence: The detector operates as an open-loop controller, completely blind
//     to actual hardware telemetry. While the internal counters are mathematically zero-sum and
//     do not leak, consistent under- or over-estimations of token lengths will cause the view of
//     pool saturation to systematically drift from the physical reality of the GPU workload.
//  2. KV Cache Blindness: Because the detector cannot observe true engine memory pressure, it is
//     highly vulnerable to continuous-batching edge cases. If actual output generations exceed
//     static estimates, the underlying KV cache will silently fill up. This forces the inference
//     engine to preempt active requests and swap KV blocks to CPU memory, causing severe latency
//     degradation (TPOT spikes) that remains completely invisible to this detector.
package concurrencydetector
