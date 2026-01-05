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

// Package concurrencydetector implements a real-time saturation detection and scheduling filter mechanism based on
// active in-flight request accounting.
//
// # Role in Flow Control (The Gatekeeper)
//
// The Detector implements the SaturationDetector interface to act as a "Circuit Breaker".
// It signals saturation when every available candidate pod has reached the configured MaxConcurrency limit.
// This indicates that the backend pool has no remaining capacity for new work, triggering the Flow Controller to queue
// incoming requests.
//
// # Role in Scheduling (The Traffic Shaper)
//
// The Detector implements the Filter interface to protect individual pods.
// It removes pods from candidate lists if they exceed the specific safety limit:
//
//	Limit = MaxConcurrency * (1 + Headroom)
//
// This two-tier approach allows the Flow Controller to manage average pool load, while the Scheduler retains the
// flexibility to burst slightly above ideal targets (the "Headroom") to satisfy affinity or scoring objectives.
//
// # Consistency & Drift Warning
//
// The Detector relies on a strict symmetry between PreRequest (increment) and ResponseComplete (decrement) calls.
// It assumes the EPP framework guarantees that every PreRequest is eventually paired with a ResponseComplete.
//
// If the application panics, crashes, or if the framework fails to invoke the ompletion hook for a request, the
// internal counters for a pod will drift upwards. This can lead to a "false saturated" state where the detector
// believes a pod is full when it is actually empty.
//
// Currently, the only mechanism to reset a drifted counter is the DeletePod signal (when a backend is removed from the
// pool). Future iterations may require a reconciliation loop or a TTL-based cleanup to recover from persistent drift.
package concurrencydetector

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const ConcurrencyDetectorType = "concurrency-detector"

func init() {
	plugins.Register(ConcurrencyDetectorType, func(_ string, params json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
		var cfg Config
		if len(params) > 0 {
			if err := json.Unmarshal(params, &cfg); err != nil {
				return nil, fmt.Errorf("failed to unmarshal concurrency detector config: %w", err)
			}
		}
		return NewDetector(cfg), nil
	})
}

var (
	_ requestcontrol.PreRequest       = &Detector{}
	_ requestcontrol.ResponseComplete = &Detector{}
	_ framework.Filter                = &Detector{}
)

// Detector implements a saturation detector and scheduling filter based on active request concurrency.
type Detector struct {
	tracker *concurrencyTracker
	config  Config
}

// NewDetector creates a new instance of the Concurrency Detector.
func NewDetector(config Config) *Detector {
	// TODO: Replace with more robust validation and defaulting logic once Saturation Detector becomes an official
	// extension point.
	if config.MaxConcurrency <= 0 {
		config.MaxConcurrency = DefaultMaxConcurrency
	}
	if config.Headroom < 0 {
		config.Headroom = DefaultHeadroom
	}

	return &Detector{
		tracker: newConcurrencyTracker(),
		config:  config,
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (d *Detector) TypedName() plugins.TypedName {
	return plugins.TypedName{
		Type: ConcurrencyDetectorType,
		Name: ConcurrencyDetectorType,
	}
}

// IsSaturated acts as the global circuit breaker.
//
// It iterates through the provided list of candidate pods. If it finds at least one pod where the current in-flight
// requests are below the MaxConcurrency threshold, it returns false (not saturated), allowing the Flow Controller to
// admit the request.
//
// If all candidate pods are at or above the MaxConcurrency limit, it returns true, signaling the Flow Controller to
// halt dispatch and queue incoming requests.
func (d *Detector) IsSaturated(ctx context.Context, candidatePods []metrics.PodMetrics) bool {
	if len(candidatePods) == 0 {
		return true
	}

	for _, pod := range candidatePods {
		if pod.GetMetadata() == nil {
			continue
		}

		podID := pod.GetMetadata().NamespacedName.String()
		inflight := d.tracker.get(podID)
		if inflight < d.config.MaxConcurrency {
			return false
		}
	}
	return true
}

// Filter blocks traffic to specific pods that are physically saturated or exceeding their safety limits.
//
// It applies a relaxed limit (MaxConcurrency * (1 + Headroom)) to allow for scheduling flexibility and burst tolerance.
func (d *Detector) Filter(
	_ context.Context,
	_ *types.CycleState,
	_ *types.LLMRequest,
	pods []types.Pod,
) []types.Pod {
	limit := int64(float64(d.config.MaxConcurrency) * (1.0 + d.config.Headroom))

	// Pre-allocate assuming most pods will pass the filter to minimize allocations.
	filtered := make([]types.Pod, 0, len(pods))

	for _, pod := range pods {
		podID := pod.GetPod().NamespacedName.String()
		if d.tracker.get(podID) <= limit {
			filtered = append(filtered, pod)
		}
	}
	return filtered
}

// PreRequest increments the atomic in-flight counter for the target pod.
// We assume the scheduling result is valid based on the Director's contract.
func (d *Detector) PreRequest(_ context.Context, _ *types.LLMRequest, result *types.SchedulingResult) {
	d.tracker.inc(result.ProfileResults[result.PrimaryProfileName].TargetPods[0].GetPod().NamespacedName.String())
}

// ResponseComplete decrements the atomic in-flight counter for the target pod.
func (d *Detector) ResponseComplete(
	_ context.Context,
	_ *types.LLMRequest,
	_ *requestcontrol.Response,
	targetPod *backend.Pod,
) {
	d.tracker.dec(targetPod.NamespacedName.String())
}

// DeletePod removes a pod from the concurrency tracker to prevent memory leaks.
// This should be called by the controller when a backend is removed from the pool.
func (d *Detector) DeletePod(podID string) {
	d.tracker.delete(podID)
}

// concurrencyTracker manages thread-safe counters for inflight requests.
// It is optimized for a read-heavy workload.
type concurrencyTracker struct {
	mu sync.RWMutex
	// counts stores the inflight count per pod ID.
	// We use *atomic.Int64 to allow safe concurrent updates without holding the map lock.
	counts map[string]*atomic.Int64
}

func newConcurrencyTracker() *concurrencyTracker {
	return &concurrencyTracker{
		counts: make(map[string]*atomic.Int64),
	}
}

// get returns the current inflight count for the given pod.
// It returns 0 if the pod is not tracked.
func (ct *concurrencyTracker) get(podID string) int64 {
	ct.mu.RLock()
	counter, exists := ct.counts[podID]
	ct.mu.RUnlock()

	if !exists {
		return 0
	}
	return counter.Load()
}

// inc increments the inflight count for the given pod.
// It creates the counter if it does not exist.
func (ct *concurrencyTracker) inc(podID string) {
	// Fast path: Try with read lock first.
	ct.mu.RLock()
	counter, exists := ct.counts[podID]
	ct.mu.RUnlock()

	if exists {
		counter.Add(1)
		return
	}

	// Slow path: Create counter with write lock.
	ct.mu.Lock()
	defer ct.mu.Unlock()

	// Double-check existence to handle race conditions.
	if counter, exists = ct.counts[podID]; exists {
		counter.Add(1)
		return
	}

	counter = &atomic.Int64{}
	counter.Store(1)
	ct.counts[podID] = counter
}

// dec decrements the inflight count for the given pod.
func (ct *concurrencyTracker) dec(podID string) {
	ct.mu.RLock()
	counter, exists := ct.counts[podID]
	ct.mu.RUnlock()

	if exists {
		counter.Add(-1)
	}
	// If it doesn't exist, we silently ignore.
	// This can happen if a pod was deleted/garbage collected while a request was inflight.
}

// delete removes the counter for the given pod.
func (ct *concurrencyTracker) delete(podID string) {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	delete(ct.counts, podID)
}
