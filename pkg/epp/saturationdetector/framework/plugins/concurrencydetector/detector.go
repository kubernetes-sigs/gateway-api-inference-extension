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
// It signals saturation when every available candidate endpoint has reached the configured MaxConcurrency limit.
// This indicates that the backend pool has no remaining capacity for new work, triggering the Flow Controller to queue
// incoming requests.
//
// # Role in Scheduling (The Traffic Shaper)
//
// The Detector implements the Filter interface to protect individual endpoints.
// It removes endpoints from candidate lists if they exceed the specific safety limit:
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
// internal counters for a endpoint will drift upwards. This can lead to a "false saturated" state where the detector
// believes a endpoint is full when it is actually empty.
//
// Currently, the only mechanism to reset a drifted counter is the DeleteEndpoint signal (when a backend is removed from the
// pool). Future iterations may require a reconciliation loop or a TTL-based cleanup to recover from persistent drift.
package concurrencydetector

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
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
// It iterates through the provided list of candidate endpoints. If it finds at least one endpoint where the current
// in-flight requests are below the MaxConcurrency threshold, it returns false (not saturated), allowing the Flow
// Controller to admit the request.
//
// If all candidate endpoints are at or above the MaxConcurrency limit, it returns true, signaling the Flow Controller
// to halt dispatch and queue incoming requests.
func (d *Detector) IsSaturated(ctx context.Context, candidateEndpoints []metrics.PodMetrics) bool {
	if len(candidateEndpoints) == 0 {
		return true
	}

	for _, endpoint := range candidateEndpoints {
		if endpoint.GetMetadata() == nil {
			continue
		}

		endpointID := endpoint.GetMetadata().NamespacedName.String()
		inflight := d.tracker.get(endpointID)
		if inflight < d.config.MaxConcurrency {
			return false
		}
	}
	return true
}

// Filter blocks traffic to specific endpoints that are physically saturated or exceeding their safety limits.
//
// It applies a relaxed limit (MaxConcurrency * (1 + Headroom)) to allow for scheduling flexibility and burst tolerance.
func (d *Detector) Filter(
	_ context.Context,
	_ *types.CycleState,
	_ *types.LLMRequest,
	endpoints []types.Endpoint,
) []types.Endpoint {
	limit := int64(float64(d.config.MaxConcurrency) * (1.0 + d.config.Headroom))

	// Pre-allocate assuming most endpoints will pass the filter to minimize allocations.
	filtered := make([]types.Endpoint, 0, len(endpoints))

	for _, endpoint := range endpoints {
		endpointID := endpoint.GetMetadata().NamespacedName.String()
		if d.tracker.get(endpointID) <= limit {
			filtered = append(filtered, endpoint)
		}
	}
	return filtered
}

// PreRequest increments the atomic in-flight counter for the target endpoint.
// We assume the scheduling result is valid based on the Director's contract.
func (d *Detector) PreRequest(_ context.Context, _ *types.LLMRequest, result *types.SchedulingResult) {
	d.tracker.inc(result.ProfileResults[result.PrimaryProfileName].TargetEndpoints[0].GetMetadata().NamespacedName.String())
}

// ResponseComplete decrements the atomic in-flight counter for the target endpoint.
func (d *Detector) ResponseComplete(
	_ context.Context,
	_ *types.LLMRequest,
	_ *requestcontrol.Response,
	targetEndpoint *datalayer.EndpointMetadata,
) {
	d.tracker.dec(targetEndpoint.NamespacedName.String())
}

// DeleteEndpoint removes an endpoint from the concurrency tracker to prevent memory leaks.
// This should be called by the controller when a backend is removed from the pool.
func (d *Detector) DeleteEndpoint(endpointID string) {
	d.tracker.delete(endpointID)
}

// concurrencyTracker manages thread-safe counters for inflight requests.
// It is optimized for a read-heavy workload.
type concurrencyTracker struct {
	mu sync.RWMutex
	// counts stores the inflight count per endpoint ID.
	// We use *atomic.Int64 to allow safe concurrent updates without holding the map lock.
	counts map[string]*atomic.Int64
}

func newConcurrencyTracker() *concurrencyTracker {
	return &concurrencyTracker{
		counts: make(map[string]*atomic.Int64),
	}
}

// get returns the current inflight count for the given endpoint.
// It returns 0 if the endpoint is not tracked.
func (ct *concurrencyTracker) get(endpointID string) int64 {
	ct.mu.RLock()
	counter, exists := ct.counts[endpointID]
	ct.mu.RUnlock()

	if !exists {
		return 0
	}
	return counter.Load()
}

// inc increments the inflight count for the given endpoint.
// It creates the counter if it does not exist.
func (ct *concurrencyTracker) inc(endpointID string) {
	// Fast path: Try with read lock first.
	ct.mu.RLock()
	counter, exists := ct.counts[endpointID]
	ct.mu.RUnlock()

	if exists {
		counter.Add(1)
		return
	}

	// Slow path: Create counter with write lock.
	ct.mu.Lock()
	defer ct.mu.Unlock()

	// Double-check existence to handle race conditions.
	if counter, exists = ct.counts[endpointID]; exists {
		counter.Add(1)
		return
	}

	counter = &atomic.Int64{}
	counter.Store(1)
	ct.counts[endpointID] = counter
}

// dec decrements the inflight count for the given endpoint.
func (ct *concurrencyTracker) dec(endpointID string) {
	ct.mu.RLock()
	counter, exists := ct.counts[endpointID]
	ct.mu.RUnlock()

	if exists {
		counter.Add(-1)
	}
	// If it doesn't exist, we silently ignore.
	// This can happen if a endpoint was deleted/garbage collected while a request was inflight.
}

// delete removes the counter for the given endpoint.
func (ct *concurrencyTracker) delete(endpointID string) {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	delete(ct.counts, endpointID)
}
