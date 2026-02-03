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
// The Detector implements the SaturationDetector interface to provide a utilization gradient, allowing the Flow
// Controller to apply proportional backpressure.
//
//	Saturation = Total Inflight Requests / Total MaxConcurrency Capacity
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
// If the application panics, crashes, or if the framework fails to invoke the completion hook for a request, the
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
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const ConcurrencyDetectorType = "concurrency-detector"
const ConcurrencyModeRequests = "requests"
const ConcurrencyModeTokens = "tokens"

func init() {
	fwkplugin.Register(ConcurrencyDetectorType, func(_ string, params json.RawMessage, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
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
	requestTracker *concurrencyTracker // requests in flight per endpoint
	tokenTracker   *concurrencyTracker // tokens in flight per endpoint
	tokenLedger    *tokenLedger        // requestID -> {endpointID -> tokensAdded}
	tokenEstimator TokenEstimator      // SimpleTokenEstimator with CharactersPerToken
	config         Config
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
	if config.ConcurrencyMode != ConcurrencyModeRequests && config.ConcurrencyMode != ConcurrencyModeTokens {
		config.ConcurrencyMode = DefaultConcurrencyMode
	}
	if config.MaxTokenConcurrency < 0 {
		config.MaxTokenConcurrency = DefaultMaxTokenConcurrency
	}

	return &Detector{
		requestTracker: newConcurrencyTracker(),
		tokenTracker:   newConcurrencyTracker(),
		tokenLedger:    newTokenLedger(),
		tokenEstimator: NewSimpleTokenEstimator(),
		config:         config,
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (d *Detector) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{
		Type: ConcurrencyDetectorType,
		Name: ConcurrencyDetectorType,
	}
}

// Saturation calculates the saturation level of the pool.
//
// It returns an aggregate saturation signal where:
//
//	Saturation = Total Inflight Requests / Total MaxConcurrency Capacity.
func (d *Detector) Saturation(_ context.Context, candidateEndpoints []metrics.PodMetrics) float64 {
	var totalInflight, totalCapacity int64
	for _, endpoint := range candidateEndpoints {
		if endpoint.GetMetadata() == nil {
			continue
		}
		endpointID := endpoint.GetMetadata().NamespacedName.String()

		if d.config.ConcurrencyMode == ConcurrencyModeTokens {
			tokenCount := d.tokenTracker.get(endpointID)
			totalInflight += tokenCount
			totalCapacity += d.config.MaxTokenConcurrency
		} else {
			inflight := d.requestTracker.get(endpointID)
			totalInflight += inflight
			totalCapacity += d.config.MaxConcurrency
		}
	}

	if totalCapacity == 0 {
		return 1.0
	}

	return float64(totalInflight) / float64(totalCapacity)
}

// Filter blocks traffic to specific endpoints that are physically saturated or exceeding their safety limits.
//
// It applies a relaxed limit (MaxConcurrency * (1 + Headroom)) to allow for scheduling flexibility and burst tolerance.
func (d *Detector) Filter(
	_ context.Context,
	_ *framework.CycleState,
	_ *framework.LLMRequest,
	endpoints []framework.Endpoint,
) []framework.Endpoint {
	// Pre-allocate assuming most endpoints will pass the filter to minimize allocations.
	filtered := make([]framework.Endpoint, 0, len(endpoints))

	for _, endpoint := range endpoints {
		endpointID := endpoint.GetMetadata().NamespacedName.String()
		if d.config.ConcurrencyMode == ConcurrencyModeTokens {
			limit := int64(float64(d.config.MaxTokenConcurrency) * (1.0 + d.config.Headroom))
			if d.tokenTracker.get(endpointID) < limit {
				filtered = append(filtered, endpoint)
			}
		} else {
			limit := int64(float64(d.config.MaxConcurrency) * (1.0 + d.config.Headroom))
			if d.requestTracker.get(endpointID) < limit {
				filtered = append(filtered, endpoint)
			}
		}
	}
	return filtered
}

// PreRequest increments the atomic in-flight counter for the target endpoint.
// We assume the scheduling result is valid based on the Director's contract.
func (d *Detector) PreRequest(_ context.Context, request *framework.LLMRequest, result *framework.SchedulingResult) {
	eid := result.ProfileResults[result.PrimaryProfileName].TargetEndpoints[0].GetMetadata().NamespacedName.String()
	if d.config.ConcurrencyMode == ConcurrencyModeTokens {
		tokens := d.tokenEstimator.Estimate(request)
		d.tokenTracker.add(eid, tokens)
		d.tokenLedger.add(request.RequestId, eid, tokens)
	} else {
		d.requestTracker.inc(eid)
	}
}

// ResponseComplete decrements the atomic in-flight counter for the target endpoint.
func (d *Detector) ResponseComplete(
	_ context.Context,
	request *framework.LLMRequest,
	_ *requestcontrol.Response,
	targetEndpoint *fwkdl.EndpointMetadata,
) {
	eid := targetEndpoint.NamespacedName.String()
	if d.config.ConcurrencyMode == ConcurrencyModeTokens {
		tokenCount, ok := d.tokenLedger.remove(request.RequestId)
		if ok {
			d.tokenTracker.add(eid, -tokenCount)
		}
	} else {
		d.requestTracker.dec(eid)
	}
}

// DeleteEndpoint removes an endpoint from the concurrency tracker to prevent memory leaks.
// This should be called by the controller when a backend is removed from the pool.
func (d *Detector) DeleteEndpoint(endpointID string) {
	if d.config.ConcurrencyMode == ConcurrencyModeTokens {
		for _, e := range d.tokenLedger.removeForEndpoint(endpointID) {
			d.tokenTracker.add(endpointID, -e.tokenCount)
		}
	} else {
		d.requestTracker.delete(endpointID)
	}
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
	ct.add(endpointID, 1)
}

// Adds detla count for the given endpoint, creates if it doesnt exist
func (ct *concurrencyTracker) add(endpointID string, delta int64) {
	// Fast path: Try with read lock first.
	ct.mu.RLock()
	counter, exists := ct.counts[endpointID]
	ct.mu.RUnlock()

	if exists {
		counter.Add(delta)
		return
	}

	// Slow path: Create counter with write lock.
	ct.mu.Lock()
	defer ct.mu.Unlock()

	// Double-check existence to handle race conditions.
	if counter, exists = ct.counts[endpointID]; exists {
		counter.Add(delta)
		return
	}

	counter = &atomic.Int64{}
	counter.Store(delta)
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
