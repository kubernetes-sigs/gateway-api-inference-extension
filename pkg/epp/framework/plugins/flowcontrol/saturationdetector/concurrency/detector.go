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
// Package concurrency implements a synchronous saturation detector and scheduling filter for LLM
// routing. It actively tracks in-flight requests and tokens using an open-loop accounting mechanism
// to provide instantaneous backpressure and protect endpoints from sudden traffic bursts.
//
// For detailed architectural trade-offs and configuration, see the package README.
package concurrency

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	ConcurrencyDetectorType = "concurrency-detector"
)

// ConcurrencyDetectorFactory instantiates the detector plugin using the provided JSON parameters.
func ConcurrencyDetectorFactory(
	name string,
	params json.RawMessage,
	handle fwkplugin.Handle,
) (fwkplugin.Plugin, error) {
	var apiCfg apiConfig
	if len(params) > 0 {
		if err := json.Unmarshal(params, &apiCfg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal concurrency detector config: %w", err)
		}
	}
	cfg, err := buildConfig(&apiCfg)
	if err != nil {
		return nil, err
	}
	return newDetector(name, *cfg, log.FromContext(handle.Context())), nil
}

var (
	_ requestcontrol.PreRequest      = &detector{}
	_ requestcontrol.ResponseBody    = &detector{}
	_ framework.Filter               = &detector{}
	_ flowcontrol.SaturationDetector = &detector{}
)

// detector implements a saturation detector and scheduling filter based on active request concurrency.
type detector struct {
	requestTracker *concurrencyTracker // requests in flight per endpoint
	tokenTracker   *concurrencyTracker // tokens in flight per endpoint
	tokenEstimator TokenEstimator      // SimpleTokenEstimator with CharactersPerToken
	config         config
	typedName      fwkplugin.TypedName
}

// newDetector creates a new instance of the Concurrency Detector.
func newDetector(name string, cfg config, logger logr.Logger) *detector {
	typedName := fwkplugin.TypedName{
		Type: ConcurrencyDetectorType,
		Name: name,
	}

	pluginLogger := logger.WithName(typedName.String())
	pluginLogger.V(logutil.DEFAULT).Info("Creating new ConcurrencyDetector",
		"mode", cfg.mode,
		"maxConcurrency", cfg.maxConcurrency,
		"maxTokenConcurrency", cfg.maxTokenConcurrency,
		"headroom", cfg.headroom)

	if cfg.headroom > 1.0 {
		pluginLogger.Info("Unusually high headroom configured; verify value is a fraction, not a percentage",
			"headroom", cfg.headroom,
			"effectiveBurst", fmt.Sprintf("%.0f%%", cfg.headroom*100))
	}

	return &detector{
		requestTracker: newConcurrencyTracker(),
		tokenTracker:   newConcurrencyTracker(),
		tokenEstimator: NewSimpleTokenEstimator(),
		config:         cfg,
		typedName:      typedName,
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (d *detector) TypedName() fwkplugin.TypedName {
	return d.typedName
}

// Saturation calculates the saturation level of the pool.
//
// It returns an aggregate saturation signal where:
//
//	Saturation = Total Inflight Requests / Total MaxConcurrency Capacity.
func (d *detector) Saturation(_ context.Context, endpoints []datalayer.Endpoint) float64 {
	var totalInflight, totalCapacity int64
	for _, e := range endpoints {
		if e.GetMetadata() == nil {
			continue
		}
		endpointID := e.GetMetadata().NamespacedName.String()

		if d.config.mode == modeTokens {
			tokenCount := d.tokenTracker.get(endpointID)
			totalInflight += tokenCount
			totalCapacity += d.config.maxTokenConcurrency
		} else {
			inflight := d.requestTracker.get(endpointID)
			totalInflight += inflight
			totalCapacity += d.config.maxConcurrency
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
func (d *detector) Filter(
	_ context.Context,
	_ *framework.CycleState,
	_ *framework.LLMRequest,
	endpoints []framework.Endpoint,
) []framework.Endpoint {
	// Pre-allocate assuming most endpoints will pass the filter to minimize allocations.
	filtered := make([]framework.Endpoint, 0, len(endpoints))

	var limit int64
	if d.config.mode == modeTokens {
		limit = int64(float64(d.config.maxTokenConcurrency) * (1.0 + d.config.headroom))
	} else {
		limit = int64(float64(d.config.maxConcurrency) * (1.0 + d.config.headroom))
	}

	for _, e := range endpoints {
		endpointID := e.GetMetadata().NamespacedName.String()
		if d.config.mode == modeTokens {
			if d.tokenTracker.get(endpointID) < limit {
				filtered = append(filtered, e)
			}
		} else {
			if d.requestTracker.get(endpointID) < limit {
				filtered = append(filtered, e)
			}
		}
	}
	return filtered
}

// PreRequest increments the atomic in-flight counter for the target endpoint.
// We assume the scheduling result is valid based on the Director's contract.
func (d *detector) PreRequest(_ context.Context, request *framework.LLMRequest, result *framework.SchedulingResult) {
	eid := result.ProfileResults[result.PrimaryProfileName].TargetEndpoints[0].GetMetadata().NamespacedName.String()
	if d.config.mode == modeTokens {
		tokens := d.tokenEstimator.Estimate(request)
		d.tokenTracker.add(eid, tokens)
	} else {
		d.requestTracker.inc(eid)
	}
}

// ResponseBody decrements the atomic in-flight counter for the target endpoint when the response is endOfStream.
// For token mode, the estimate is recalculated from the request; request may be nil in some paths and
// TokenEstimator.Estimate returns 0 in that case (may contribute to drift).
func (d *detector) ResponseBody(
	_ context.Context,
	request *framework.LLMRequest,
	resp *requestcontrol.Response,
	targetEndpoint *datalayer.EndpointMetadata,
) {
	if targetEndpoint == nil || resp == nil || !resp.EndOfStream {
		return
	}
	eid := targetEndpoint.NamespacedName.String()
	if d.config.mode == modeTokens {
		tokens := d.tokenEstimator.Estimate(request)
		d.tokenTracker.add(eid, -tokens)
	} else {
		d.requestTracker.dec(eid)
	}
}

// DeleteEndpoint removes an endpoint from the concurrency tracker to prevent memory leaks.
// This should be called by the controller when a backend is removed from the pool.
func (d *detector) DeleteEndpoint(endpointID string) {
	if d.config.mode == modeTokens {
		d.tokenTracker.delete(endpointID)
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

// add adds delta to the inflight count for the given endpoint, creating the counter if it doesn't exist.
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
