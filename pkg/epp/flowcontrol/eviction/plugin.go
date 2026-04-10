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

package eviction

import (
	"context"
	"net"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	reqcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/request"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

var _ requestcontrol.PreRequest = &RequestEvictor{}
var _ requestcontrol.ResponseBodyProcessor = &RequestEvictor{}

// RequestEvictor tracks in-flight requests via RequestControl hooks and provides eviction capability.
type RequestEvictor struct {
	queue            *EvictionQueue
	evictor          Evictor
	evictionRegistry *EvictionRegistry
}

// NewRequestEvictor creates a RequestEvictor with the given policies and evictor.
func NewRequestEvictor(
	ordering flowcontrol.EvictionOrderingPolicy,
	filter flowcontrol.EvictionFilterPolicy,
	evictor Evictor,
) *RequestEvictor {
	return &RequestEvictor{
		queue:            NewEvictionQueue(ordering, filter),
		evictor:          evictor,
		evictionRegistry: NewEvictionRegistry(),
	}
}

// EvictionRegistry returns the shared eviction registry.
// The ext_proc Process() goroutine uses this to look up eviction channels for dispatched requests.
func (p *RequestEvictor) EvictionRegistry() *EvictionRegistry {
	return p.evictionRegistry
}

func (p *RequestEvictor) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "EvictionPlugin", Name: "eviction"}
}

// PreRequest is called after scheduling, before the request reaches the model server.
// It tracks the request and, if the filter policy accepts it, adds it to the eviction queue.
func (p *RequestEvictor) PreRequest(
	ctx context.Context,
	request *scheduling.InferenceRequest,
	result *scheduling.SchedulingResult,
) {
	if request == nil || result == nil || len(result.ProfileResults) == 0 {
		return
	}

	profileResult := result.ProfileResults[result.PrimaryProfileName]
	if profileResult == nil || len(profileResult.TargetEndpoints) == 0 {
		return
	}

	targetEndpoint := profileResult.TargetEndpoints[0]
	metadata := targetEndpoint.GetMetadata()
	requestID := request.Headers[reqcommon.RequestIdHeaderKey]
	if requestID == "" {
		return
	}

	evictCh := make(chan struct{})

	item := &flowcontrol.EvictionItem{
		RequestID:      requestID,
		Priority:       request.Objectives.Priority,
		DispatchTime:   time.Now(),
		TargetURL:      "http://" + net.JoinHostPort(metadata.GetIPAddress(), metadata.GetPort()),
		Request:        request,
		TargetEndpoint: metadata,
		EvictCh:        evictCh,
	}

	p.queue.Track(item)
	p.evictionRegistry.Register(requestID, evictCh)

	// Bind untrack to the request context's lifetime as a safety net.
	// If the client disconnects and ResponseBody(EndOfStream) never fires,
	// ctx.Done() ensures the request is still cleaned up. Untrack is idempotent.
	go func() {
		<-ctx.Done()
		p.cleanupRequest(requestID)
	}()

	log.FromContext(ctx).V(logutil.DEBUG).Info("Tracked in-flight request",
		"requestID", requestID,
		"priority", item.Priority,
		"evictable", p.queue.EvictableLen(),
		"inFlight", p.queue.InFlightLen())
}

// ResponseBody is called for every response data chunk (streaming) or once (non-streaming).
// On the final call (EndOfStream == true), it removes the request from tracking and the eviction queue.
func (p *RequestEvictor) ResponseBody(
	ctx context.Context,
	request *scheduling.InferenceRequest,
	response *requestcontrol.Response,
	targetEndpoint *datalayer.EndpointMetadata,
) {
	if !response.EndOfStream {
		return
	}
	if request == nil {
		return
	}
	requestID := request.Headers[reqcommon.RequestIdHeaderKey]
	if requestID == "" {
		return
	}

	p.cleanupRequest(requestID)

	log.FromContext(ctx).V(logutil.DEBUG).Info("Untracked completed request",
		"requestID", requestID,
		"evictable", p.queue.EvictableLen(),
		"inFlight", p.queue.InFlightLen())
}

// EvictN attempts to evict up to n requests from the eviction queue.
// Each request is only removed from tracking after a successful eviction. If the eviction fails,
// the request remains in the queue for a future eviction attempt.
// Returns the request IDs that were successfully evicted.
func (p *RequestEvictor) EvictN(ctx context.Context, n int) ([]string, error) {
	logger := log.FromContext(ctx)
	evicted := make([]string, 0, n)

	for range n {
		items := p.queue.PopN(1)
		if len(items) == 0 {
			break
		}
		item := items[0]

		if err := p.evictor.Evict(ctx, item); err != nil {
			logger.Error(err, "Failed to evict request, re-tracking", "requestID", item.RequestID, "targetURL", item.TargetURL)
			p.queue.Track(item)
			continue
		}
		evicted = append(evicted, item.RequestID)
	}

	if len(evicted) > 0 {
		logger.Info("Eviction complete", "requested", n, "evicted", len(evicted))
	}
	return evicted, nil
}

// Stats returns the current in-flight and evictable request counts.
func (p *RequestEvictor) Stats() (inFlight int, evictable int) {
	return p.queue.InFlightLen(), p.queue.EvictableLen()
}

// cleanupRequest removes a request from all tracking structures.
// If the evictor supports cleanup (e.g., ImmediateResponseEvictor), it also
// cleans up evictor-internal state to prevent unbounded map growth.
func (p *RequestEvictor) cleanupRequest(requestID string) {
	p.queue.Untrack(requestID)
	p.evictionRegistry.Deregister(requestID)
	if c, ok := p.evictor.(EvictorWithCleanup); ok {
		c.Cleanup(requestID)
	}
}

// EvictorWithCleanup is an optional interface for evictors that maintain per-request state
// that needs to be cleaned up when a request completes or is untracked.
type EvictorWithCleanup interface {
	Cleanup(requestID string)
}
