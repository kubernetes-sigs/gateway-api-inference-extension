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

var _ requestcontrol.PreRequest = &Plugin{}
var _ requestcontrol.ResponseBodyProcessor = &Plugin{}

// Plugin tracks in-flight requests via RequestControl hooks and provides eviction capability.
type Plugin struct {
	queue   *EvictionQueue
	aborter Aborter
}

// NewPlugin creates an eviction plugin with the given policies and aborter.
func NewPlugin(
	ordering flowcontrol.EvictionOrderingPolicy,
	filter flowcontrol.EvictionFilterPolicy,
	aborter Aborter,
) *Plugin {
	return &Plugin{
		queue:   NewEvictionQueue(ordering, filter),
		aborter: aborter,
	}
}

func (p *Plugin) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "EvictionPlugin", Name: "eviction"}
}

// PreRequest is called after scheduling, before the request reaches the model server.
// It tracks the request and, if the filter policy accepts it, adds it to the eviction queue.
func (p *Plugin) PreRequest(
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

	item := &flowcontrol.EvictionItem{
		RequestID:      requestID,
		Priority:       request.Objectives.Priority,
		DispatchTime:   time.Now(),
		TargetURL:      "http://" + net.JoinHostPort(metadata.GetIPAddress(), metadata.GetPort()),
		Request:        request,
		TargetEndpoint: metadata,
	}

	p.queue.Track(item)

	// Bind untrack to the request context's lifetime as a safety net.
	// If the client disconnects and ResponseBody(EndOfStream) never fires,
	// ctx.Done() ensures the request is still cleaned up. Untrack is idempotent.
	go func() {
		<-ctx.Done()
		p.queue.Untrack(requestID)
	}()

	log.FromContext(ctx).V(logutil.DEBUG).Info("Tracked in-flight request",
		"requestID", requestID,
		"priority", item.Priority,
		"evictable", p.queue.EvictableLen(),
		"inFlight", p.queue.InFlightLen())
}

// ResponseBody is called for every response data chunk (streaming) or once (non-streaming).
// On the final call (EndOfStream == true), it removes the request from tracking and the eviction queue.
func (p *Plugin) ResponseBody(
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

	p.queue.Untrack(requestID)

	log.FromContext(ctx).V(logutil.DEBUG).Info("Untracked completed request",
		"requestID", requestID,
		"evictable", p.queue.EvictableLen(),
		"inFlight", p.queue.InFlightLen())
}

// EvictN attempts to evict up to n requests from the eviction queue.
// Each request is only removed from tracking after a successful abort. If the abort fails,
// the request remains in the queue for a future eviction attempt.
// Returns the request IDs that were successfully aborted.
func (p *Plugin) EvictN(ctx context.Context, n int) ([]string, error) {
	logger := log.FromContext(ctx)
	aborted := make([]string, 0, n)

	for range n {
		items := p.queue.PopN(1)
		if len(items) == 0 {
			break
		}
		item := items[0]

		if err := p.aborter.Abort(ctx, item); err != nil {
			logger.Error(err, "Failed to abort request, re-tracking", "requestID", item.RequestID, "targetURL", item.TargetURL)
			p.queue.Track(item)
			continue
		}
		aborted = append(aborted, item.RequestID)
	}

	if len(aborted) > 0 {
		logger.Info("Eviction complete", "requested", n, "aborted", len(aborted))
	}
	return aborted, nil
}

// Stats returns the current in-flight and evictable request counts.
func (p *Plugin) Stats() (inFlight int, evictable int) {
	return p.queue.InFlightLen(), p.queue.EvictableLen()
}
