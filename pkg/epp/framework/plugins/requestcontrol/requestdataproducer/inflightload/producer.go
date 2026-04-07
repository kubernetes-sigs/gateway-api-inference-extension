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

package inflightload

import (
	"context"
	"encoding/json"
	"reflect"
	"sync"
	"sync/atomic"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrconcurrency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

const (
	InFlightLoadProducerType = "inflight-load-producer"
)

func InFlightLoadProducerFactory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return &InFlightLoadProducer{
		typedName:      fwkplugin.TypedName{Type: InFlightLoadProducerType, Name: name},
		requestTracker: newConcurrencyTracker(),
		tokenTracker:   newConcurrencyTracker(),
		tokenEstimator: NewSimpleTokenEstimator(),
	}, nil
}

var (
	_ requestcontrol.PreRequest        = &InFlightLoadProducer{}
	_ requestcontrol.ResponseBody      = &InFlightLoadProducer{}
	_ requestcontrol.PrepareDataPlugin = &InFlightLoadProducer{}
	_ datalayer.NotificationExtractor  = &InFlightLoadProducer{}
)

type InFlightLoadProducer struct {
	typedName      fwkplugin.TypedName
	requestTracker *concurrencyTracker
	tokenTracker   *concurrencyTracker
	tokenEstimator TokenEstimator
}

func (p *InFlightLoadProducer) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// GVK returns the GroupVersionKind this extractor handles.
func (p *InFlightLoadProducer) GVK() schema.GroupVersionKind {
	return corev1.SchemeGroupVersion.WithKind("Pod")
}

// ExpectedInputType defines the type expected by the extractor.
func (p *InFlightLoadProducer) ExpectedInputType() reflect.Type {
	return nil // Not used for NotificationExtractors.
}

// Extract transforms the raw data into structured attributes (not used for notifications).
func (p *InFlightLoadProducer) Extract(context.Context, any, datalayer.Endpoint) error {
	return nil
}

// ExtractNotification handles pod deletion events to prune stateful trackers.
func (p *InFlightLoadProducer) ExtractNotification(ctx context.Context, event datalayer.NotificationEvent) error {
	if event.Type != datalayer.EventDelete || event.Object == nil {
		return nil
	}

	nsn := types.NamespacedName{
		Namespace: event.Object.GetNamespace(),
		Name:      event.Object.GetName(),
	}
	id := nsn.String()

	p.DeleteEndpoint(id)
	log.FromContext(ctx).V(logutil.DEFAULT).Info("Cleaned up in-flight load for deleted pod", "pod", id)
	return nil
}

func (p *InFlightLoadProducer) PrepareRequestData(_ context.Context, _ *framework.InferenceRequest, endpoints []framework.Endpoint) error {
	for _, e := range endpoints {
		endpointID := e.GetMetadata().NamespacedName.String()
		e.Put(attrconcurrency.InFlightLoadKey, &attrconcurrency.InFlightLoad{
			Tokens:   p.tokenTracker.get(endpointID),
			Requests: p.requestTracker.get(endpointID),
		})
	}
	return nil
}

func (p *InFlightLoadProducer) PreRequest(_ context.Context, request *framework.InferenceRequest, result *framework.SchedulingResult) {
	if result == nil || len(result.ProfileResults) == 0 {
		return
	}

	// For the foundational Scorer PR, we only track the primary target endpoint.
	primaryResult := result.ProfileResults[result.PrimaryProfileName]
	if primaryResult == nil || len(primaryResult.TargetEndpoints) == 0 {
		return
	}

	for _, endpoint := range primaryResult.TargetEndpoints {
		if endpoint == nil || endpoint.GetMetadata() == nil {
			continue
		}
		eid := endpoint.GetMetadata().NamespacedName.String()
		p.requestTracker.inc(eid)
		tokens := p.tokenEstimator.Estimate(request)
		p.tokenTracker.add(eid, tokens)
	}
}

func (p *InFlightLoadProducer) ResponseBody(
	_ context.Context,
	request *framework.InferenceRequest,
	resp *requestcontrol.Response,
	targetEndpoint *datalayer.EndpointMetadata,
) {
	// For the foundational Scorer PR, we only perform cleanup on EndOfStream.
	if targetEndpoint == nil || resp == nil || !resp.EndOfStream {
		return
	}

	eid := targetEndpoint.NamespacedName.String()
	p.requestTracker.dec(eid)
	tokens := p.tokenEstimator.Estimate(request)
	p.tokenTracker.add(eid, -tokens)
}

func (p *InFlightLoadProducer) Produces() map[string]any {
	return map[string]any{
		attrconcurrency.InFlightLoadKey: &attrconcurrency.InFlightLoad{},
	}
}

func (p *InFlightLoadProducer) Consumes() map[string]any {
	return nil
}

// DeleteEndpoint removes an endpoint from the concurrency trackers to prevent memory leaks.
// This matches the design of the previous saturation detector and is called by the
// ExtractNotification hook to ensure deterministic cleanup of stateful data.
func (p *InFlightLoadProducer) DeleteEndpoint(endpointID string) {
	p.requestTracker.delete(endpointID)
	p.tokenTracker.delete(endpointID)
}

// concurrencyTracker manages thread-safe counters for inflight requests.
type concurrencyTracker struct {
	mu     sync.RWMutex
	counts map[string]*atomic.Int64
}

func newConcurrencyTracker() *concurrencyTracker {
	return &concurrencyTracker{
		counts: make(map[string]*atomic.Int64),
	}
}

func (ct *concurrencyTracker) get(endpointID string) int64 {
	ct.mu.RLock()
	counter, exists := ct.counts[endpointID]
	ct.mu.RUnlock()

	if !exists {
		return 0
	}
	return counter.Load()
}

func (ct *concurrencyTracker) inc(endpointID string) {
	ct.add(endpointID, 1)
}

func (ct *concurrencyTracker) add(endpointID string, delta int64) {
	ct.mu.RLock()
	counter, exists := ct.counts[endpointID]
	ct.mu.RUnlock()

	if exists {
		counter.Add(delta)
		return
	}

	ct.mu.Lock()
	defer ct.mu.Unlock()

	if counter, exists = ct.counts[endpointID]; exists {
		counter.Add(delta)
		return
	}

	counter = &atomic.Int64{}
	counter.Store(delta)
	ct.counts[endpointID] = counter
}

func (ct *concurrencyTracker) dec(endpointID string) {
	ct.add(endpointID, -1)
}

func (ct *concurrencyTracker) delete(endpointID string) {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	delete(ct.counts, endpointID)
}
