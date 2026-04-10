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

package flowcontrol

import (
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// EvictionItem represents an in-flight request that has been dispatched to a model server
// and may be evicted to free resources.
type EvictionItem struct {
	// RequestID is the unique identifier for the request (from x-request-id header).
	RequestID string
	// Priority is the request's priority level from the InferenceObjective.
	Priority int
	// DispatchTime is the timestamp when the request was dispatched to the model server.
	DispatchTime time.Time
	// TargetURL is the base URL of the model server serving this request.
	TargetURL string
	// Request is the original scheduling request.
	Request *scheduling.InferenceRequest
	// TargetEndpoint is the metadata of the endpoint serving this request.
	TargetEndpoint *datalayer.EndpointMetadata
}

// EvictionOrderingPolicy determines which in-flight request gets evicted first.
// The eviction queue is a min-heap: the item for which Less returns true is evicted first (popped first).
//
// Implementations are singletons registered via plugin.Register and instantiated from config.
type EvictionOrderingPolicy interface {
	plugin.Plugin

	// Less returns true if item a should be evicted before item b.
	Less(a, b *EvictionItem) bool
}

// EvictionFilterPolicy determines which in-flight requests are eligible for eviction.
// Only requests for which Accept returns true are added to the eviction queue.
//
// Implementations are singletons registered via plugin.Register and instantiated from config.
type EvictionFilterPolicy interface {
	plugin.Plugin

	// Accept returns true if the request should be tracked in the eviction queue.
	Accept(item *EvictionItem) bool
}
