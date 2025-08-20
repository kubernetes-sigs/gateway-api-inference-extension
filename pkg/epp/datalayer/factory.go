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

package datalayer

import (
	"context"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

// PoolInfo represents the DataStore information needed for endpoints.
// TODO:
// Consider if to remove/simplify in follow-ups. This is mostly for backward
// compatibility with backend.metrics' expectations and allowing a shared
// implementation during the transition.
//   - Endpoint metric scraping uses PoolGet to access the pool's Port and Name.
//   - Global metrics logging uses PoolGet solely for error return and PodList to enumerate
//     all endpoints for metrics summarization.
type PoolInfo interface {
	PoolGet() (*v1.InferencePool, error)
	PodList(func(Endpoint) bool) []Endpoint
}

// EndpointFactory defines an interface for managing Endpoint lifecycle. Specifically,
// providing methods to allocate and retire endpoints. This can potentially be used for
// pooled memory or other management chores in the implementation.
type EndpointFactory interface {
	NewEndpoint(parent context.Context, inpod *corev1.Pod, poolinfo PoolInfo) Endpoint
	ReleaseEndpoint(ep Endpoint)
}

// EndpointLifecycle manages the life cycle (creation and termination) of
// endpoints.
type EndpointLifecycle struct {
	sources         []DataSource  // data sources for collectors
	collectors      sync.Map      // collectors map. key: Pod namespaced name, value: *Collector
	refreshInterval time.Duration // metrics refresh interval
}

// NewEndpointFactory returns a new endpoint for factory, managing collectors for
// its endpoints. This function assumes that sources are not modified afterwards.
func NewEndpointFactory(sources []DataSource, refreshMetricsInterval time.Duration) *EndpointLifecycle {
	return &EndpointLifecycle{
		sources:         sources,
		collectors:      sync.Map{},
		refreshInterval: refreshMetricsInterval,
	}
}

// NewEndpoint implements EndpointFactory.NewEndpoint.
// Creates a new endpoint and starts its associated collector with its own ticker.
// Guards against multiple concurrent calls for the same endpoint.
func (lc *EndpointLifecycle) NewEndpoint(parent context.Context, inpod *corev1.Pod, _ PoolInfo) Endpoint {
	key := types.NamespacedName{Namespace: inpod.Namespace, Name: inpod.Name}
	logger := log.FromContext(parent).WithValues("pod", key)

	if _, ok := lc.collectors.Load(key); ok {
		logger.Info("collector already running for endpoint", "endpoint", key)
		return nil
	}

	endpoint := NewEndpoint()
	endpoint.UpdatePod(inpod)
	collector := NewCollector() // for full backward compatibility, set the logger and poolinfo

	if _, loaded := lc.collectors.LoadOrStore(key, collector); loaded {
		// another goroutine already created and stored a collector for this endpoint.
		// No need to start the new collector.
		logger.Info("collector already running for endpoint", "endpoint", key)
		return nil
	}

	ticker := NewTimeTicker(lc.refreshInterval)
	if err := collector.Start(parent, ticker, endpoint, lc.sources); err != nil {
		logger.Error(err, "failed to start collector for endpoint", "endpoint", key)
		lc.collectors.Delete(key)
	}

	return endpoint
}

// ReleaseEndpoint implements EndpointFactory.ReleaseEndpoint
// Stops the collector and cleans up resources for the endpoint
func (lc *EndpointLifecycle) ReleaseEndpoint(ep Endpoint) {
	key := ep.GetPod().GetNamespacedName()

	if value, ok := lc.collectors.LoadAndDelete(key); ok {
		collector := value.(*Collector)
		_ = collector.Stop()
	}
}

// Shutdown gracefully stops all collectors and cleans up all resources.
func (lc *EndpointLifecycle) Shutdown() {
	lc.collectors.Range(func(key, value any) bool {
		collector := value.(*Collector)
		_ = collector.Stop()
		lc.collectors.Delete(key)
		return true
	})
}
