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

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

const (
	ExperimentalDatalayerFeatureGate = "dataLayer"
	PrepareDataPluginsFeatureGate    = "prepareDataPlugins"
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
	PoolGet() (*EndpointPool, error)
	PodList(func(fwkdl.Endpoint) bool) []fwkdl.Endpoint
	// EndpointSetHealthy marks an endpoint as healthy or unhealthy based on metrics scraping results.
	// When healthy is false, the endpoint is removed from PodList results.
	// When healthy is true, the endpoint is added back.
	EndpointSetHealthy(ep fwkdl.Endpoint, healthy bool)
}

// EndpointFactory defines an interface for managing Endpoint lifecycle. Specifically,
// providing methods to allocate and retire endpoints. This can potentially be used for
// pooled memory or other management chores in the implementation.
type EndpointFactory interface {
	SetSources(sources []fwkdl.DataSource)
	NewEndpoint(parent context.Context, inEnpointMetadata *fwkdl.EndpointMetadata, poolinfo PoolInfo) fwkdl.Endpoint
	ReleaseEndpoint(ep fwkdl.Endpoint)
	// ReleaseEndpointsByPodName releases all endpoints associated with the given pod name.
	// This is used when a pod is deleted to ensure all collectors are stopped,
	// even for endpoints that may have been removed from the datastore due to being unhealthy.
	ReleaseEndpointsByPodName(podName string)
}

// collectorEntry stores a collector and its associated pod name for cleanup.
type collectorEntry struct {
	collector *Collector
	podName   string
}

// EndpointLifecycle manages the life cycle (creation and termination) of
// endpoints.
type EndpointLifecycle struct {
	sources         []fwkdl.DataSource // data sources for collectors
	collectors      sync.Map           // collectors map. key: Pod namespaced name, value: *collectorEntry
	refreshInterval time.Duration      // metrics refresh interval
}

// NewEndpointFactory returns a new endpoint for factory, managing collectors for
// its endpoints. This function assumes that sources are not modified afterwards.
func NewEndpointFactory(sources []fwkdl.DataSource, refreshMetricsInterval time.Duration) *EndpointLifecycle {
	eplc := &EndpointLifecycle{
		collectors:      sync.Map{},
		refreshInterval: refreshMetricsInterval,
	}
	eplc.SetSources(sources)
	return eplc
}

// SetSources sets the slice of collectors associated with the endpoint life cycle.
// This overrides any sources which may have previously been set on creation.
func (lc *EndpointLifecycle) SetSources(sources []fwkdl.DataSource) {
	lc.sources = make([]fwkdl.DataSource, len(sources)) // clone the source slice
	copy(lc.sources, sources)
}

// NewEndpoint implements EndpointFactory.NewEndpoint.
// Creates a new endpoint and starts its associated collector with its own ticker.
// Guards against multiple concurrent calls for the same endpoint.
func (lc *EndpointLifecycle) NewEndpoint(parent context.Context, inEndpointMetadata *fwkdl.EndpointMetadata, poolInfo PoolInfo) fwkdl.Endpoint {
	key := types.NamespacedName{Namespace: inEndpointMetadata.GetNamespacedName().Namespace, Name: inEndpointMetadata.GetNamespacedName().Name}
	logger := log.FromContext(parent).WithValues("pod", key)

	if _, ok := lc.collectors.Load(key); ok {
		logger.Info("collector already running for endpoint", "endpoint", key)
		return nil
	}

	endpoint := fwkdl.NewEndpoint(inEndpointMetadata, nil)
	collector := NewCollector(poolInfo)
	entry := &collectorEntry{
		collector: collector,
		podName:   inEndpointMetadata.PodName,
	}

	if _, loaded := lc.collectors.LoadOrStore(key, entry); loaded {
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
func (lc *EndpointLifecycle) ReleaseEndpoint(ep fwkdl.Endpoint) {
	key := ep.GetMetadata().GetNamespacedName()

	if value, ok := lc.collectors.LoadAndDelete(key); ok {
		entry := value.(*collectorEntry)
		_ = entry.collector.Stop()
	}
}

// ReleaseEndpointsByPodName implements EndpointFactory.ReleaseEndpointsByPodName
// Releases all endpoints associated with the given pod name.
func (lc *EndpointLifecycle) ReleaseEndpointsByPodName(podName string) {
	lc.collectors.Range(func(key, value any) bool {
		entry := value.(*collectorEntry)
		if entry.podName == podName {
			_ = entry.collector.Stop()
			lc.collectors.Delete(key)
		}
		return true
	})
}

// Shutdown gracefully stops all collectors and cleans up all resources.
func (lc *EndpointLifecycle) Shutdown() {
	lc.collectors.Range(func(key, value any) bool {
		entry := value.(*collectorEntry)
		_ = entry.collector.Stop()
		lc.collectors.Delete(key)
		return true
	})
}
