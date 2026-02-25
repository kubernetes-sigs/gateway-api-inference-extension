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

// Package metrics is a library to interact with backend metrics.
package metrics

import (
	"context"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

func PodsWithFreshMetrics(stalenessThreshold time.Duration) func(PodMetrics) bool {
	return func(pm PodMetrics) bool {
		if pm == nil {
			return false // Skip nil pods
		}
		return time.Since(pm.GetMetrics().UpdateTime) <= stalenessThreshold
	}
}

func NewPodMetricsFactory(pmc PodMetricsClient, refreshMetricsInterval time.Duration) *PodMetricsFactory {
	return &PodMetricsFactory{
		pmc:                    pmc,
		refreshMetricsInterval: refreshMetricsInterval,
		endpoints:              sync.Map{},
	}
}

// endpointEntry stores an endpoint and its associated pod name for cleanup.
type endpointEntry struct {
	endpoint PodMetrics
	podName  string
}

type PodMetricsFactory struct {
	pmc                    PodMetricsClient
	refreshMetricsInterval time.Duration
	endpoints              sync.Map // key: types.NamespacedName, value: *endpointEntry
}

func (f *PodMetricsFactory) SetSources(_ []fwkdl.DataSource) {
	// no-op
}

func (f *PodMetricsFactory) NewEndpoint(parentCtx context.Context, metadata *fwkdl.EndpointMetadata, ds datalayer.PoolInfo) fwkdl.Endpoint {
	key := metadata.NamespacedName

	// Check if endpoint already exists (e.g., unhealthy endpoint with running collector).
	// Return nil to prevent duplicate goroutines and let the existing collector recover.
	if _, ok := f.endpoints.Load(key); ok {
		return nil
	}

	pm := &podMetrics{
		pmc:       f.pmc,
		ds:        ds,
		interval:  f.refreshMetricsInterval,
		startOnce: sync.Once{},
		stopOnce:  sync.Once{},
		done:      make(chan struct{}),
		logger:    log.FromContext(parentCtx).WithValues("endpoint", metadata.NamespacedName),
	}
	pm.metadata.Store(metadata)
	pm.metrics.Store(fwkdl.NewMetrics())

	// Track the endpoint for cleanup.
	entry := &endpointEntry{
		endpoint: pm,
		podName:  metadata.PodName,
	}
	// Use LoadOrStore for atomic operation in case of concurrent calls.
	if _, loaded := f.endpoints.LoadOrStore(key, entry); loaded {
		return nil
	}

	pm.startRefreshLoop(parentCtx)
	return pm
}

func (f *PodMetricsFactory) ReleaseEndpoint(ep PodMetrics) {
	if pm, ok := ep.(*podMetrics); ok {
		pm.stopRefreshLoop()
		f.endpoints.Delete(pm.GetMetadata().NamespacedName)
	}
}

// ReleaseEndpointsByPodName releases all endpoints associated with the given pod name.
func (f *PodMetricsFactory) ReleaseEndpointsByPodName(podName string) {
	f.endpoints.Range(func(key, value any) bool {
		entry := value.(*endpointEntry)
		if entry.podName == podName {
			if pm, ok := entry.endpoint.(*podMetrics); ok {
				pm.stopRefreshLoop()
			}
			f.endpoints.Delete(key)
		}
		return true
	})
}

type PodMetrics = fwkdl.Endpoint
