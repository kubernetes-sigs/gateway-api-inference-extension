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
	}
}

type PodMetricsFactory struct {
	pmc                    PodMetricsClient
	refreshMetricsInterval time.Duration
}

func (f *PodMetricsFactory) NewEndpoint(parentCtx context.Context, metadata *datalayer.EndpointMetadata, ds datalayer.PoolInfo) datalayer.Endpoint {
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
	pm.metrics.Store(NewMetricsState())

	pm.startRefreshLoop(parentCtx)
	return pm
}

func (f *PodMetricsFactory) ReleaseEndpoint(ep PodMetrics) {
	if pm, ok := ep.(*podMetrics); ok {
		pm.stopRefreshLoop()
	}
}

type PodMetrics = datalayer.Endpoint
