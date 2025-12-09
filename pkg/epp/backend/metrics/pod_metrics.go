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

package metrics

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	fetchMetricsTimeout = 5 * time.Second
)

type podMetrics struct {
	metadata atomic.Pointer[datalayer.EndpointMetadata]
	metrics  atomic.Pointer[MetricsState]
	pmc      PodMetricsClient
	ds       datalayer.PoolInfo
	interval time.Duration

	startOnce sync.Once // ensures the refresh loop goroutine is started only once
	stopOnce  sync.Once // ensures the done channel is closed only once
	done      chan struct{}

	logger logr.Logger
}

type PodMetricsClient interface {
	FetchMetrics(ctx context.Context, pod *datalayer.EndpointMetadata, existing *MetricsState) (*MetricsState, error)
}

func (pm *podMetrics) String() string {
	return fmt.Sprintf("Metadata: %v; Metrics: %v", pm.GetMetadata(), pm.GetMetrics())
}

func (pm *podMetrics) GetMetadata() *datalayer.EndpointMetadata {
	return pm.metadata.Load()
}

func (pm *podMetrics) GetMetrics() *MetricsState {
	return pm.metrics.Load()
}

func (pm *podMetrics) UpdateMetadata(pod *datalayer.EndpointMetadata) {
	pm.metadata.Store(pod)
}

// start starts a goroutine exactly once to periodically update metrics. The goroutine will be
// stopped either when stop() is called, or the given ctx is cancelled.
func (pm *podMetrics) startRefreshLoop(ctx context.Context) {
	pm.startOnce.Do(func() {
		go func() {
			pm.logger.V(logutil.DEFAULT).Info("Starting refresher", "metadata", pm.GetMetadata())
			ticker := time.NewTicker(pm.interval)
			defer ticker.Stop()
			for {
				select {
				case <-pm.done:
					return
				case <-ctx.Done():
					return
				case <-ticker.C: // refresh metrics periodically
					if err := pm.refreshMetrics(); err != nil {
						pm.logger.V(logutil.TRACE).Error(err, "Failed to refresh metrics", "metadata", pm.GetMetadata())
					}
				}
			}
		}()
	})
}

func (pm *podMetrics) refreshMetrics() error {
	ctx, cancel := context.WithTimeout(context.Background(), fetchMetricsTimeout)
	defer cancel()
	updated, err := pm.pmc.FetchMetrics(ctx, pm.GetMetadata(), pm.GetMetrics())
	if err != nil {
		pm.logger.V(logutil.TRACE).Info("Failed to refreshed metrics:", "err", err)
	}
	// Optimistically update metrics even if there was an error.
	// The FetchMetrics can return an error for the following reasons:
	// 1. As refresher is running in the background, it's possible that the pod is deleted but
	// the refresh goroutine doesn't read the done channel yet. In this case, the updated
	// metrics object will be nil. And the refresher will soon be stopped.
	// 2. The FetchMetrics call can partially fail. For example, due to one metric missing. In
	// this case, the updated metrics object will have partial updates. A partial update is
	// considered better than no updates.
	if updated != nil {
		pm.UpdateMetrics(updated)
	}

	return nil
}

func (pm *podMetrics) stopRefreshLoop() {
	pm.logger.V(logutil.DEFAULT).Info("Stopping refresher", "metadata", pm.GetMetadata())
	pm.stopOnce.Do(func() {
		close(pm.done)
	})
}

// Allowing forward compatibility between PodMetrics and datalayer.Endpoint, by
// implementing missing functions (e.g., extended attributes support) as no-op.
func (*podMetrics) Put(string, datalayer.Cloneable)        {}
func (*podMetrics) Get(string) (datalayer.Cloneable, bool) { return nil, false }
func (*podMetrics) Keys() []string                         { return nil }
func (*podMetrics) GetAttributes() *datalayer.Attributes {
	return nil
}

func (pm *podMetrics) UpdateMetrics(updated *MetricsState) {
	updated.UpdateTime = time.Now()
	pm.logger.V(logutil.TRACE).Info("Refreshed metrics", "updated", updated)
	pm.metrics.Store(updated)
}
