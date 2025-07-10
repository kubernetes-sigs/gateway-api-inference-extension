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
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	fetchMetricsTimeout = 5 * time.Second
)

type podMetrics struct {
	pod      atomic.Pointer[backend.Pod]
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
	FetchMetrics(ctx context.Context, pod *backend.Pod, existing *MetricsState, port int32) (*MetricsState, error)
}

func (pm *podMetrics) String() string {
	pod := pm.GetPod()
	metrics := pm.GetMetrics()
	requestCount := 0
	if pod != nil && pod.RunningRequests != nil {
		requestCount = pod.RunningRequests.GetSize()
	}

	return fmt.Sprintf("PodMetrics{%s, %s, %d running requests, waiting: %d, running: %d, kv_cache: %.2f%%}",
		pod.NamespacedName.String(),
		pod.Address,
		requestCount,
		metrics.WaitingQueueSize,
		metrics.RunningQueueSize,
		metrics.KVCacheUsagePercent)
}

func (pm *podMetrics) GetPod() *backend.Pod {
	return pm.pod.Load()
}

func (pm *podMetrics) GetMetrics() *MetricsState {
	return pm.metrics.Load()
}

// New methods for priority queue integration
func (pm *podMetrics) GetRunningRequests() *backend.RequestPriorityQueue {
	pod := pm.GetPod()
	if pod == nil {
		return nil
	}
	return pod.RunningRequests
}

func (pm *podMetrics) AddRequest(requestID string, tpot float64) bool {
	pod := pm.GetPod()
	if pod == nil || pod.RunningRequests == nil {
		return false
	}
	success := pod.RunningRequests.Add(requestID, tpot)
	// No need to update metrics since we removed ActualRunningRequests
	return success
}

func (pm *podMetrics) RemoveRequest(requestID string) bool {
	pod := pm.GetPod()
	if pod == nil || pod.RunningRequests == nil {
		return false
	}
	_, success := pod.RunningRequests.Remove(requestID)
	// No need to update metrics since we removed ActualRunningRequests
	return success
}

func (pm *podMetrics) UpdateRequest(requestID string, tpot float64) bool {
	pod := pm.GetPod()
	if pod == nil || pod.RunningRequests == nil {
		return false
	}
	return pod.RunningRequests.Update(requestID, tpot)
}

func (pm *podMetrics) GetRequestCount() int {
	pod := pm.GetPod()
	if pod == nil || pod.RunningRequests == nil {
		return 0
	}
	return pod.RunningRequests.GetSize()
}

func (pm *podMetrics) ContainsRequest(requestID string) bool {
	pod := pm.GetPod()
	if pod == nil || pod.RunningRequests == nil {
		return false
	}
	return pod.RunningRequests.Contains(requestID)
}

func (pm *podMetrics) PeekRequestPriorityQueue() *backend.Request {
	pod := pm.GetPod()
	if pod == nil || pod.RunningRequests == nil {
		return nil
	}
	return pod.RunningRequests.Peek()
}

func (pm *podMetrics) UpdatePod(k8sPod *corev1.Pod) {
	currentPod := pm.GetPod()
	updatedPod := toInternalPod(k8sPod)

	// Preserve the existing running requests queue if it exists
	if currentPod != nil && currentPod.RunningRequests != nil {
		updatedPod.RunningRequests = currentPod.RunningRequests
	}

	pm.pod.Store(updatedPod)
}
func toInternalPod(pod *corev1.Pod, existingQueue *backend.RequestPriorityQueue) *backend.Pod {
	labels := make(map[string]string, len(pod.GetLabels()))
	for key, value := range pod.GetLabels() {
		labels[key] = value
	}

	queue := existingQueue
	if queue == nil {
		queue = backend.NewRequestPriorityQueue()
	}

	return &backend.Pod{
		NamespacedName: types.NamespacedName{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		},
		Address:         pod.Status.PodIP,
		Labels:          labels,
		RunningRequests: queue,
	}
}

// start starts a goroutine exactly once to periodically update metrics. The goroutine will be
// stopped either when stop() is called, or the given ctx is cancelled.
func (pm *podMetrics) startRefreshLoop(ctx context.Context) {
	pm.startOnce.Do(func() {
		go func() {
			pm.logger.V(logutil.DEFAULT).Info("Starting refresher", "pod", pm.GetPod())
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
						pm.logger.V(logutil.TRACE).Error(err, "Failed to refresh metrics", "pod", pm.GetPod())
					}
				}
			}
		}()
	})
}

func (pm *podMetrics) refreshMetrics() error {
	pool, err := pm.ds.PoolGet()
	if err != nil {
		// No inference pool or not initialize.
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), fetchMetricsTimeout)
	defer cancel()
	if len(pool.Spec.TargetPorts) != 1 {
		return fmt.Errorf("expected 1 target port, got %d", len(pool.Spec.TargetPorts))
	}
	updated, err := pm.pmc.FetchMetrics(ctx, pm.GetPod(), pm.GetMetrics(), int32(pool.Spec.TargetPorts[0].Number))
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
	pm.logger.V(logutil.DEFAULT).Info("Stopping refresher", "pod", pm.GetPod())
	pm.stopOnce.Do(func() {
		close(pm.done)
	})
}

// Allowing forward compatibility between PodMetrics and datalayer.Endpoint, by
// implementing missing functions (e.g., extended attributes support) as no-op.
func (*podMetrics) Put(string, datalayer.Cloneable)        {}
func (*podMetrics) Get(string) (datalayer.Cloneable, bool) { return nil, false }
func (*podMetrics) Keys() []string                         { return nil }

func (pm *podMetrics) UpdateMetrics(updated *MetricsState) {
	updated.UpdateTime = time.Now()
	pm.logger.V(logutil.TRACE).Info("Refreshed metrics", "updated", updated)
	pm.metrics.Store(updated)
}
