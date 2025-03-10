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
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	fetchMetricsTimeout = 5 * time.Second
)

type podMetrics struct {
	pod      unsafe.Pointer // stores a *Pod
	metrics  unsafe.Pointer // stores a *Metrics
	pmc      PodMetricsClient
	ds       Datastore
	interval time.Duration

	parentCtx context.Context
	once      sync.Once // ensure the StartRefreshLoop is only called once.
	done      chan struct{}

	logger logr.Logger
}

type PodMetricsClient interface {
	FetchMetrics(ctx context.Context, pod *Pod, existing *Metrics, port int32) (*Metrics, error)
}

func (pm *podMetrics) GetPod() *Pod {
	return (*Pod)(atomic.LoadPointer(&pm.pod))
}

func (pm *podMetrics) GetMetrics() *Metrics {
	return (*Metrics)(atomic.LoadPointer(&pm.metrics))
}

func (pm *podMetrics) UpdatePod(in *corev1.Pod) {
	atomic.StorePointer(&pm.pod, unsafe.Pointer(toInternalPod(in)))
}

func toInternalPod(in *corev1.Pod) *Pod {
	return &Pod{
		NamespacedName: types.NamespacedName{
			Name:      in.Name,
			Namespace: in.Namespace,
		},
		Address: in.Status.PodIP,
	}
}

// start starts a goroutine exactly once to periodically update metrics. The goroutine will be
// stopped either when stop() is called, or the parentCtx is cancelled.
func (pm *podMetrics) startRefreshLoop() {
	pm.once.Do(func() {
		go func() {
			pm.logger.V(logutil.DEFAULT).Info("Starting refresher", "pod", pm.GetPod())
			for {
				select {
				case <-pm.done:
					return
				case <-pm.parentCtx.Done():
					return
				default:
				}

				err := pm.refreshMetrics()
				if err != nil {
					pm.logger.V(logutil.TRACE).Error(err, "Failed to refresh metrics", "pod", pm.GetPod())
				}

				time.Sleep(pm.interval)
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
	updated, err := pm.pmc.FetchMetrics(ctx, pm.GetPod(), pm.GetMetrics(), pool.Spec.TargetPortNumber)
	if err != nil {
		// As refresher is running in the background, it's possible that the pod is deleted but
		// the refresh goroutine doesn't read the done channel yet. In this case, we just return nil.
		// The refresher will be stopped after this interval.
		return nil
	}
	updated.UpdateTime = time.Now()

	pm.logger.V(logutil.TRACE).Info("Refreshed metrics", "updated", updated)

	atomic.StorePointer(&pm.metrics, unsafe.Pointer(updated))
	return nil
}

func (pm *podMetrics) StopRefreshLoop() {
	pm.logger.V(logutil.DEFAULT).Info("Stopping refresher", "pod", pm.GetPod())
	close(pm.done)
}
