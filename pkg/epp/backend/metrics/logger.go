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
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	// Note currently the EPP treats stale metrics same as fresh.
	// TODO: https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/336
	metricsValidityPeriod = 5 * time.Second
	debugPrintInterval    = 5 * time.Second
)

type Datastore interface {
	PoolGet() (*v1alpha2.InferencePool, error)
	// PodMetrics operations
	// PodGetAll returns all pods and metrics, including fresh and stale.
	PodGetAll() []PodMetrics
	PodList(func(PodMetrics) bool) []PodMetrics
}

// StartMetricsLogger starts goroutines to 1) Print metrics debug logs if the DEBUG log level is
// enabled; 2) flushes Prometheus metrics about the backend servers.
func StartMetricsLogger(ctx context.Context, datastore Datastore, refreshPrometheusMetricsInterval time.Duration) {
	logger := log.FromContext(ctx)
	ticker := time.NewTicker(refreshPrometheusMetricsInterval)
	go func() {
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				logger.V(logutil.DEFAULT).Info("Shutting down prometheus metrics thread")
				return
			case <-ticker.C: // Periodically refresh prometheus metrics for inference pool
				refreshPrometheusMetrics(logger, datastore)
			}
		}
	}()

	// Periodically print out the pods and metrics for DEBUGGING.
	if logger := logger.V(logutil.DEBUG); logger.Enabled() {
		go func() {
			ticker := time.NewTicker(debugPrintInterval)
			defer ticker.Stop()
			for {
				select {
				case <-ctx.Done():
					logger.V(logutil.DEFAULT).Info("Shutting down metrics logger thread")
					return
				case <-ticker.C:
					podsWithFreshMetrics := datastore.PodList(func(pm PodMetrics) bool {
						return time.Since(pm.GetMetrics().UpdateTime) <= metricsValidityPeriod
					})
					podsWithStaleMetrics := datastore.PodList(func(pm PodMetrics) bool {
						return time.Since(pm.GetMetrics().UpdateTime) > metricsValidityPeriod
					})
					s := fmt.Sprintf("Current Pods and metrics gathered. Fresh metrics: %+v, Stale metrics: %+v", podsWithFreshMetrics, podsWithStaleMetrics)
					logger.V(logutil.VERBOSE).Info(s)
				}
			}
		}()
	}
}

func refreshPrometheusMetrics(logger logr.Logger, datastore Datastore) {
	pool, err := datastore.PoolGet()
	if err != nil {
		// No inference pool or not initialize.
		logger.V(logutil.DEFAULT).Info("Pool is not initialized, skipping refreshing metrics")
		return
	}

	var kvCacheTotal float64
	var queueTotal int

	podMetrics := datastore.PodGetAll()
	logger.V(logutil.TRACE).Info("Refreshing Prometheus Metrics", "ReadyPods", len(podMetrics))
	if len(podMetrics) == 0 {
		return
	}

	for _, pod := range podMetrics {
		kvCacheTotal += pod.GetMetrics().KVCacheUsagePercent
		queueTotal += pod.GetMetrics().WaitingQueueSize
	}

	podTotalCount := len(podMetrics)
	metrics.RecordInferencePoolAvgKVCache(pool.Name, kvCacheTotal/float64(podTotalCount))
	metrics.RecordInferencePoolAvgQueueSize(pool.Name, float64(queueTotal/podTotalCount))
	metrics.RecordinferencePoolReadyPods(pool.Name, float64(podTotalCount))
}
