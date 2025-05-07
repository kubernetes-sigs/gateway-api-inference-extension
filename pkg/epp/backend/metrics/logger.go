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
	podinfo "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/pod-info"
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
	// Pod operations
	// PodGetAll returns all pod info objects.
	PodGetAll() []podinfo.PodInfo
	PodList(func(podinfo.PodInfo) bool) []podinfo.PodInfo
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
					podsWithFreshMetrics := datastore.PodList(func(podInfo podinfo.PodInfo) bool {
						if metrics := getMetricsFromPodInfo(podInfo); metrics != nil {
							return time.Since(metrics.UpdateTime) <= metricsValidityPeriod
						}
						return false
					})
					podsWithStaleMetrics := datastore.PodList(func(podInfo podinfo.PodInfo) bool {
						if metrics := getMetricsFromPodInfo(podInfo); metrics != nil {
							return time.Since(metrics.UpdateTime) > metricsValidityPeriod
						}
						return false
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

	podsInfo := datastore.PodGetAll()
	logger.V(logutil.TRACE).Info("Refreshing Prometheus Metrics", "ReadyPods", len(podsInfo))
	if len(podsInfo) == 0 {
		return
	}

	for _, podInfo := range podsInfo {
		if metrics := getMetricsFromPodInfo(podInfo); metrics != nil {
			kvCacheTotal += metrics.KVCacheUsagePercent
			queueTotal += metrics.WaitingQueueSize
		}
	}

	podTotalCount := len(podsInfo)
	metrics.RecordInferencePoolAvgKVCache(pool.Name, kvCacheTotal/float64(podTotalCount))
	metrics.RecordInferencePoolAvgQueueSize(pool.Name, float64(queueTotal/podTotalCount))
	metrics.RecordinferencePoolReadyPods(pool.Name, float64(podTotalCount))
}

func getMetricsFromPodInfo(podInfo podinfo.PodInfo) *MetricsData {
	metrics, ok := podInfo.GetData(MetricsDataKey)
	if !ok {
		return nil // no entry in the map with metrics key
	}
	return metrics.(*MetricsData)
}
