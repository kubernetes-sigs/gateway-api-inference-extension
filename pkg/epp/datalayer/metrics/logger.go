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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const debugPrintInterval = 5 * time.Second

// StartMetricsLogger starts background goroutines for:
// 1. Refreshing Prometheus metrics periodically
// 2. Debug logging (if DEBUG level enabled)
func StartMetricsLogger(ctx context.Context, datastore datalayer.PoolInfo, refreshInterval, stalenessThreshold time.Duration) {
	logger := log.FromContext(ctx)

	go runPrometheusRefresher(ctx, logger, datastore, refreshInterval, stalenessThreshold)

	if logger.V(logutil.DEBUG).Enabled() {
		go runDebugLogger(ctx, logger, datastore, stalenessThreshold)
	}
}

func runPrometheusRefresher(ctx context.Context, logger logr.Logger, datastore datalayer.PoolInfo, interval, stalenessThreshold time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			logger.V(logutil.DEFAULT).Info("Shutting down prometheus metrics thread")
			return
		case <-ticker.C:
			refreshPrometheusMetrics(logger, datastore, stalenessThreshold)
		}
	}
}

func runDebugLogger(ctx context.Context, logger logr.Logger, datastore datalayer.PoolInfo, stalenessThreshold time.Duration) {
	ticker := time.NewTicker(debugPrintInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			logger.V(logutil.DEFAULT).Info("Shutting down metrics logger thread")
			return
		case <-ticker.C:
			printDebugMetrics(logger, datastore, stalenessThreshold)
		}
	}
}

func podsWithFreshMetrics(stalenessThreshold time.Duration) func(datalayer.Endpoint) bool {
	return func(ep datalayer.Endpoint) bool {
		if ep == nil {
			return false // Skip nil pods
		}
		return time.Since(ep.GetMetrics().UpdateTime) <= stalenessThreshold
	}
}

func podsWithStaleMetrics(stalenessThreshold time.Duration) func(datalayer.Endpoint) bool {
	return func(ep datalayer.Endpoint) bool {
		if ep == nil {
			return false // Skip nil pods
		}
		return time.Since(ep.GetMetrics().UpdateTime) > stalenessThreshold
	}
}

func printDebugMetrics(logger logr.Logger, datastore datalayer.PoolInfo, stalenessThreshold time.Duration) {
	freshPods := datastore.PodList(podsWithFreshMetrics(stalenessThreshold))
	stalePods := datastore.PodList(podsWithStaleMetrics(stalenessThreshold))

	logger.V(logutil.TRACE).Info("Current Pods and metrics gathered",
		"Fresh metrics", fmt.Sprintf("%+v", freshPods), "Stale metrics", fmt.Sprintf("%+v", stalePods))
}

func refreshPrometheusMetrics(logger logr.Logger, datastore datalayer.PoolInfo, stalenessThreshold time.Duration) {
	pool, err := datastore.PoolGet()
	if err != nil {
		logger.V(logutil.DEFAULT).Info("Pool is not initialized, skipping refreshing metrics")
		return
	}

	podMetrics := datastore.PodList(podsWithFreshMetrics(stalenessThreshold))
	logger.V(logutil.TRACE).Info("Refreshing Prometheus Metrics", "ReadyPods", len(podMetrics))

	if len(podMetrics) == 0 {
		return
	}

	totals := calculateTotals(podMetrics)
	podCount := len(podMetrics)

	metrics.RecordInferencePoolAvgKVCache(pool.Name, totals.kvCache/float64(podCount))
	metrics.RecordInferencePoolAvgQueueSize(pool.Name, float64(totals.queueSize/podCount))
	metrics.RecordInferencePoolReadyPods(pool.Name, float64(podCount))
}

// totals holds aggregated metric values
type totals struct {
	kvCache   float64
	queueSize int
}

func calculateTotals(endpoints []datalayer.Endpoint) totals {
	var result totals
	for _, pod := range endpoints {
		metrics := pod.GetMetrics()
		result.kvCache += metrics.KVCacheUsagePercent
		result.queueSize += metrics.WaitingQueueSize
	}
	return result
}
