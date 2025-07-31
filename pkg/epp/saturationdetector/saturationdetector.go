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

// Package saturationdetector implements a mechanism to determine if the
// backend model servers are considered saturated based on observed metrics.
//
// The current implementation provides a global saturation signal (IsSaturated)
// primarily based on backend queue depths and KV cache utilization, reflecting
// the saturation signals previously used by the Scheduler before the
// introduction of the FlowController. It fetches live metrics from the
// provided Datastore.
//
// TODO: Explore more advanced saturation signals in the future, such as:
//   - Latency-objective-based saturation.
//   - Predictive saturation based on trends.
//   - Hysteresis bands or other smoothing techniques to prevent rapid
//     oscillations of the saturation signal.
package saturationdetector

import (
	"context"
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	// loggerName is the name to use for loggers created by this package.
	loggerName = "SaturationDetector"
)

// Config holds the configuration for the SaturationDetector.
type Config struct {
	// QueueDepthThreshold defines the backend waiting queue size above which a
	// pod is considered to have insufficient capacity for new requests.
	QueueDepthThreshold int
	// KVCacheUtilThreshold defines the KV cache utilization (0.0 to 1.0) above
	// which a pod is considered to have insufficient capacity.
	KVCacheUtilThreshold float64
	// MetricsStalenessThreshold defines how old a pod's metrics can be.
	// If a pod's metrics are older than this, it might be excluded from
	// "good capacity" considerations or treated as having no capacity for
	// safety.
	MetricsStalenessThreshold time.Duration
}

// Datastore provides an interface to access backend pod metrics.
type Datastore interface {
	PodList(predicate func(backendmetrics.PodMetrics) bool) []backendmetrics.PodMetrics
}

// Detector determines system saturation based on metrics from the Datastore.
//
// The Detector currently holds a direct dependency on a Datastore interface.
// This design choice was made to encapsulate the logic of fetching and
// interpreting metrics for saturation, thereby simplifying the dependencies
// for primary consumers like the FlowController--to be added soon--(which
// would otherwise need to manage Datastore interactions itself).
// This architectural decision may be revisited in the future if a more
// decoupled approach (e.g., passing metrics directly to IsSaturated) proves
// more beneficial.
type Detector struct {
	datastore Datastore
	config    *Config
}

// NewDetector creates a new SaturationDetector.
// The datastore is expected to provide access to live/recently-updated pod
// metrics.
// The config provides the thresholds for determining saturation.
func NewDetector(config *Config, datastore Datastore, logger logr.Logger) *Detector {
	logger.WithName(loggerName).V(logutil.DEFAULT).Info("Creating new SaturationDetector",
		"queueDepthThreshold", config.QueueDepthThreshold,
		"kvCacheUtilThreshold", config.KVCacheUtilThreshold,
		"metricsStalenessThreshold", config.MetricsStalenessThreshold.String())

	return &Detector{
		datastore: datastore,
		config:    config,
	}
}

// IsSaturated checks if the system is currently considered saturated.
// The system is saturated if NO pod currently has "good capacity".
// "Good capacity" means:
//  1. Metrics are fresh (not stale).
//  2. WaitingQueueSize <= QueueDepthThreshold.
//  3. KVCacheUsagePercent <= KVCacheUtilThreshold.
//
// If no pods are found in the datastore, the system is considered saturated
// (no capacity).
func (d *Detector) IsSaturated(ctx context.Context) bool {
	logger := log.FromContext(ctx).WithName(loggerName)
	// TODO: filter out stale metrics here if needed.
	allPodsMetrics := d.datastore.PodList(backendmetrics.AllPodPredicate)
	if len(allPodsMetrics) == 0 {
		logger.V(logutil.VERBOSE).Info("No pods found in datastore; system is considered SATURATED (no capacity).")
		// If there are no pods, there is no capacity to serve requests.
		// Treat this as a saturated state to enable FlowController queuing.
		return true
	}

	for _, podMetric := range allPodsMetrics {
		metrics := podMetric.GetMetrics()
		podNn := "unknown-pod"
		if podMetric.GetPod() != nil {
			podNn = podMetric.GetPod().NamespacedName.String()
		}

		if metrics == nil {
			logger.V(logutil.TRACE).Info("Pod has nil metrics, skipping for saturation check",
				"pod", podNn)
			continue
		}

		// Check for metric staleness
		if time.Since(metrics.UpdateTime) > d.config.MetricsStalenessThreshold {
			logger.V(logutil.TRACE).Info("Pod metrics are stale, considered as not having good capacity",
				"pod", podNn, "updateTime", metrics.UpdateTime, "stalenessThreshold", d.config.MetricsStalenessThreshold)
			continue
		}

		// Check queue depth
		if metrics.WaitingQueueSize > d.config.QueueDepthThreshold {
			logger.V(logutil.TRACE).Info("Pod WaitingQueueSize is above threshold, considered as not having good capacity",
				"pod", podNn, "waitingQueueSize", metrics.WaitingQueueSize, "threshold", d.config.QueueDepthThreshold)
			continue // WaitingQueueSize is above threshold, considered saturated.
		}

		// Check KV cache utilization
		if metrics.KVCacheUsagePercent > d.config.KVCacheUtilThreshold {
			logger.V(logutil.TRACE).Info("Pod KVCacheUsagePercent is above threshold, considered as not having good capacity",
				"pod", podNn, "kvCacheUsagePercent", metrics.KVCacheUsagePercent, "threshold", d.config.KVCacheUtilThreshold)
			continue // KVCacheUsagePercent is above threshold, considered saturated.
		}

		logger.V(logutil.TRACE).Info("Found pod with good capacity", "pod", podNn, "waitingQueue", metrics.WaitingQueueSize,
			"queueThreshold", d.config.QueueDepthThreshold, "kvCacheUtil", metrics.KVCacheUsagePercent, "kvCacheThreshold", d.config.KVCacheUtilThreshold)

		return false // Found at least one pod with good capacity, so system is NOT saturated.
	}

	logger.V(logutil.VERBOSE).Info("No pods found with good capacity; system is considered SATURATED.")
	return true
}
