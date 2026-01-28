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

// Package utilizationdetector implements a mechanism to determine the aggregate saturation level of backend model
// servers based on observed metrics for compute and memory resources.
//
// # Saturation Logic (The Roofline Model)
//
// The detector calculates a continuous saturation gradient where:
//
//	Saturation = Average(PodSaturationScore)
//
// For each pod, the score is determined by the most constrained resource (Compute vs. Memory), following a Roofline
// performance model:
//
//	PodSaturationScore = Max(WaitingQueue / QueueThreshold, KVCacheUsage / KVCacheThreshold)
package utilizationdetector

import (
	"context"
	"time"

	"github.com/go-logr/logr"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/util/logging"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
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

// Detector determines system saturation based on metrics of the given candidate pods.
type Detector struct {
	config *Config
}

// NewDetector creates a new SaturationDetector.
// The config provides the thresholds for determining saturation.
func NewDetector(config *Config, logger logr.Logger) *Detector {
	logger.WithName(loggerName).V(logutil.DEFAULT).Info("Creating new SaturationDetector",
		"queueDepthThreshold", config.QueueDepthThreshold,
		"kvCacheUtilThreshold", config.KVCacheUtilThreshold,
		"metricsStalenessThreshold", config.MetricsStalenessThreshold.String())

	return &Detector{
		config: config,
	}
}

// Saturation calculates the saturation level of the pool.
//
// It returns an aggregate saturation signal where:
//
//	Saturation = Average(PodSaturationScore)
//
// For each pod, the score is determined by the most constrained resource (Compute or Memory):
//
//	PodScore = Max(WaitingQueue / QueueThreshold, KVCacheUsage / KVCacheThreshold)
func (d *Detector) Saturation(_ context.Context, candidatePods []backendmetrics.PodMetrics) float64 {
	if len(candidatePods) == 0 {
		return 1.0
	}

	var totalScore float64
	for _, podMetric := range candidatePods {
		metrics := podMetric.GetMetrics()

		if metrics == nil || time.Since(metrics.UpdateTime) > d.config.MetricsStalenessThreshold {
			totalScore += 1.0
			continue
		}

		qRatio := float64(metrics.WaitingQueueSize) / float64(d.config.QueueDepthThreshold)
		kvRatio := metrics.KVCacheUsagePercent / d.config.KVCacheUtilThreshold

		// Roofline Analysis: The pod is saturated if either resource is exhausted.
		totalScore += max(qRatio, kvRatio)
	}

	return totalScore / float64(len(candidatePods))
}
