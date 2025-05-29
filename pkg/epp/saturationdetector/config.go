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
package saturationdetector

import (
	"fmt"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"
	commonconfig "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/common/config"
	envutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/env"
)

// Default configuration values
const (
	DefaultQueueDepthThreshold  = commonconfig.DefaultQueueThresholdCritical
	DefaultKVCacheUtilThreshold = commonconfig.DefaultKVCacheThreshold
	// DefaultMetricsStalenessThreshold defines how old metrics can be before they
	// are considered stale.
	// Given the pod metrics refresh interval is 50ms, a threshold slightly above
	// that should be fine.
	DefaultMetricsStalenessThreshold = 200 * time.Millisecond
)

// Environment variable names for SaturationDetector configuration
const (
	EnvSdQueueDepthThreshold       = "SD_QUEUE_DEPTH_THRESHOLD"
	EnvSdKVCacheUtilThreshold      = "SD_KV_CACHE_UTIL_THRESHOLD"
	EnvSdMetricsStalenessThreshold = "SD_METRICS_STALENESS_THRESHOLD"
)

// LoadConfigFromEnv loads SaturationDetector Config from environment variables.
func LoadConfigFromEnv() *Config {
	// Use a default logger for initial configuration loading.
	logger := log.Log.WithName("saturation-detector-config")

	cfg := &Config{}

	cfg.QueueDepthThreshold = envutil.GetEnvInt(EnvSdQueueDepthThreshold, DefaultQueueDepthThreshold, logger)
	if cfg.QueueDepthThreshold <= 0 {
		cfg.QueueDepthThreshold = DefaultQueueDepthThreshold
	}

	cfg.KVCacheUtilThreshold = envutil.GetEnvFloat(EnvSdKVCacheUtilThreshold, DefaultKVCacheUtilThreshold, logger)
	if cfg.KVCacheUtilThreshold <= 0 || cfg.KVCacheUtilThreshold >= 1 {
		cfg.KVCacheUtilThreshold = DefaultKVCacheUtilThreshold
	}

	cfg.MetricsStalenessThreshold = envutil.GetEnvDuration(EnvSdMetricsStalenessThreshold, DefaultMetricsStalenessThreshold, logger)
	if cfg.MetricsStalenessThreshold <= 0 {
		cfg.MetricsStalenessThreshold = DefaultMetricsStalenessThreshold
	}

	// NewDetector validates the config and assigns defaults.
	logger.Info("SaturationDetector configuration loaded from env", "config", fmt.Sprintf("%+v", cfg))
	return cfg
}
