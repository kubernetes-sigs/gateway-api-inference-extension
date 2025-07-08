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

// Package config holds common configuration default values used across
// different EPP components.
package config

import "time"

const (
	// DefaultKVCacheThreshold is the default KV cache utilization (0.0 to 1.0)
	// threshold.
	DefaultKVCacheThreshold = 0.8
	// DefaultQueueThresholdCritical is the default backend waiting queue size
	// threshold.
	DefaultQueueThresholdCritical = 5
	// DefaultMetricsStalenessThreshold defines how old metrics can be before they
	// are considered stale.
	// The staleness is determined by the refresh internal plus the latency of the metrics API.
	// To be on the safer side, we start with a larger threshold.
	DefaultMetricsStalenessThreshold                = 2 * time.Second                  // default for --metricsStalenessThreshold
	DefaultGrpcPort                                 = 9002                             // default for --grpcPort
	DefaultDestinationEndpointHintMetadataNamespace = "envoy.lb"                       // default for --destinationEndpointHintMetadataNamespace
	DefaultDestinationEndpointHintKey               = "x-gateway-destination-endpoint" // default for --destinationEndpointHintKey
	DefaultPoolName                                 = ""                               // required but no default
	DefaultPoolNamespace                            = "default"                        // default for --poolNamespace
	DefaultRefreshMetricsInterval                   = 50 * time.Millisecond            // default for --refreshMetricsInterval
	DefaultRefreshPrometheusMetricsInterval         = 5 * time.Second                  // default for --refreshPrometheusMetricsInterval
	DefaultSecureServing                            = true                             // default for --secureServing
	DefaultHealthChecking                           = false                            // default for --healthChecking
)
