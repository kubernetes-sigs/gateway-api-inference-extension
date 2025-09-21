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
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
)

func newMockPodMetrics(name string, metrics *backendmetrics.MetricsState) *backendmetrics.FakePodMetrics {
	return &backendmetrics.FakePodMetrics{
		Pod: &backend.Pod{
			NamespacedName: types.NamespacedName{Name: name, Namespace: "ns1"},
		},
		Metrics: metrics,
	}
}

// --- Tests ---

func TestNewDetector(t *testing.T) {
	tests := []struct {
		name           string
		config         *Config
		expectedConfig *Config
	}{
		{
			name: "Valid config",
			config: &Config{
				QueueDepthThreshold:       10,
				KVCacheUtilThreshold:      0.8,
				MetricsStalenessThreshold: 100 * time.Millisecond,
			},
			expectedConfig: &Config{
				QueueDepthThreshold:       10,
				KVCacheUtilThreshold:      0.8,
				MetricsStalenessThreshold: 100 * time.Millisecond,
			},
		},
		{
			name: "invalid thresholds, fallback to default",
			config: &Config{
				QueueDepthThreshold:       -1,
				KVCacheUtilThreshold:      -5,
				MetricsStalenessThreshold: 0,
			},
			expectedConfig: &Config{
				QueueDepthThreshold:       DefaultQueueDepthThreshold,
				KVCacheUtilThreshold:      DefaultKVCacheUtilThreshold,
				MetricsStalenessThreshold: DefaultMetricsStalenessThreshold,
			},
		},
		{
			name: "kv cache threshold above range, fallback to default",
			config: &Config{
				QueueDepthThreshold:       10,
				KVCacheUtilThreshold:      1.5,
				MetricsStalenessThreshold: 100 * time.Millisecond,
			},
			expectedConfig: &Config{
				QueueDepthThreshold:       10,
				KVCacheUtilThreshold:      DefaultKVCacheUtilThreshold,
				MetricsStalenessThreshold: 100 * time.Millisecond,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// validate configuration values are loaded from env vars properly, including the use of default values when provided value is invalid.
			os.Setenv(EnvSdQueueDepthThreshold, strconv.Itoa(test.config.QueueDepthThreshold))
			os.Setenv(EnvSdKVCacheUtilThreshold, fmt.Sprintf("%v", test.config.KVCacheUtilThreshold))
			os.Setenv(EnvSdMetricsStalenessThreshold, test.config.MetricsStalenessThreshold.String())

			detector := NewDetector(LoadConfigFromEnv(), logr.Discard())
			if diff := cmp.Diff(test.expectedConfig, detector.config); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}

func TestDetector_IsSaturated(t *testing.T) {
	baseTime := time.Now()
	defaultConfig := &Config{
		QueueDepthThreshold:       5,
		KVCacheUtilThreshold:      0.90,
		MetricsStalenessThreshold: 100 * time.Millisecond,
	}

	tests := []struct {
		name               string
		config             *Config
		pods               []backendmetrics.PodMetrics
		expectedSaturation bool
	}{
		{
			name:               "No candidate pods",
			config:             defaultConfig,
			pods:               []backendmetrics.PodMetrics{},
			expectedSaturation: true, // No capacity = saturated
		},
		{
			name:   "Single pod with good capacity",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    2,
					KVCacheUsagePercent: 0.5,
				}),
			},
			expectedSaturation: false,
		},
		{
			name:   "Single pod with stale metrics",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime.Add(-200 * time.Millisecond), // Stale
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.1,
				}),
			},
			expectedSaturation: true,
		},
		{
			name:   "Single pod with high queue depth",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    10, // Exceeds threshold 5
					KVCacheUsagePercent: 0.1,
				}),
			},
			expectedSaturation: true,
		},
		{
			name:   "Single pod with high KV cache utilization",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.95, // Exceeds threshold 0.90
				}),
			},
			expectedSaturation: true,
		},
		{
			name:   "Single pod with nil metrics",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", nil),
			},
			expectedSaturation: true,
		},
		{
			name:   "Multiple pods, all good capacity",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.1,
				}),
				newMockPodMetrics("pod2", &backendmetrics.MetricsState{
					UpdateTime:          baseTime.Add(-10 * time.Millisecond),
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.2,
				}),
			},
			expectedSaturation: false,
		},
		{
			name:   "Multiple pods, one good, one bad (stale)",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime, // Good
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.1,
				}),
				newMockPodMetrics("pod2", &backendmetrics.MetricsState{
					UpdateTime:          baseTime.Add(-300 * time.Millisecond), // Stale
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.2,
				}),
			},
			expectedSaturation: false, // One good pod is enough
		},
		{
			name:   "Multiple pods, one good, one bad (high queue)",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.1,
				}),
				newMockPodMetrics("pod2", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    15, // Bad queue
					KVCacheUsagePercent: 0.2,
				}),
			},
			expectedSaturation: false,
		},
		{
			name:   "Multiple pods, all bad capacity",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime.Add(-200 * time.Millisecond), // Stale
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.1,
				}),
				newMockPodMetrics("pod2", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    20, // High queue
					KVCacheUsagePercent: 0.2,
				}),
				newMockPodMetrics("pod3", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.99, // High KV
				}),
			},
			expectedSaturation: true,
		},
		{
			name:   "Queue depth exactly at threshold",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    defaultConfig.QueueDepthThreshold, // Exactly at threshold (good)
					KVCacheUsagePercent: 0.1,
				}),
			},
			expectedSaturation: false,
		},
		{
			name:   "KV cache exactly at threshold",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    1,
					KVCacheUsagePercent: defaultConfig.KVCacheUtilThreshold, // Exactly at threshold (good)
				}),
			},
			expectedSaturation: false,
		},
		{
			name:   "Metrics age just over staleness threshold",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime.Add(-defaultConfig.MetricsStalenessThreshold - time.Nanosecond), // Just over (stale)
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.1,
				}),
			},
			expectedSaturation: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			detector := NewDetector(test.config, logr.Discard())

			if got := detector.IsSaturated(context.Background(), test.pods); got != test.expectedSaturation {
				t.Errorf("IsSaturated() = %v, want %v", got, test.expectedSaturation)
			}
		})
	}
}
