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
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/common/config"
)

// --- Mock Implementations ---

type mockDatastore struct {
	pods []*backendmetrics.FakePodMetrics
}

// PodGetAll returns all pod metrics from the fake datastore.
func (fds *mockDatastore) PodGetAll() []backendmetrics.PodMetrics {
	pm := make([]backendmetrics.PodMetrics, 0, len(fds.pods))
	for _, pod := range fds.pods {
		pm = append(pm, pod)
	}
	return pm
}

// PodGetAll returns all pod metrics from the fake datastore.
func (fds *mockDatastore) PodList(predicate func(backendmetrics.PodMetrics) bool) []backendmetrics.PodMetrics {
	pm := make([]backendmetrics.PodMetrics, 0, len(fds.pods))
	for _, pod := range fds.pods {
		pm = append(pm, pod)
	}
	return pm
}

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
		name                         string
		config                       *Config
		datastore                    Datastore
		expectedQueueDepthThreshold  int
		expectedKVCacheUtilThreshold float64
		expectedStalenessThreshold   time.Duration
	}{
		{
			name: "Valid config",
			config: &Config{
				QueueDepthThreshold:  10,
				KVCacheUtilThreshold: 0.8,
			},
			datastore:                    &mockDatastore{},
			expectedQueueDepthThreshold:  10,
			expectedKVCacheUtilThreshold: 0.8,
			expectedStalenessThreshold:   100 * time.Millisecond,
		},
		{
			name: "invalid thresholds, fallback to default",
			config: &Config{
				QueueDepthThreshold:  -1,
				KVCacheUtilThreshold: -5,
			},
			datastore:                    &mockDatastore{},
			expectedQueueDepthThreshold:  DefaultQueueDepthThreshold,
			expectedKVCacheUtilThreshold: DefaultKVCacheUtilThreshold,
			expectedStalenessThreshold:   config.DefaultMetricsStalenessThreshold,
		},
		{
			name: "kv cache threshold above range, fallback to default",
			config: &Config{
				QueueDepthThreshold:  10,
				KVCacheUtilThreshold: 1.5,
			},
			datastore:                    &mockDatastore{},
			expectedQueueDepthThreshold:  10,
			expectedKVCacheUtilThreshold: DefaultKVCacheUtilThreshold,
			expectedStalenessThreshold:   100 * time.Millisecond,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// validate configuration values are loaded from env vars properly, including the use of default values when provided value is invalid.
			os.Setenv(EnvSdQueueDepthThreshold, strconv.Itoa(test.config.QueueDepthThreshold))
			os.Setenv(EnvSdKVCacheUtilThreshold, fmt.Sprintf("%v", test.config.KVCacheUtilThreshold))

			detector := NewDetector(LoadConfigFromEnv(), test.datastore, logr.Discard())
			if detector == nil {
				t.Fatalf("NewDetector() returned nil detector for valid config")
			}
			if detector.config.QueueDepthThreshold != test.expectedQueueDepthThreshold {
				t.Errorf("NewDetector() QueueDepthThreshold = %d, want %d", detector.config.QueueDepthThreshold, test.expectedQueueDepthThreshold)
			}
			if detector.config.KVCacheUtilThreshold != test.expectedKVCacheUtilThreshold {
				t.Errorf("NewDetector() KVCacheUtilThreshold = %f, want %f", detector.config.KVCacheUtilThreshold, test.expectedKVCacheUtilThreshold)
			}
		})
	}
}

func TestDetector_IsSaturated(t *testing.T) {
	baseTime := time.Now()
	defaultConfig := &Config{
		QueueDepthThreshold:  5,
		KVCacheUtilThreshold: 0.90,
	}

	tests := []struct {
		name            string
		config          *Config
		pods            []*backendmetrics.FakePodMetrics
		expectedSaturat bool
	}{
		{
			name:            "No pods in datastore",
			config:          defaultConfig,
			pods:            []*backendmetrics.FakePodMetrics{},
			expectedSaturat: true, // No capacity = saturated
		},
		{
			name:   "Single pod with good capacity",
			config: defaultConfig,
			pods: []*backendmetrics.FakePodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    2,
					KVCacheUsagePercent: 0.5,
				}),
			},
			expectedSaturat: false,
		},
		{
			name:   "Single pod with high queue depth",
			config: defaultConfig,
			pods: []*backendmetrics.FakePodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    10, // Exceeds threshold 5
					KVCacheUsagePercent: 0.1,
				}),
			},
			expectedSaturat: true,
		},
		{
			name:   "Single pod with high KV cache utilization",
			config: defaultConfig,
			pods: []*backendmetrics.FakePodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.95, // Exceeds threshold 0.90
				}),
			},
			expectedSaturat: true,
		},
		{
			name:   "Single pod with nil metrics",
			config: defaultConfig,
			pods: []*backendmetrics.FakePodMetrics{
				newMockPodMetrics("pod1", nil),
			},
			expectedSaturat: true,
		},
		{
			name:   "Multiple pods, all good capacity",
			config: defaultConfig,
			pods: []*backendmetrics.FakePodMetrics{
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
			expectedSaturat: false,
		},
		{
			name:   "Multiple pods, one good, one bad (stale)",
			config: defaultConfig,
			pods: []*backendmetrics.FakePodMetrics{
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
			expectedSaturat: false, // One good pod is enough
		},
		{
			name:   "Multiple pods, one good, one bad (high queue)",
			config: defaultConfig,
			pods: []*backendmetrics.FakePodMetrics{
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
			expectedSaturat: false,
		},
		{
			name:   "Multiple pods, all bad capacity",
			config: defaultConfig,
			pods: []*backendmetrics.FakePodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    20, // High queue
					KVCacheUsagePercent: 0.2,
				}),
				newMockPodMetrics("pod2", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.99, // High KV
				}),
			},
			expectedSaturat: true,
		},
		{
			name:   "Queue depth exactly at threshold",
			config: defaultConfig,
			pods: []*backendmetrics.FakePodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    defaultConfig.QueueDepthThreshold, // Exactly at threshold (good)
					KVCacheUsagePercent: 0.1,
				}),
			},
			expectedSaturat: false,
		},
		{
			name:   "KV cache exactly at threshold",
			config: defaultConfig,
			pods: []*backendmetrics.FakePodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    1,
					KVCacheUsagePercent: defaultConfig.KVCacheUtilThreshold, // Exactly at threshold (good)
				}),
			},
			expectedSaturat: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			detector := NewDetector(test.config, &mockDatastore{pods: test.pods}, logr.Discard())

			if got := detector.IsSaturated(context.Background()); got != test.expectedSaturat {
				t.Errorf("IsSaturated() = %v, want %v", got, test.expectedSaturat)
			}
		})
	}
}
