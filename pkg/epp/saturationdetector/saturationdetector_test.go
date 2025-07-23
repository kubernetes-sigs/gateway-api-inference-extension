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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
)

// --- Mock Implementations ---

type mockDatastore struct {
	pods []backendmetrics.PodMetrics
}

// PodGetAll returns all pod metrics from the fake datastore.
func (fds *mockDatastore) PodGetAll() []backendmetrics.PodMetrics {
	return fds.pods
}

// Helper function to create a properly initialized fake pod metrics
func newMockPodMetrics(name string, metrics *backendmetrics.MetricsState) backendmetrics.PodMetrics {
	// Create a proper k8s pod
	k8sPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "ns1",
			Labels:    map[string]string{"app": "test"},
		},
		Status: corev1.PodStatus{
			PodIP: "192.168.1.1",
		},
	}

	// Use the proper constructor
	fakePodMetrics := backendmetrics.NewFakePodMetrics(k8sPod)

	// Create a custom fake that can return the specified metrics
	return &testPodMetrics{
		FakePodMetrics: fakePodMetrics,
		customMetrics:  metrics,
	}
}

// testPodMetrics wraps FakePodMetrics to allow custom metrics for testing
type testPodMetrics struct {
	*backendmetrics.FakePodMetrics
	customMetrics *backendmetrics.MetricsState
}

// AddRequest implements metrics.PodMetrics.
// Subtle: this method shadows the method (*FakePodMetrics).AddRequest of testPodMetrics.FakePodMetrics.
func (t *testPodMetrics) AddRequest(requestID string, tpot float64) bool {
	panic("unimplemented")
}

// ContainsRequest implements metrics.PodMetrics.
// Subtle: this method shadows the method (*FakePodMetrics).ContainsRequest of testPodMetrics.FakePodMetrics.
func (t *testPodMetrics) ContainsRequest(requestID string) bool {
	panic("unimplemented")
}

// GetPod implements metrics.PodMetrics.
// Subtle: this method shadows the method (*FakePodMetrics).GetPod of testPodMetrics.FakePodMetrics.
func (t *testPodMetrics) GetPod() *backend.Pod {
	panic("unimplemented")
}

// GetRequestCount implements metrics.PodMetrics.
// Subtle: this method shadows the method (*FakePodMetrics).GetRequestCount of testPodMetrics.FakePodMetrics.
func (t *testPodMetrics) GetRequestCount() int {
	panic("unimplemented")
}

// GetRunningRequests implements metrics.PodMetrics.
// Subtle: this method shadows the method (*FakePodMetrics).GetRunningRequests of testPodMetrics.FakePodMetrics.
func (t *testPodMetrics) GetRunningRequests() *backend.RequestPriorityQueue {
	panic("unimplemented")
}

// PeekRequestPriorityQueue implements metrics.PodMetrics.
func (t *testPodMetrics) PeekRequestPriorityQueue() *backend.Request {
	panic("unimplemented")
}

// RemoveRequest implements metrics.PodMetrics.
// Subtle: this method shadows the method (*FakePodMetrics).RemoveRequest of testPodMetrics.FakePodMetrics.
func (t *testPodMetrics) RemoveRequest(requestID string) bool {
	panic("unimplemented")
}

// StopRefreshLoop implements metrics.PodMetrics.
// Subtle: this method shadows the method (*FakePodMetrics).StopRefreshLoop of testPodMetrics.FakePodMetrics.
func (t *testPodMetrics) StopRefreshLoop() {
	panic("unimplemented")
}

// String implements metrics.PodMetrics.
// Subtle: this method shadows the method (*FakePodMetrics).String of testPodMetrics.FakePodMetrics.
func (t *testPodMetrics) String() string {
	panic("unimplemented")
}

// UpdatePod implements metrics.PodMetrics.
// Subtle: this method shadows the method (*FakePodMetrics).UpdatePod of testPodMetrics.FakePodMetrics.
func (t *testPodMetrics) UpdatePod(*corev1.Pod) {
	panic("unimplemented")
}

// UpdateRequest implements metrics.PodMetrics.
// Subtle: this method shadows the method (*FakePodMetrics).UpdateRequest of testPodMetrics.FakePodMetrics.
func (t *testPodMetrics) UpdateRequest(requestID string, tpot float64) bool {
	panic("unimplemented")
}

// Override GetMetrics to return custom metrics for testing
func (t *testPodMetrics) GetMetrics() *backendmetrics.MetricsState {
	return t.customMetrics // Return exactly what was passed, including nil
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
				QueueDepthThreshold:       10,
				KVCacheUtilThreshold:      0.8,
				MetricsStalenessThreshold: 100 * time.Millisecond,
			},
			datastore:                    &mockDatastore{},
			expectedQueueDepthThreshold:  10,
			expectedKVCacheUtilThreshold: 0.8,
			expectedStalenessThreshold:   100 * time.Millisecond,
		},
		{
			name: "invalid thresholds, fallback to default",
			config: &Config{
				QueueDepthThreshold:       -1,
				KVCacheUtilThreshold:      -5,
				MetricsStalenessThreshold: 0,
			},
			datastore:                    &mockDatastore{},
			expectedQueueDepthThreshold:  DefaultQueueDepthThreshold,
			expectedKVCacheUtilThreshold: DefaultKVCacheUtilThreshold,
			expectedStalenessThreshold:   DefaultMetricsStalenessThreshold,
		},
		{
			name: "kv cache threshold above range, fallback to default",
			config: &Config{
				QueueDepthThreshold:       10,
				KVCacheUtilThreshold:      1.5,
				MetricsStalenessThreshold: 100 * time.Millisecond,
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
			os.Setenv(EnvSdMetricsStalenessThreshold, test.config.MetricsStalenessThreshold.String())

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
			if detector.config.MetricsStalenessThreshold != test.expectedStalenessThreshold {
				t.Errorf("NewDetector() MetricsStalenessThreshold = %v, want %v", detector.config.MetricsStalenessThreshold, test.expectedStalenessThreshold)
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
		name            string
		config          *Config
		pods            []backendmetrics.PodMetrics
		expectedSaturat bool
	}{
		{
			name:            "No pods in datastore",
			config:          defaultConfig,
			pods:            []backendmetrics.PodMetrics{},
			expectedSaturat: true, // No capacity = saturated
		},
		{
			name:   "Single pod with good capacity",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    2,
					KVCacheUsagePercent: 0.5,
					ActiveModels:        make(map[string]int),
					WaitingModels:       make(map[string]int),
				}),
			},
			expectedSaturat: false,
		},
		{
			name:   "Single pod with stale metrics",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime.Add(-200 * time.Millisecond), // Stale
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.1,
					ActiveModels:        make(map[string]int),
					WaitingModels:       make(map[string]int),
				}),
			},
			expectedSaturat: true,
		},
		{
			name:   "Single pod with high queue depth",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    10, // Exceeds threshold 5
					KVCacheUsagePercent: 0.1,
					ActiveModels:        make(map[string]int),
					WaitingModels:       make(map[string]int),
				}),
			},
			expectedSaturat: true,
		},
		{
			name:   "Single pod with high KV cache utilization",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.95, // Exceeds threshold 0.90
					ActiveModels:        make(map[string]int),
					WaitingModels:       make(map[string]int),
				}),
			},
			expectedSaturat: true,
		},
		{
			name:   "Single pod with nil metrics",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", nil),
			},
			expectedSaturat: true,
		},
		{
			name:   "Multiple pods, all good capacity",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.1,
					ActiveModels:        make(map[string]int),
					WaitingModels:       make(map[string]int),
				}),
				newMockPodMetrics("pod2", &backendmetrics.MetricsState{
					UpdateTime:          baseTime.Add(-10 * time.Millisecond),
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.2,
					ActiveModels:        make(map[string]int),
					WaitingModels:       make(map[string]int),
				}),
			},
			expectedSaturat: false,
		},
		{
			name:   "Multiple pods, one good, one bad (stale)",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime, // Good
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.1,
					ActiveModels:        make(map[string]int),
					WaitingModels:       make(map[string]int),
				}),
				newMockPodMetrics("pod2", &backendmetrics.MetricsState{
					UpdateTime:          baseTime.Add(-300 * time.Millisecond), // Stale
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.2,
					ActiveModels:        make(map[string]int),
					WaitingModels:       make(map[string]int),
				}),
			},
			expectedSaturat: false, // One good pod is enough
		},
		{
			name:   "Multiple pods, one good, one bad (high queue)",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.1,
					ActiveModels:        make(map[string]int),
					WaitingModels:       make(map[string]int),
				}),
				newMockPodMetrics("pod2", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    15, // Bad queue
					KVCacheUsagePercent: 0.2,
					ActiveModels:        make(map[string]int),
					WaitingModels:       make(map[string]int),
				}),
			},
			expectedSaturat: false,
		},
		{
			name:   "Multiple pods, all bad capacity",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime.Add(-200 * time.Millisecond), // Stale
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.1,
					ActiveModels:        make(map[string]int),
					WaitingModels:       make(map[string]int),
				}),
				newMockPodMetrics("pod2", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    20, // High queue
					KVCacheUsagePercent: 0.2,
					ActiveModels:        make(map[string]int),
					WaitingModels:       make(map[string]int),
				}),
				newMockPodMetrics("pod3", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.99, // High KV
					ActiveModels:        make(map[string]int),
					WaitingModels:       make(map[string]int),
				}),
			},
			expectedSaturat: true,
		},
		{
			name:   "Queue depth exactly at threshold",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    defaultConfig.QueueDepthThreshold, // Exactly at threshold (good)
					KVCacheUsagePercent: 0.1,
					ActiveModels:        make(map[string]int),
					WaitingModels:       make(map[string]int),
				}),
			},
			expectedSaturat: false,
		},
		{
			name:   "KV cache exactly at threshold",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime,
					WaitingQueueSize:    1,
					KVCacheUsagePercent: defaultConfig.KVCacheUtilThreshold, // Exactly at threshold (good)
					ActiveModels:        make(map[string]int),
					WaitingModels:       make(map[string]int),
				}),
			},
			expectedSaturat: false,
		},
		{
			name:   "Metrics age just over staleness threshold",
			config: defaultConfig,
			pods: []backendmetrics.PodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime.Add(-defaultConfig.MetricsStalenessThreshold - time.Nanosecond), // Just over (stale)
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.1,
					ActiveModels:        make(map[string]int),
					WaitingModels:       make(map[string]int),
				}),
			},
			expectedSaturat: true,
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
