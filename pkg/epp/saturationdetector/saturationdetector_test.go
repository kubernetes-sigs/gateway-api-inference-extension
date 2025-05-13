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
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
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

// mockClock allows controlling time in tests.
type mockClock struct {
	mu   sync.RWMutex
	time time.Time
}

func newMockClock(t time.Time) *mockClock {
	return &mockClock{time: t}
}

func (c *mockClock) now() time.Time {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.time
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
		name                    string
		config                  Config
		datastore               Datastore
		expectError             error
		expectedStalenessThresh time.Duration
	}{
		{
			name: "Valid config",
			config: Config{
				QueueDepthThreshold:       10,
				KVCacheUtilThreshold:      0.8,
				MetricsStalenessThreshold: 100 * time.Millisecond,
			},
			datastore:               &mockDatastore{},
			expectError:             nil,
			expectedStalenessThresh: 100 * time.Millisecond,
		},
		{
			name:                    "Nil datastore",
			config:                  Config{},
			datastore:               nil,
			expectError:             ErrNilDatastore,
			expectedStalenessThresh: DefaultMetricsStalenessThreshold, // Default will be set if error didn't occur first
		},
		{
			name: "Zero staleness threshold uses default",
			config: Config{
				QueueDepthThreshold:       5,
				KVCacheUtilThreshold:      0.9,
				MetricsStalenessThreshold: 0, // Should use default
			},
			datastore:               &mockDatastore{},
			expectError:             nil,
			expectedStalenessThresh: DefaultMetricsStalenessThreshold,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			detector, err := NewDetector(tt.config, tt.datastore, logr.Discard())

			if !errors.Is(err, tt.expectError) {
				t.Errorf("NewDetector() error = %v, wantErr %v", err, tt.expectError)
			}

			if err == nil && detector != nil {
				detector.clock = newMockClock(time.Now())
				if detector.config.MetricsStalenessThreshold != tt.expectedStalenessThresh {
					t.Errorf("NewDetector() MetricsStalenessThreshold = %v, want %v", detector.config.MetricsStalenessThreshold, tt.expectedStalenessThresh)
				}
				if detector.config.QueueDepthThreshold != tt.config.QueueDepthThreshold {
					t.Errorf("NewDetector() QueueDepthThreshold = %d, want %d", detector.config.QueueDepthThreshold, tt.config.QueueDepthThreshold)
				}
				if detector.config.KVCacheUtilThreshold != tt.config.KVCacheUtilThreshold {
					t.Errorf("NewDetector() KVCacheUtilThreshold = %f, want %f", detector.config.KVCacheUtilThreshold, tt.config.KVCacheUtilThreshold)
				}
			}
		})
	}
}

func TestDetector_IsSaturated(t *testing.T) {
	baseTime := time.Now()
	defaultConfig := Config{
		QueueDepthThreshold:       5,
		KVCacheUtilThreshold:      0.90,
		MetricsStalenessThreshold: 100 * time.Millisecond,
	}

	tests := []struct {
		name            string
		config          Config
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
			name:   "Single pod with stale metrics",
			config: defaultConfig,
			pods: []*backendmetrics.FakePodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime.Add(-200 * time.Millisecond), // Stale
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.1,
				}),
			},
			expectedSaturat: true,
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
		{
			name:   "Metrics age exactly at staleness threshold",
			config: defaultConfig,
			pods: []*backendmetrics.FakePodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime.Add(-defaultConfig.MetricsStalenessThreshold), // Exactly at threshold (good)
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.1,
				}),
			},
			expectedSaturat: false,
		},
		{
			name:   "Metrics age just over staleness threshold",
			config: defaultConfig,
			pods: []*backendmetrics.FakePodMetrics{
				newMockPodMetrics("pod1", &backendmetrics.MetricsState{
					UpdateTime:          baseTime.Add(-defaultConfig.MetricsStalenessThreshold - time.Nanosecond), // Just over (stale)
					WaitingQueueSize:    1,
					KVCacheUsagePercent: 0.1,
				}),
			},
			expectedSaturat: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockDS := &mockDatastore{pods: tt.pods}

			detector, err := NewDetector(tt.config, mockDS, logr.Discard())
			if err != nil {
				t.Fatalf("NewDetector() failed: %v", err)
			}
			detector.clock = newMockClock(baseTime)

			if got := detector.IsSaturated(context.Background()); got != tt.expectedSaturat {
				t.Errorf("IsSaturated() = %v, want %v", got, tt.expectedSaturat)
			}
		})
	}
}
