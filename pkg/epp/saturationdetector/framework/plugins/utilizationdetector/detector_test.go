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

package utilizationdetector

import (
	"context"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

func makePodMetric(name string, queueDepth int, kvUsage float64, updateTime time.Time) *backendmetrics.FakePodMetrics {
	return &backendmetrics.FakePodMetrics{
		Metadata: &fwkdl.EndpointMetadata{
			NamespacedName: types.NamespacedName{Name: name, Namespace: "ns1"},
		},
		Metrics: &backendmetrics.MetricsState{
			WaitingQueueSize:    queueDepth,
			KVCacheUsagePercent: kvUsage,
			UpdateTime:          updateTime,
		},
	}
}

func TestDetector_Saturation(t *testing.T) {
	t.Parallel()

	baseTime := time.Now()

	// Config: Queue=5, KV=0.9
	config := &Config{
		QueueDepthThreshold:       5,
		KVCacheUtilThreshold:      0.90,
		MetricsStalenessThreshold: 100 * time.Millisecond,
	}

	tests := []struct {
		name           string
		pods           []backendmetrics.PodMetrics
		wantSaturation float64
	}{
		{
			name:           "No candidate pods",
			pods:           []backendmetrics.PodMetrics{},
			wantSaturation: 1.0, // Fail closed
		},
		{
			name: "Single pod with good capacity",
			pods: []backendmetrics.PodMetrics{
				// Q=2/5 (0.4). KV=0.5/0.9 (0.555...).
				// Max(0.4, 0.555...) = 0.555...
				makePodMetric("pod1", 2, 0.5, baseTime),
			},
			wantSaturation: 0.5 / 0.9,
		},
		{
			name: "Single pod with stale metrics",
			pods: []backendmetrics.PodMetrics{
				// Stale = 1.0
				makePodMetric("pod1", 1, 0.1, baseTime.Add(-200*time.Millisecond)),
			},
			wantSaturation: 1.0,
		},
		{
			name: "Single pod with high queue depth",
			pods: []backendmetrics.PodMetrics{
				// Q=10/5 (2.0). KV=0.1/0.9 (0.11).
				// Max(2.0, 0.11) = 2.0
				makePodMetric("pod1", 10, 0.1, baseTime),
			},
			wantSaturation: 2.0,
		},
		{
			name: "Single pod with high KV cache utilization",
			pods: []backendmetrics.PodMetrics{
				// Q=1/5 (0.2). KV=0.95/0.90 (1.055...).
				// Max(0.2, 1.055...) = 1.055...
				makePodMetric("pod1", 1, 0.95, baseTime),
			},
			wantSaturation: 0.95 / 0.90,
		},
		{
			name: "Single pod with nil metrics",
			pods: []backendmetrics.PodMetrics{
				&backendmetrics.FakePodMetrics{
					Metadata: &fwkdl.EndpointMetadata{
						NamespacedName: types.NamespacedName{Name: "pod1", Namespace: "ns1"},
					},
					Metrics: nil,
				},
			},
			wantSaturation: 1.0,
		},
		{
			name: "Multiple pods, all good capacity",
			pods: []backendmetrics.PodMetrics{
				// Pod1: Q=1/5(0.2), KV=0.1/0.9(0.11). Max=0.2.
				makePodMetric("pod1", 1, 0.1, baseTime),
				// Pod2: Q=0/5(0.0), KV=0.2/0.9(0.22). Max=0.22...
				makePodMetric("pod2", 0, 0.2, baseTime),
			},
			// Avg(0.2, 0.222...) = 0.2111...
			wantSaturation: (0.2 + (0.2 / 0.9)) / 2.0,
		},
		{
			name: "Multiple pods, one good, one stale",
			pods: []backendmetrics.PodMetrics{
				// Pod1 (Good): Q=1/5(0.2), KV=0.1/0.9(0.11). Max=0.2.
				makePodMetric("pod1", 1, 0.1, baseTime),
				// Pod2 (Stale): 1.0.
				makePodMetric("pod2", 0, 0.2, baseTime.Add(-300*time.Millisecond)),
			},
			// Avg(0.2, 1.0) = 0.6
			wantSaturation: 0.6,
		},
		{
			name: "Multiple pods, one good, one bad (high queue)",
			pods: []backendmetrics.PodMetrics{
				// Pod1 (Good): Max=0.2.
				makePodMetric("pod1", 1, 0.1, baseTime),
				// Pod2 (Bad): Q=15/5(3.0). Max=3.0.
				makePodMetric("pod2", 15, 0.2, baseTime),
			},
			// Avg(0.2, 3.0) = 1.6
			wantSaturation: 1.6,
		},
		{
			name: "Multiple pods, all bad capacity",
			pods: []backendmetrics.PodMetrics{
				// Pod1 (Stale): 1.0
				makePodMetric("pod1", 1, 0.1, baseTime.Add(-200*time.Millisecond)),
				// Pod2 (High Q): 20/5 = 4.0
				makePodMetric("pod2", 20, 0.2, baseTime),
				// Pod3 (High KV): 0.99/0.90 = 1.1
				makePodMetric("pod3", 1, 0.99, baseTime),
			},
			// Avg(1.0, 4.0, 1.1) = 6.1 / 3 = 2.033...
			wantSaturation: (1.0 + 4.0 + 1.1) / 3.0,
		},
		{
			name: "Queue depth exactly at threshold",
			pods: []backendmetrics.PodMetrics{
				// Q=5/5(1.0). KV=Low.
				// Max=1.0
				makePodMetric("pod1", 5, 0.1, baseTime),
			},
			wantSaturation: 1.0,
		},
		{
			name: "Metrics age just over staleness threshold",
			pods: []backendmetrics.PodMetrics{
				makePodMetric("pod1", 1, 0.1, baseTime.Add(-101*time.Millisecond)),
			},
			wantSaturation: 1.0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			detector := NewDetector(config, logr.Discard())

			got := detector.Saturation(context.Background(), tc.pods)
			require.InDelta(t, tc.wantSaturation, got, 1e-4, "Saturation mismatch")
		})
	}
}
