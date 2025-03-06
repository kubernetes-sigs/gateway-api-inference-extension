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

package triton

import (
	"testing"

	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/proto"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func TestPromToPodMetrics(t *testing.T) {
	logger := logutil.NewTestLogger()

	podName := "test-pod"
	podAddress := "10.0.0.1"

	testCases := []struct {
		name              string
		metricFamilies    map[string]*dto.MetricFamily
		expectedMetrics   *datastore.PodMetrics
		expectedErr       bool
		initialPodMetrics *datastore.PodMetrics
	}{
		{
			name:           "all metrics available",
			metricFamilies: allMetricsAvailable(podName),
			expectedMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					NamespacedName: types.NamespacedName{Name: podName},
					Address:        podAddress,
				},
				Metrics: datastore.Metrics{
					RunningQueueSize:    1,
					WaitingQueueSize:    2,
					KVCacheUsagePercent: 0.5, // used / max = 50 / 100
				},
			},
			initialPodMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					NamespacedName: types.NamespacedName{Name: podName},
					Address:        podAddress,
				},
				Metrics: datastore.Metrics{},
			},
			expectedErr: false,
		},
		{
			name:           "missing metrics",
			metricFamilies: map[string]*dto.MetricFamily{}, // No metrics provided
			expectedMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					NamespacedName: types.NamespacedName{Name: podName},
					Address:        podAddress,
				},
				Metrics: datastore.Metrics{
					RunningQueueSize:    0, // Default int value
					WaitingQueueSize:    0, // Default int value
					KVCacheUsagePercent: 0, // Default float64 value
				},
			},
			initialPodMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					NamespacedName: types.NamespacedName{Name: podName},
					Address:        podAddress,
				},
				Metrics: datastore.Metrics{},
			},
			expectedErr: false,
		},
		{
			name:           "multiple timestamps",
			metricFamilies: multipleMetricsWithDifferentTimestamps(podName),
			expectedMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					NamespacedName: types.NamespacedName{Name: podName},
					Address:        podAddress,
				},
				Metrics: datastore.Metrics{
					RunningQueueSize:    1,   // from latest
					WaitingQueueSize:    2,   // from latest
					KVCacheUsagePercent: 0.5, // used / max = 50 / 100  (from latest)
				},
			},
			initialPodMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					NamespacedName: types.NamespacedName{Name: podName},
					Address:        podAddress,
				},
				Metrics: datastore.Metrics{},
			},
			expectedErr: false,
		},
		{
			name: "empty metric family",
			metricFamilies: map[string]*dto.MetricFamily{
				TRTLLMRequestMetricsName: {
					Name:   proto.String(TRTLLMRequestMetricsName),
					Type:   dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{}, // Empty
				},
			},
			expectedMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					NamespacedName: types.NamespacedName{Name: podName},
					Address:        podAddress,
				},
				Metrics: datastore.Metrics{},
			},
			initialPodMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					NamespacedName: types.NamespacedName{Name: podName},
					Address:        podAddress,
				},
				Metrics: datastore.Metrics{},
			},
			expectedErr: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			updated, err := promToPodMetrics(logger, tc.metricFamilies, tc.initialPodMetrics)
			if tc.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expectedMetrics, updated)
			}
		})
	}
}

// --- Helper Functions ---

func allMetricsAvailable(podName string) map[string]*dto.MetricFamily {
	return map[string]*dto.MetricFamily{
		TRTLLMRequestMetricsName: {
			Name: proto.String(TRTLLMRequestMetricsName),
			Type: dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{
				trtLlmRequestMetric("active", 1, 200),
				trtLlmRequestMetric("scheduled", 2, 200),
			},
		},
		TRTLLMKvCacheMetricsName: {
			Name: proto.String(TRTLLMKvCacheMetricsName),
			Type: dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{
				trtLlmKvCacheMetric("max", 100, 200),
				trtLlmKvCacheMetric("used", 50, 200),
				trtLlmKvCacheMetric("tokens_per", 50, 200),
			},
		},
	}
}

func multipleMetricsWithDifferentTimestamps(podName string) map[string]*dto.MetricFamily {
	return map[string]*dto.MetricFamily{
		TRTLLMRequestMetricsName: {
			Name: proto.String(TRTLLMRequestMetricsName),
			Type: dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{
				trtLlmRequestMetric("active", 0, 100),    // Older
				trtLlmRequestMetric("scheduled", 3, 100), // Older
				trtLlmRequestMetric("active", 1, 200),    // Newer
				trtLlmRequestMetric("scheduled", 2, 200), // Newer

			},
		},
		TRTLLMKvCacheMetricsName: {
			Name: proto.String(TRTLLMKvCacheMetricsName),
			Type: dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{
				trtLlmKvCacheMetric("max", 110, 100),       //Older
				trtLlmKvCacheMetric("used", 60, 100),       //Older
				trtLlmKvCacheMetric("tokens_per", 40, 100), //Older
				trtLlmKvCacheMetric("max", 100, 200),       // Newer
				trtLlmKvCacheMetric("used", 50, 200),       // Newer
				trtLlmKvCacheMetric("tokens_per", 50, 200), // Newer
			},
		},
	}
}

func trtLlmRequestMetric(requestType string, value float64, timestampMs int64) *dto.Metric {
	return &dto.Metric{
		Label: []*dto.LabelPair{
			{Name: proto.String(TRTLLMRequestMetricsLabel), Value: proto.String(requestType)},
		},
		Gauge:       &dto.Gauge{Value: &value},
		TimestampMs: &timestampMs,
	}
}

func trtLlmKvCacheMetric(blockType string, value float64, timestampMs int64) *dto.Metric {
	return &dto.Metric{
		Label: []*dto.LabelPair{
			{Name: proto.String(TRTLLMKvCacheMetricsLabel), Value: proto.String(blockType)},
		},
		Gauge:       &dto.Gauge{Value: &value},
		TimestampMs: &timestampMs,
	}
}
