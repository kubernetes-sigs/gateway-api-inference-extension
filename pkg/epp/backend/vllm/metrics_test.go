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

package vllm

import (
	"errors"
	"testing"

	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/proto"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func TestPromToPodMetrics(t *testing.T) {
	logger := logutil.NewTestLogger()

	testCases := []struct {
		name            string
		metricFamilies  map[string]*dto.MetricFamily
		initialMetrics  *metrics.Metrics
		expectedMetrics *metrics.Metrics
		expectedErr     error
	}{
		{
			name: "all metrics available",
			metricFamilies: map[string]*dto.MetricFamily{
				RunningQueueSizeMetricName: {
					Metric: []*dto.Metric{
						{
							Gauge: &dto.Gauge{
								Value: proto.Float64(10),
							},
							TimestampMs: proto.Int64(100),
						},
						{
							Gauge: &dto.Gauge{
								Value: proto.Float64(15),
							},
							TimestampMs: proto.Int64(200), // This is the latest
						},
					},
				},
				WaitingQueueSizeMetricName: {
					Metric: []*dto.Metric{
						{
							Gauge: &dto.Gauge{
								Value: proto.Float64(20),
							},
							TimestampMs: proto.Int64(100),
						},
						{
							Gauge: &dto.Gauge{
								Value: proto.Float64(25),
							},
							TimestampMs: proto.Int64(200), // This is the latest
						},
					},
				},
				KVCacheUsagePercentMetricName: {
					Metric: []*dto.Metric{
						{
							Gauge: &dto.Gauge{
								Value: proto.Float64(0.8),
							},
							TimestampMs: proto.Int64(100),
						},
						{
							Gauge: &dto.Gauge{
								Value: proto.Float64(0.9),
							},
							TimestampMs: proto.Int64(200), // This is the latest
						},
					},
				},
				LoraRequestInfoMetricName: {
					Metric: []*dto.Metric{
						{
							Label: []*dto.LabelPair{
								{
									Name:  proto.String(LoraRequestInfoRunningAdaptersMetricName),
									Value: proto.String("lora3,lora4"),
								},
								{
									Name:  proto.String(LoraRequestInfoMaxAdaptersMetricName),
									Value: proto.String("2"),
								},
							},
							Gauge: &dto.Gauge{
								Value: proto.Float64(100),
							},
						},
						{
							Label: []*dto.LabelPair{
								{
									Name:  proto.String(LoraRequestInfoRunningAdaptersMetricName),
									Value: proto.String("lora2"),
								},
								{
									Name:  proto.String(LoraRequestInfoMaxAdaptersMetricName),
									Value: proto.String("2"),
								},
							},
							Gauge: &dto.Gauge{
								Value: proto.Float64(90),
							},
						},
					},
				},
			},
			expectedMetrics: &metrics.Metrics{
				RunningQueueSize:    15,
				WaitingQueueSize:    25,
				KVCacheUsagePercent: 0.9,
				ActiveModels: map[string]int{
					"lora3": 0,
					"lora4": 0,
				},
				MaxActiveModels: 2,
			},
			initialMetrics: &metrics.Metrics{},
			expectedErr:    nil,
		},
		{
			name: "invalid max lora",
			metricFamilies: map[string]*dto.MetricFamily{
				RunningQueueSizeMetricName: {
					Metric: []*dto.Metric{
						{
							Gauge: &dto.Gauge{
								Value: proto.Float64(10),
							},
							TimestampMs: proto.Int64(100),
						},
						{
							Gauge: &dto.Gauge{
								Value: proto.Float64(15),
							},
							TimestampMs: proto.Int64(200), // This is the latest
						},
					},
				},
				WaitingQueueSizeMetricName: {
					Metric: []*dto.Metric{
						{
							Gauge: &dto.Gauge{
								Value: proto.Float64(20),
							},
							TimestampMs: proto.Int64(100),
						},
						{
							Gauge: &dto.Gauge{
								Value: proto.Float64(25),
							},
							TimestampMs: proto.Int64(200), // This is the latest
						},
					},
				},
				KVCacheUsagePercentMetricName: {
					Metric: []*dto.Metric{
						{
							Gauge: &dto.Gauge{
								Value: proto.Float64(0.8),
							},
							TimestampMs: proto.Int64(100),
						},
						{
							Gauge: &dto.Gauge{
								Value: proto.Float64(0.9),
							},
							TimestampMs: proto.Int64(200), // This is the latest
						},
					},
				},
				LoraRequestInfoMetricName: {
					Metric: []*dto.Metric{
						{
							Label: []*dto.LabelPair{
								{
									Name:  proto.String(LoraRequestInfoRunningAdaptersMetricName),
									Value: proto.String("lora3,lora4"),
								},
								{
									Name:  proto.String(LoraRequestInfoMaxAdaptersMetricName),
									Value: proto.String("2a"),
								},
							},
							Gauge: &dto.Gauge{
								Value: proto.Float64(100),
							},
						},
						{
							Label: []*dto.LabelPair{
								{
									Name:  proto.String(LoraRequestInfoRunningAdaptersMetricName),
									Value: proto.String("lora2"),
								},
								{
									Name:  proto.String(LoraRequestInfoMaxAdaptersMetricName),
									Value: proto.String("2"),
								},
							},
							Gauge: &dto.Gauge{
								Value: proto.Float64(90),
							},
						},
					},
				},
			},
			expectedMetrics: &metrics.Metrics{
				RunningQueueSize:    15,
				WaitingQueueSize:    25,
				KVCacheUsagePercent: 0.9,
				ActiveModels: map[string]int{
					"lora3": 0,
					"lora4": 0,
				},
				MaxActiveModels: 0,
			},
			initialMetrics: &metrics.Metrics{},
			expectedErr:    errors.New("strconv.Atoi: parsing '2a': invalid syntax"),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			updated, err := promToPodMetrics(logger, tc.metricFamilies, tc.initialMetrics)
			if tc.expectedErr != nil {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expectedMetrics, updated)
			}
		})
	}
}
