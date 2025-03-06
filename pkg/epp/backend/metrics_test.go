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

package backend

import (
	"context"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"

	dto "github.com/prometheus/client_model/go"
	"go.uber.org/multierr"
	"google.golang.org/protobuf/proto"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// --- Test Helpers ---

func makeMetric(metricName string, labels map[string]string, value float64, timestampMs int64) *dto.Metric {
	labelPairs := []*dto.LabelPair{}
	for k, v := range labels {
		labelPairs = append(labelPairs, &dto.LabelPair{Name: proto.String(k), Value: proto.String(v)})
	}
	return &dto.Metric{
		Label:       labelPairs,
		Gauge:       &dto.Gauge{Value: &value},
		TimestampMs: &timestampMs,
	}
}

func makeMetricFamily(name string, metrics ...*dto.Metric) *dto.MetricFamily {
	return &dto.MetricFamily{
		Name:   &name,
		Type:   dto.MetricType_GAUGE.Enum(),
		Metric: metrics,
	}
}

// --- Tests ---

func TestGetMetric(t *testing.T) {
	logger := logutil.NewTestLogger()

	metricFamilies := map[string]*dto.MetricFamily{
		"metric1": makeMetricFamily("metric1",
			makeMetric("metric1", map[string]string{"label1": "value1"}, 1.0, 1000),
			makeMetric("metric1", map[string]string{"label1": "value2"}, 2.0, 2000),
		),
		"metric2": makeMetricFamily("metric2",
			makeMetric("metric2", map[string]string{"labelA": "A1", "labelB": "B1"}, 3.0, 1500),
			makeMetric("metric2", map[string]string{"labelA": "A2", "labelB": "B2"}, 4.0, 2500),
		),
		"metric3": makeMetricFamily("metric3",
			makeMetric("metric3", map[string]string{}, 5.0, 3000),
			makeMetric("metric3", map[string]string{}, 6.0, 1000),
		),
	}

	tests := []struct {
		name        string
		spec        MetricSpec
		wantValue   float64
		wantError   bool
		shouldPanic bool // Add this
	}{
		{
			name: "get labeled metric, exists",
			spec: MetricSpec{
				MetricName: "metric1",
				Labels:     map[string]string{"label1": "value1"},
			},
			wantValue: 1.0,
			wantError: false,
		},
		{
			name: "get labeled metric, wrong value",
			spec: MetricSpec{
				MetricName: "metric1",
				Labels:     map[string]string{"label1": "value3"},
			},
			wantValue: -1, // Expect an error, not a specific value
			wantError: true,
		},
		{
			name: "get labeled metric, missing label",
			spec: MetricSpec{
				MetricName: "metric1",
				Labels:     map[string]string{"label2": "value2"},
			},
			wantValue: -1,
			wantError: true,
		},
		{
			name: "get labeled metric, extra label present",
			spec: MetricSpec{
				MetricName: "metric2",
				Labels:     map[string]string{"labelA": "A1"},
			},
			wantValue: 3.0,
			wantError: false,
		},
		{
			name: "get unlabeled metric, exists",
			spec: MetricSpec{
				MetricName: "metric3",
				Labels:     nil, // Explicitly nil
			},
			wantValue: 5.0, // latest metric, which occurs first in our test data
			wantError: false,
		},
		{
			name: "get unlabeled metric, metric family not found",
			spec: MetricSpec{
				MetricName: "metric4",
				Labels:     nil,
			},
			wantValue: -1,
			wantError: true,
		},
		{
			name: "get labeled metric, metric family not found",
			spec: MetricSpec{
				MetricName: "metric4",
				Labels:     map[string]string{"label1": "value1"},
			},
			wantValue: -1,
			wantError: true,
		},
		{
			name: "get metric, no metrics available",
			spec: MetricSpec{
				MetricName: "empty_metric",
			},
			wantValue: -1,
			wantError: true,
		},
		{
			name: "get latest metric",
			spec: MetricSpec{
				MetricName: "metric3",
				Labels:     map[string]string{}, // Empty map, not nil
			},
			wantValue: 5.0,
			wantError: false,
		},
	}

	p := &PodMetricsClientImpl{} // No need for MetricMapping here

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.shouldPanic {
				defer func() {
					if r := recover(); r == nil {
						t.Errorf("The code did not panic")
					}
				}()
			}

			gotMetric, err := p.getMetric(logger, metricFamilies, tt.spec)

			if tt.wantError {
				if err == nil {
					t.Errorf("getMetric() expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("getMetric() unexpected error: %v", err)
				}
				if gotMetric.GetGauge().GetValue() != tt.wantValue {
					t.Errorf("getMetric() got value %v, want %v", gotMetric.GetGauge().GetValue(), tt.wantValue)
				}
			}
		})
	}
}

func TestLabelsMatch(t *testing.T) {
	tests := []struct {
		name         string
		metricLabels []*dto.LabelPair
		specLabels   map[string]string
		want         bool
	}{
		{
			name:         "empty spec labels, should match",
			metricLabels: []*dto.LabelPair{{Name: proto.String("a"), Value: proto.String("b")}},
			specLabels:   map[string]string{},
			want:         true,
		},
		{
			name:         "nil spec labels, should match",
			metricLabels: []*dto.LabelPair{{Name: proto.String("a"), Value: proto.String("b")}},
			specLabels:   nil,
			want:         true,
		},
		{
			name:         "exact match",
			metricLabels: []*dto.LabelPair{{Name: proto.String("a"), Value: proto.String("b")}},
			specLabels:   map[string]string{"a": "b"},
			want:         true,
		},
		{
			name:         "extra labels in metric",
			metricLabels: []*dto.LabelPair{{Name: proto.String("a"), Value: proto.String("b")}, {Name: proto.String("c"), Value: proto.String("d")}},
			specLabels:   map[string]string{"a": "b"},
			want:         true,
		},
		{
			name:         "missing label in metric",
			metricLabels: []*dto.LabelPair{{Name: proto.String("a"), Value: proto.String("b")}},
			specLabels:   map[string]string{"a": "b", "c": "d"},
			want:         false,
		},
		{
			name:         "value mismatch",
			metricLabels: []*dto.LabelPair{{Name: proto.String("a"), Value: proto.String("b")}},
			specLabels:   map[string]string{"a": "c"},
			want:         false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := labelsMatch(tt.metricLabels, tt.specLabels); got != tt.want {
				t.Errorf("labelsMatch() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetLatestLoraMetric(t *testing.T) {
	logger := logutil.NewTestLogger()

	testCases := []struct {
		name             string
		metricFamilies   map[string]*dto.MetricFamily
		expectedAdapters map[string]int
		expectedMax      int
		expectedErr      error
		mapping          *MetricMapping
	}{
		{
			name: "no lora metrics",
			metricFamilies: map[string]*dto.MetricFamily{
				"some_other_metric": makeMetricFamily("some_other_metric",
					makeMetric("some_other_metric", nil, 1.0, 1000),
				),
			},
			expectedAdapters: nil,
			expectedMax:      0,
			expectedErr:      fmt.Errorf("metric family \"vllm:lora_requests_info\" not found"), // Expect an error because the family is missing
			mapping: &MetricMapping{
				LoraRequestInfo: &MetricSpec{MetricName: "vllm:lora_requests_info"},
			},
		},
		{
			name: "basic lora metrics",
			metricFamilies: map[string]*dto.MetricFamily{
				"vllm:lora_requests_info": makeMetricFamily("vllm:lora_requests_info",
					makeMetric("vllm:lora_requests_info", map[string]string{"running_lora_adapters": "lora1", "max_lora": "2"}, 3000.0, 1000),       // Newer
					makeMetric("vllm:lora_requests_info", map[string]string{"running_lora_adapters": "lora2,lora3", "max_lora": "4"}, 1000.0, 1000), // Older

				),
			},
			expectedAdapters: map[string]int{"lora1": 0},
			expectedMax:      2,
			expectedErr:      nil,
			mapping: &MetricMapping{
				LoraRequestInfo: &MetricSpec{MetricName: "vllm:lora_requests_info"},
			},
		},
		{
			name: "no matching lora metrics",
			metricFamilies: map[string]*dto.MetricFamily{
				"vllm:lora_requests_info": makeMetricFamily("vllm:lora_requests_info",
					makeMetric("vllm:lora_requests_info", map[string]string{"other_label": "value"}, 5.0, 3000),
				),
			},
			expectedAdapters: nil,
			expectedMax:      0,
			expectedErr:      nil, // Expect *no* error; just no adapters found
			mapping: &MetricMapping{
				LoraRequestInfo: &MetricSpec{MetricName: "vllm:lora_requests_info"},
			},
		},
		{
			name: "no lora metrics if not in MetricMapping",
			metricFamilies: map[string]*dto.MetricFamily{
				"vllm:lora_requests_info": makeMetricFamily("vllm:lora_requests_info",
					makeMetric("vllm:lora_requests_info", map[string]string{"running_lora_adapters": "lora1", "max_lora": "2"}, 5.0, 3000),
					makeMetric("vllm:lora_requests_info", map[string]string{"running_lora_adapters": "lora2,lora3", "max_lora": "4"}, 6.0, 1000),
				),
			},
			expectedAdapters: nil,
			expectedMax:      0,
			expectedErr:      nil,
			mapping:          &MetricMapping{ // No LoRA metrics defined
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			p := &PodMetricsClientImpl{MetricMapping: tc.mapping}
			loraMetric, _, err := p.getLatestLoraMetric(logger, tc.metricFamilies)

			if tc.expectedErr != nil {
				if err == nil || err.Error() != tc.expectedErr.Error() {
					t.Errorf("getLatestLoraMetric() error = %v, wantErr %v", err, tc.expectedErr)
				}
				return // Stop here if an error was expected
			} else if err != nil {
				t.Fatalf("getLatestLoraMetric() unexpected error: %v", err)
			}

			if tc.mapping.LoraRequestInfo == nil {
				if loraMetric != nil {
					t.Errorf("getLatestLoraMetric() expected nil metric, got %v", loraMetric)
				}
				return // Stop if no Lora metrics are expected.
			}

			if tc.expectedAdapters == nil && loraMetric == nil {
				return // Both nil, as expected
			}

			if tc.expectedAdapters != nil && loraMetric != nil { // proceed with checks

				adaptersFound := make(map[string]int)
				maxLora := 0
				for _, label := range loraMetric.GetLabel() {
					if label.GetName() == "running_lora_adapters" && label.GetValue() != "" {
						for _, adapter := range strings.Split(label.GetValue(), ",") {
							adaptersFound[adapter] = 0
						}
					}
					if label.GetName() == "waiting_lora_adapters" && label.GetValue() != "" {
						for _, adapter := range strings.Split(label.GetValue(), ",") {
							adaptersFound[adapter] = 0 // Overwrite if already present
						}
					}
					if label.GetName() == "max_lora" {
						var converr error // define err in this scope.
						maxLora, converr = strconv.Atoi(label.GetValue())
						if converr != nil && tc.expectedErr == nil { // only report if we don't expect any other errors
							t.Errorf("getLatestLoraMetric() could not parse max_lora: %v", converr)
						}
					}
				}

				if !reflect.DeepEqual(adaptersFound, tc.expectedAdapters) {
					t.Errorf("getLatestLoraMetric() adapters = %v, want %v", adaptersFound, tc.expectedAdapters)
				}
				if maxLora != tc.expectedMax {
					t.Errorf("getLatestLoraMetric() maxLora = %v, want %v", maxLora, tc.expectedMax)
				}
			} else { // one is nil and the other is not
				t.Errorf("getLatestLoraMetric(): one of expectedAdapters/loraMetric is nil and the other is not, expected %v, got %v", tc.expectedAdapters, loraMetric)
			}
		})
	}
}

func TestPromToPodMetrics(t *testing.T) {
	logger := logutil.NewTestLogger()

	tests := []struct {
		name             string
		metricFamilies   map[string]*dto.MetricFamily
		mapping          *MetricMapping
		existingMetrics  *datastore.PodMetrics
		expectedMetrics  *datastore.PodMetrics
		expectedErrCount int // Count of expected errors
	}{
		{
			name: "vllm metrics",
			metricFamilies: map[string]*dto.MetricFamily{
				"vllm_running": makeMetricFamily("vllm_running",
					makeMetric("vllm_running", nil, 10.0, 2000),
					makeMetric("vllm_running", nil, 12.0, 1000), //Older
				),
				"vllm_waiting": makeMetricFamily("vllm_waiting",
					makeMetric("vllm_waiting", nil, 5.0, 1000),
					makeMetric("vllm_waiting", nil, 7.0, 2000), // Newer
				),
				"vllm_usage": makeMetricFamily("vllm_usage",
					makeMetric("vllm_usage", nil, 0.8, 2000),
					makeMetric("vllm_usage", nil, 0.7, 500),
				),
				"vllm:lora_requests_info": makeMetricFamily("vllm:lora_requests_info",
					makeMetric("vllm:lora_requests_info", map[string]string{"running_lora_adapters": "lora1,lora2", "waiting_lora_adapters": "lora3", "max_lora": "3"}, 5.0, 3000),
				),
			},
			mapping: &MetricMapping{
				RunningRequests: &MetricSpec{MetricName: "vllm_running"},
				WaitingRequests: &MetricSpec{MetricName: "vllm_waiting"},
				KVCacheUsage:    &MetricSpec{MetricName: "vllm_usage"},
				LoraRequestInfo: &MetricSpec{MetricName: "vllm:lora_requests_info"},
			},
			existingMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					Address: "127.0.0.1",
					NamespacedName: types.NamespacedName{
						Namespace: "test",
						Name:      "pod",
					},
				},
				Metrics: datastore.Metrics{}, // Initialize with empty Metrics
			},
			expectedMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					Address: "127.0.0.1",
					NamespacedName: types.NamespacedName{
						Namespace: "test",
						Name:      "pod",
					},
				},
				Metrics: datastore.Metrics{
					RunningQueueSize:    10,
					WaitingQueueSize:    7,
					KVCacheUsagePercent: 0.8,
					ActiveModels:        map[string]int{"lora1": 0, "lora2": 0, "lora3": 0},
					MaxActiveModels:     3,
				},
			},
			expectedErrCount: 0,
		},
		{
			name: "triton metrics",
			metricFamilies: map[string]*dto.MetricFamily{
				"triton_running": makeMetricFamily("triton_running",
					makeMetric("triton_running", map[string]string{"queue": "fast"}, 10.0, 2000),
					makeMetric("triton_running", map[string]string{"queue": "slow"}, 12.0, 1000), //Older, but different label
				),
				"triton_all": makeMetricFamily("triton_all",
					makeMetric("triton_all", map[string]string{"queue": "fast"}, 15.0, 1000),
					makeMetric("triton_all", map[string]string{"queue": "fast"}, 17.0, 2000), // Newer
				),
				"triton_used": makeMetricFamily("triton_used",
					makeMetric("triton_used", map[string]string{"type": "gpu"}, 80.0, 1000),
				),
				"triton_max": makeMetricFamily("triton_max",
					makeMetric("triton_max", map[string]string{"type": "gpu"}, 100.0, 1000),
				),
			},
			mapping: &MetricMapping{
				RunningRequests:   &MetricSpec{MetricName: "triton_running", Labels: map[string]string{"queue": "fast"}},
				AllRequests:       &MetricSpec{MetricName: "triton_all", Labels: map[string]string{"queue": "fast"}},
				UsedKVCacheBlocks: &MetricSpec{MetricName: "triton_used", Labels: map[string]string{"type": "gpu"}},
				MaxKVCacheBlocks:  &MetricSpec{MetricName: "triton_max", Labels: map[string]string{"type": "gpu"}},
			},
			existingMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					Address: "127.0.0.1",
					NamespacedName: types.NamespacedName{
						Namespace: "test",
						Name:      "pod",
					},
				},
				Metrics: datastore.Metrics{
					ActiveModels: map[string]int{},
				}, // Initialize with empty Metrics
			},
			expectedMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					Address: "127.0.0.1",
					NamespacedName: types.NamespacedName{
						Namespace: "test",
						Name:      "pod",
					},
				},
				Metrics: datastore.Metrics{
					ActiveModels:        map[string]int{},
					RunningQueueSize:    10,
					WaitingQueueSize:    7,   // 17 (all) - 10 (running)
					KVCacheUsagePercent: 0.8, // 80 / 100
				},
			},
			expectedErrCount: 0,
		},
		{
			name: "triton metrics, missing label",
			metricFamilies: map[string]*dto.MetricFamily{
				"triton_running": makeMetricFamily("triton_running",
					makeMetric("triton_running", map[string]string{"queue": "fast"}, 10.0, 2000),
				),
				"triton_all": makeMetricFamily("triton_all",
					makeMetric("triton_all", map[string]string{"queue": "fast"}, 17.0, 2000),
				),
				// triton_used and _max have no metrics with type=gpu label.
			},
			mapping: &MetricMapping{
				RunningRequests:   &MetricSpec{MetricName: "triton_running", Labels: map[string]string{"queue": "fast"}},
				AllRequests:       &MetricSpec{MetricName: "triton_all", Labels: map[string]string{"queue": "fast"}},
				UsedKVCacheBlocks: &MetricSpec{MetricName: "triton_used", Labels: map[string]string{"type": "gpu"}},
				MaxKVCacheBlocks:  &MetricSpec{MetricName: "triton_max", Labels: map[string]string{"type": "gpu"}},
			},
			existingMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					Address: "127.0.0.1",
					NamespacedName: types.NamespacedName{
						Namespace: "test",
						Name:      "pod",
					},
				},
				Metrics: datastore.Metrics{
					ActiveModels: map[string]int{},
				}, // Initialize with empty Metrics
			},
			expectedMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					Address: "127.0.0.1",
					NamespacedName: types.NamespacedName{
						Namespace: "test",
						Name:      "pod",
					},
				},
				Metrics: datastore.Metrics{
					ActiveModels:        map[string]int{},
					RunningQueueSize:    10,
					WaitingQueueSize:    7,
					KVCacheUsagePercent: 0.0, // expect this to still be present, but with default 0 value
				},
			},

			expectedErrCount: 2, // Two errors:  Used and Max
		},
		{
			name:           "missing metrics",
			metricFamilies: map[string]*dto.MetricFamily{}, // No metrics
			mapping: &MetricMapping{
				RunningRequests: &MetricSpec{MetricName: "vllm_running"},
				WaitingRequests: &MetricSpec{MetricName: "vllm_waiting"},
				KVCacheUsage:    &MetricSpec{MetricName: "vllm_usage"},
				LoraRequestInfo: &MetricSpec{MetricName: "vllm:lora_requests_info"},
			},
			existingMetrics:  &datastore.PodMetrics{Metrics: datastore.Metrics{ActiveModels: map[string]int{}}},
			expectedMetrics:  &datastore.PodMetrics{Metrics: datastore.Metrics{ActiveModels: map[string]int{}}},
			expectedErrCount: 4, // Errors for all 4 main metrics
		},
		{
			name: "partial metrics available + LoRA",
			metricFamilies: map[string]*dto.MetricFamily{
				"vllm_usage": makeMetricFamily("vllm_usage",
					makeMetric("vllm_usage", nil, 0.8, 2000), // Only usage is present
				),
				"vllm:lora_requests_info": makeMetricFamily("vllm:lora_requests_info",
					makeMetric("vllm:lora_requests_info", map[string]string{"running_lora_adapters": "lora1,lora2", "waiting_lora_adapters": "lora3", "max_lora": "3"}, 5.0, 3000),
				),
			},
			mapping: &MetricMapping{
				RunningRequests: &MetricSpec{MetricName: "vllm_running"}, // Not present
				WaitingRequests: &MetricSpec{MetricName: "vllm_waiting"}, // Not Present
				KVCacheUsage:    &MetricSpec{MetricName: "vllm_usage"},
				LoraRequestInfo: &MetricSpec{MetricName: "vllm:lora_requests_info"},
			},
			existingMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					Address: "127.0.0.1",
					NamespacedName: types.NamespacedName{
						Namespace: "test",
						Name:      "pod",
					},
				},
				Metrics: datastore.Metrics{}, // Initialize with empty Metrics
			},
			expectedMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					Address: "127.0.0.1",
					NamespacedName: types.NamespacedName{
						Namespace: "test",
						Name:      "pod",
					},
				},
				Metrics: datastore.Metrics{
					RunningQueueSize:    0,
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.8,
					ActiveModels:        map[string]int{"lora1": 0, "lora2": 0, "lora3": 0},
					MaxActiveModels:     3,
				},
			},
			expectedErrCount: 2, // Errors for the two missing metrics
		},
		{
			name: "use all requests for waiting queue",
			metricFamilies: map[string]*dto.MetricFamily{
				"vllm_running": makeMetricFamily("vllm_running",
					makeMetric("vllm_running", nil, 10.0, 2000),
				),
				"vllm_all": makeMetricFamily("vllm_all",
					makeMetric("vllm_all", nil, 15.0, 1000),
				),
			},
			mapping: &MetricMapping{
				RunningRequests: &MetricSpec{MetricName: "vllm_running"},
				AllRequests:     &MetricSpec{MetricName: "vllm_all"},
				// No WaitingRequests
			},
			existingMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					Address: "127.0.0.1",
					NamespacedName: types.NamespacedName{
						Namespace: "test",
						Name:      "pod",
					},
				},
				Metrics: datastore.Metrics{
					ActiveModels: map[string]int{},
				}, // Initialize with empty Metrics
			},
			expectedMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					Address: "127.0.0.1",
					NamespacedName: types.NamespacedName{
						Namespace: "test",
						Name:      "pod",
					},
				},
				Metrics: datastore.Metrics{
					ActiveModels:     map[string]int{},
					RunningQueueSize: 10,
					WaitingQueueSize: 5, // 15 - 10
				},
			},
			expectedErrCount: 0,
		},
		{
			name: "invalid max lora",
			metricFamilies: map[string]*dto.MetricFamily{
				"vllm:lora_requests_info": makeMetricFamily("vllm:lora_requests_info",
					makeMetric("vllm:lora_requests_info", map[string]string{"running_lora_adapters": "lora1", "max_lora": "invalid"}, 3000.0, 1000),
				),
			},
			mapping: &MetricMapping{
				LoraRequestInfo: &MetricSpec{MetricName: "vllm:lora_requests_info"},
			},
			existingMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					Address: "127.0.0.1",
					NamespacedName: types.NamespacedName{
						Namespace: "test",
						Name:      "pod",
					},
				},
				Metrics: datastore.Metrics{},
			},
			expectedMetrics: &datastore.PodMetrics{
				Pod: datastore.Pod{
					Address: "127.0.0.1",
					NamespacedName: types.NamespacedName{
						Namespace: "test",
						Name:      "pod",
					},
				},
				Metrics: datastore.Metrics{
					ActiveModels:    map[string]int{"lora1": 0},
					MaxActiveModels: 0, // Should still default to 0.

				},
			},
			expectedErrCount: 1, // Expect *one* error
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := &PodMetricsClientImpl{MetricMapping: tc.mapping}
			updated, err := p.promToPodMetrics(logger, tc.metricFamilies, tc.existingMetrics)

			if tc.expectedErrCount == 0 {
				if err != nil {
					t.Errorf("promToPodMetrics() unexpected error: %v", err)
				}
			} else {
				if err == nil {
					t.Errorf("promToPodMetrics() expected errors, got nil")
				} else {
					// Check the *number* of errors.  multierr.Errors() gives us a slice
					if len(multierr.Errors(err)) != tc.expectedErrCount {
						t.Errorf("promToPodMetrics() wrong number of errors: got %d, want %d.  Errors: %v", len(multierr.Errors(err)), tc.expectedErrCount, err)
					}

				}
			}
			// Use podMetricsEqual for comparison with tolerance.
			if !reflect.DeepEqual(updated, tc.expectedMetrics) {
				t.Errorf("promToPodMetrics() got %+v, want %+v", updated, tc.expectedMetrics)
			}
		})
	}
}

// TestFetchMetrics is a basic integration test.  A more complete test would mock
// the HTTP client.
func TestFetchMetrics(t *testing.T) {
	// This test is very basic as it doesn't mock the HTTP client.  It assumes
	// there's no server running on the specified port.  A real-world test
	// suite should use a mock server.
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	existing := &datastore.PodMetrics{
		Pod: datastore.Pod{
			Address: "127.0.0.1",
			NamespacedName: types.NamespacedName{
				Namespace: "test",
				Name:      "pod",
			},
		},
	}
	p := &PodMetricsClientImpl{} // No MetricMapping needed for this basic test

	_, err := p.FetchMetrics(ctx, existing, 9999) // Use a port that's unlikely to be in use.
	if err == nil {
		t.Errorf("FetchMetrics() expected error, got nil")
	}
	// Check for a specific error message (fragile, but OK for this example)
	expectedSubstr := "connection refused"
	if err != nil && !strings.Contains(err.Error(), expectedSubstr) {
		t.Errorf("FetchMetrics() error = %v, want error containing %q", err, expectedSubstr)
	}
}
