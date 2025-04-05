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

package metrics

import (
	"context"
	"errors"
	"reflect"
	"strconv"
	"strings"
	"testing"

	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"go.uber.org/multierr"
	"google.golang.org/protobuf/proto"
	"k8s.io/apimachinery/pkg/types"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// --- Test Helpers ---

func makeMetric(labels map[string]string, value float64, timestampMs int64) *dto.Metric {
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

	metricFamilies := map[string]*dto.MetricFamily{
		"metric1": makeMetricFamily("metric1",
			makeMetric(map[string]string{"label1": "value1"}, 1.0, 1000),
			makeMetric(map[string]string{"label1": "value2"}, 2.0, 2000),
		),
		"metric2": makeMetricFamily("metric2",
			makeMetric(map[string]string{"labelA": "A1", "labelB": "B1"}, 3.0, 1500),
			makeMetric(map[string]string{"labelA": "A2", "labelB": "B2"}, 4.0, 2500),
		),
		"metric3": makeMetricFamily("metric3",
			makeMetric(map[string]string{}, 5.0, 3000),
			makeMetric(map[string]string{}, 6.0, 1000),
		),
	}

	tests := []struct {
		name           string
		spec           MetricSpec
		wantGaugeValue float64
		wantError      bool
	}{
		{
			name: "get labeled metric, exists",
			spec: MetricSpec{
				MetricName: "metric1",
				Labels:     map[string]string{"label1": "value1"},
			},
			wantGaugeValue: 1.0,
			wantError:      false,
		},
		{
			name: "get labeled metric, wrong value",
			spec: MetricSpec{
				MetricName: "metric1",
				Labels:     map[string]string{"label1": "value3"},
			},
			wantGaugeValue: -1, // Expect an error, not a specific value
			wantError:      true,
		},
		{
			name: "get labeled metric, missing label",
			spec: MetricSpec{
				MetricName: "metric1",
				Labels:     map[string]string{"label2": "value2"},
			},
			wantGaugeValue: -1,
			wantError:      true,
		},
		{
			name: "get labeled metric, extra label present",
			spec: MetricSpec{
				MetricName: "metric2",
				Labels:     map[string]string{"labelA": "A1"},
			},
			wantGaugeValue: 3.0,
			wantError:      false,
		},
		{
			name: "get unlabeled metric, exists",
			spec: MetricSpec{
				MetricName: "metric3",
				Labels:     nil, // Explicitly nil
			},
			wantGaugeValue: 5.0, // latest metric, which occurs first in our test data
			wantError:      false,
		},
		{
			name: "get unlabeled metric, metric family not found",
			spec: MetricSpec{
				MetricName: "metric4",
				Labels:     nil,
			},
			wantGaugeValue: -1,
			wantError:      true,
		},
		{
			name: "get labeled metric, metric family not found",
			spec: MetricSpec{
				MetricName: "metric4",
				Labels:     map[string]string{"label1": "value1"},
			},
			wantGaugeValue: -1,
			wantError:      true,
		},
		{
			name: "get metric, no metrics available",
			spec: MetricSpec{
				MetricName: "empty_metric",
			},
			wantGaugeValue: -1,
			wantError:      true,
		},
		{
			name: "get latest metric",
			spec: MetricSpec{
				MetricName: "metric3",
				Labels:     map[string]string{}, // Empty map, not nil
			},
			wantGaugeValue: 5.0,
			wantError:      false,
		},
	}

	p := &PodMetricsClientImpl{} // No need for MetricMapping here

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			gotMetric, err := p.getMetric(metricFamilies, tt.spec)

			if tt.wantError {
				if err == nil {
					t.Errorf("getMetric() expected error, got nil")
				}
			} else {
				if err != nil {
					t.Fatalf("getMetric() unexpected error: %v", err)
				}
				if gotMetric.GetGauge().GetValue() != tt.wantGaugeValue {
					t.Errorf("getMetric() got value %v, want %v", gotMetric.GetGauge().GetValue(), tt.wantGaugeValue)
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
					makeMetric(nil, 1.0, 1000),
				),
			},
			expectedAdapters: nil,
			expectedMax:      0,
			expectedErr:      errors.New("metric family \"vllm:lora_requests_info\" not found"), // Expect an error because the family is missing
			mapping: &MetricMapping{
				LoraRequestInfo: &MetricSpec{MetricName: "vllm:lora_requests_info"},
			},
		},
		{
			name: "basic lora metrics",
			metricFamilies: map[string]*dto.MetricFamily{
				"vllm:lora_requests_info": makeMetricFamily("vllm:lora_requests_info",
					makeMetric(map[string]string{"running_lora_adapters": "lora1", "max_lora": "2"}, 3000.0, 1000),       // Newer
					makeMetric(map[string]string{"running_lora_adapters": "lora2,lora3", "max_lora": "4"}, 1000.0, 1000), // Older

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
					makeMetric(map[string]string{"other_label": "value"}, 5.0, 3000),
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
					makeMetric(map[string]string{"running_lora_adapters": "lora1", "max_lora": "2"}, 5.0, 3000),
					makeMetric(map[string]string{"running_lora_adapters": "lora2,lora3", "max_lora": "4"}, 6.0, 1000),
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
			loraMetric, err := p.getLatestLoraMetric(tc.metricFamilies)

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
	tests := []struct {
		name            string
		metricFamilies  map[string]*dto.MetricFamily
		mapping         *MetricMapping
		existingMetrics *Metrics
		expectedMetrics *Metrics
		expectedErr     error // Count of expected errors
	}{
		{
			name: "vllm metrics",
			metricFamilies: map[string]*dto.MetricFamily{
				"vllm_waiting": makeMetricFamily("vllm_waiting",
					makeMetric(nil, 5.0, 1000),
					makeMetric(nil, 7.0, 2000), // Newer
				),
				"vllm_usage": makeMetricFamily("vllm_usage",
					makeMetric(nil, 0.8, 2000),
					makeMetric(nil, 0.7, 500),
				),
				"vllm:lora_requests_info": makeMetricFamily("vllm:lora_requests_info",
					makeMetric(map[string]string{"running_lora_adapters": "lora1,lora2", "waiting_lora_adapters": "lora3", "max_lora": "3"}, 3000.0, 1000),
				),
			},
			mapping: &MetricMapping{
				TotalQueuedRequests: &MetricSpec{MetricName: "vllm_waiting"},
				KVCacheUtilization:  &MetricSpec{MetricName: "vllm_usage"},
				LoraRequestInfo:     &MetricSpec{MetricName: "vllm:lora_requests_info"},
			},
			existingMetrics: &Metrics{},
			expectedMetrics: &Metrics{
				WaitingQueueSize:    7,
				KVCacheUsagePercent: 0.8,
				ActiveModels:        map[string]int{"lora1": 0, "lora2": 0},
				WaitingModels:       map[string]int{"lora3": 0},
				MaxActiveModels:     3,
			},
		},
		{
			name:           "missing metrics",
			metricFamilies: map[string]*dto.MetricFamily{}, // No metrics
			mapping: &MetricMapping{
				TotalQueuedRequests: &MetricSpec{MetricName: "vllm_waiting"},
				KVCacheUtilization:  &MetricSpec{MetricName: "vllm_usage"},
				LoraRequestInfo:     &MetricSpec{MetricName: "vllm:lora_requests_info"},
			},
			existingMetrics: &Metrics{ActiveModels: map[string]int{}, WaitingModels: map[string]int{}},
			expectedMetrics: &Metrics{ActiveModels: map[string]int{}, WaitingModels: map[string]int{}},
			expectedErr:     multierr.Combine(errors.New("metric family \"vllm_waiting\" not found"), errors.New("metric family \"vllm_usage\" not found"), errors.New("metric family \"vllm:lora_requests_info\" not found")),
		},
		{
			name: "partial metrics available + LoRA",
			metricFamilies: map[string]*dto.MetricFamily{
				"vllm_usage": makeMetricFamily("vllm_usage",
					makeMetric(nil, 0.8, 2000), // Only usage is present
				),
				"vllm:lora_requests_info": makeMetricFamily("vllm:lora_requests_info",
					makeMetric(map[string]string{"running_lora_adapters": "lora1,lora2", "waiting_lora_adapters": "lora3", "max_lora": "3"}, 3000.0, 1000),
				),
			},
			mapping: &MetricMapping{
				TotalQueuedRequests: &MetricSpec{MetricName: "vllm_waiting"}, // Not Present
				KVCacheUtilization:  &MetricSpec{MetricName: "vllm_usage"},
				LoraRequestInfo:     &MetricSpec{MetricName: "vllm:lora_requests_info"},
			},
			existingMetrics: &Metrics{},
			expectedMetrics: &Metrics{
				WaitingQueueSize:    0,
				KVCacheUsagePercent: 0.8,
				ActiveModels:        map[string]int{"lora1": 0, "lora2": 0},
				WaitingModels:       map[string]int{"lora3": 0},
				MaxActiveModels:     3,
			},
			expectedErr: errors.New("metric family \"vllm_waiting\" not found"),
		},
		{
			name: "invalid max lora",
			metricFamilies: map[string]*dto.MetricFamily{
				"vllm:lora_requests_info": makeMetricFamily("vllm:lora_requests_info",
					makeMetric(map[string]string{"running_lora_adapters": "lora1", "max_lora": "invalid"}, 3000.0, 1000),
				),
			},
			mapping: &MetricMapping{
				LoraRequestInfo: &MetricSpec{MetricName: "vllm:lora_requests_info"},
			},
			existingMetrics: &Metrics{},
			expectedMetrics: &Metrics{
				ActiveModels:    map[string]int{"lora1": 0},
				WaitingModels:   map[string]int{},
				MaxActiveModels: 0, // Should still default to 0.

			},
			expectedErr: errors.New("strconv.Atoi: parsing \"invalid\": invalid syntax"),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := &PodMetricsClientImpl{MetricMapping: tc.mapping}
			updated, err := p.promToPodMetrics(tc.metricFamilies, tc.existingMetrics)
			if tc.expectedErr != nil {
				assert.Error(t, err)
				assert.EqualError(t, err, tc.expectedErr.Error())
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expectedMetrics, updated)
			}
		})
	}
}

// TestFetchMetrics is a basic integration test. It assumes
// there's no server running on the specified port.
func TestFetchMetrics(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	pod := &Pod{
		Address: "127.0.0.1",
		NamespacedName: types.NamespacedName{
			Namespace: "test",
			Name:      "pod",
		},
	}
	existing := &Metrics{}
	p := &PodMetricsClientImpl{} // No MetricMapping needed for this basic test

	_, err := p.FetchMetrics(ctx, pod, existing, 9999) // Use a port that's unlikely to be in use.
	if err == nil {
		t.Errorf("FetchMetrics() expected error, got nil")
	}
	// Check for a specific error message (fragile, but OK for this example)
	expectedSubstr := "connection refused"
	if err != nil && !strings.Contains(err.Error(), expectedSubstr) {
		t.Errorf("FetchMetrics() error = %v, want error containing %q", err, expectedSubstr)
	}
}
