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
	"errors"
	"strconv"
	"strings"
	"testing"

	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
	"k8s.io/utils/ptr"
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

// TestStringToMetricSpec determines parsing of metric's specification.
func TestStringToMetricSpec(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    *Spec
		wantErr bool
	}{
		{
			name:    "empty string",
			input:   "",
			want:    nil,
			wantErr: false,
		},
		{
			name:  "no labels",
			input: "my_metric",
			want: &Spec{
				Name:   "my_metric",
				Labels: map[string]string{},
			},
			wantErr: false,
		},
		{
			name:  "one label",
			input: "my_metric{label1=value1}",
			want: &Spec{
				Name: "my_metric",
				Labels: map[string]string{
					"label1": "value1",
				},
			},
			wantErr: false,
		},
		{
			name:  "label with quoted value",
			input: "my_metric{label1=\"value1\"}",
			want: &Spec{
				Name: "my_metric",
				Labels: map[string]string{
					"label1": "value1",
				},
			},
			wantErr: false,
		},
		{
			name:  "multiple labels",
			input: "my_metric{label1=value1,label2=value2}",
			want: &Spec{
				Name: "my_metric",
				Labels: map[string]string{
					"label1": "value1",
					"label2": "value2",
				},
			},
			wantErr: false,
		},
		{
			name:  "extra whitespace",
			input: "  my_metric  {  label1  =  value1  ,  label2  =  value2  }  ",
			want: &Spec{
				Name: "my_metric",
				Labels: map[string]string{
					"label1": "value1",
					"label2": "value2",
				},
			},
			wantErr: false,
		},
		{
			name:    "missing closing brace",
			input:   "my_metric{label1=value1",
			want:    nil,
			wantErr: true,
		},
		{
			name:    "missing opening brace",
			input:   "my_metriclabel1=value1}",
			want:    nil,
			wantErr: true,
		},
		{
			name:    "invalid label pair",
			input:   "my_metric{label1}",
			want:    nil,
			wantErr: true,
		},
		{
			name:    "empty label name",
			input:   "my_metric{=value1}",
			want:    nil,
			wantErr: true,
		},
		{
			name:    "empty label value",
			input:   "my_metric{label1=}",
			want:    nil,
			wantErr: true,
		},
		{
			name:    "empty label name and value with spaces",
			input:   "my_metric{  =  }",
			want:    nil,
			wantErr: true,
		},
		{
			name:    "characters after closing brace",
			input:   "my_metric{label=val}extra",
			want:    nil,
			wantErr: true,
		},
		{
			name:    "empty metric name",
			input:   "{label=val}",
			want:    nil,
			wantErr: true,
		},
		{
			name:  "no labels and just metric name with space",
			input: "my_metric ",
			want: &Spec{
				Name:   "my_metric",
				Labels: map[string]string{},
			},
			wantErr: false,
		},
		{
			name:  "no labels and just metric name with space before and after",
			input: "  my_metric  ",
			want: &Spec{
				Name:   "my_metric",
				Labels: map[string]string{},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseStringToSpec(tt.input)

			if tt.wantErr {
				assert.Error(t, err, "expected error but got none")
			} else {
				assert.NoError(t, err, "unexpected error from parseStringToSpec")
				if tt.want != nil && got != nil {
					assert.Equal(t, tt.want, got)
				} else {
					assert.Equal(t, tt.want, got, "expected and actual values don't match")
				}
			}
		})
	}
}

// TestGetMetric checks retrieving of standard metrics from a metric families.
func TestGetMetric(t *testing.T) {
	metricFamilies := PrometheusMetricMap{
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
		name      string
		spec      Spec
		expected  float64
		wantError bool
	}{
		{
			name: "get labeled metric, exists",
			spec: Spec{
				Name:   "metric1",
				Labels: map[string]string{"label1": "value1"},
			},
			expected:  1.0,
			wantError: false,
		},
		{
			name: "get labeled metric, wrong value",
			spec: Spec{
				Name:   "metric1",
				Labels: map[string]string{"label1": "value3"},
			},
			expected:  -1, // Expect an error, not a specific value
			wantError: true,
		},
		{
			name: "get labeled metric, missing label",
			spec: Spec{
				Name:   "metric1",
				Labels: map[string]string{"label2": "value2"},
			},
			expected:  -1,
			wantError: true,
		},
		{
			name: "get labeled metric, extra label present",
			spec: Spec{
				Name:   "metric2",
				Labels: map[string]string{"labelA": "A1"},
			},
			expected:  3.0,
			wantError: false,
		},
		{
			name: "get unlabeled metric, exists",
			spec: Spec{
				Name:   "metric3",
				Labels: nil, // Explicitly nil
			},
			expected:  5.0, // latest metric, which occurs first in our test data
			wantError: false,
		},
		{
			name: "get unlabeled metric, metric family not found",
			spec: Spec{
				Name:   "metric4",
				Labels: nil,
			},
			expected:  -1,
			wantError: true,
		},
		{
			name: "get labeled metric, metric family not found",
			spec: Spec{
				Name:   "metric4",
				Labels: map[string]string{"label1": "value1"},
			},
			expected:  -1,
			wantError: true,
		},
		{
			name: "get metric, no metrics available",
			spec: Spec{
				Name: "empty_metric",
			},
			expected:  -1,
			wantError: true,
		},
		{
			name: "get latest metric",
			spec: Spec{
				Name:   "metric3",
				Labels: map[string]string{}, // Empty map, not nil
			},
			expected:  5.0,
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metric, err := tt.spec.getLatestMetric(metricFamilies)
			if tt.wantError {
				assert.Error(t, err, "expected error but got nil")
			} else {
				require.NoError(t, err, "unexpected error from getLatestMetric")
				value := extractValue(metric)
				assert.Equal(t, tt.expected, value, "metric value mismatch")
			}
		})
	}
}

func TestGetLoRAMetric(t *testing.T) {
	loraSpec := &LoRASpec{
		Spec: &Spec{
			Name: "vllm:lora_requests_info",
		},
	}

	testCases := []struct {
		name             string
		metricFamilies   PrometheusMetricMap
		expectedAdapters map[string]int
		expectedMax      int
		expectedErr      error
		spec             *LoRASpec
	}{
		{
			name: "no lora metrics",
			metricFamilies: PrometheusMetricMap{
				"some_other_metric": makeMetricFamily("some_other_metric",
					makeMetric(nil, 1.0, 1000),
				),
			},
			expectedAdapters: map[string]int{},
			expectedMax:      0,
			expectedErr:      errors.New("metric family \"vllm:lora_requests_info\" not found"), // Expect an error because the family is missing
			spec:             loraSpec,
		},
		{
			name: "basic lora metrics",
			metricFamilies: PrometheusMetricMap{
				"vllm:lora_requests_info": makeMetricFamily("vllm:lora_requests_info",
					makeMetric(map[string]string{"running_lora_adapters": "lora1", "max_lora": "2"}, 3000.0, 1000),       // Newer
					makeMetric(map[string]string{"running_lora_adapters": "lora2,lora3", "max_lora": "4"}, 1000.0, 1000), // Older

				),
			},
			expectedAdapters: map[string]int{"lora1": 0},
			expectedMax:      2,
			expectedErr:      nil,
			spec:             loraSpec,
		},
		{
			name: "no matching lora metrics",
			metricFamilies: PrometheusMetricMap{
				"vllm:lora_requests_info": makeMetricFamily("vllm:lora_requests_info",
					makeMetric(map[string]string{"other_label": "value"}, 5.0, 3000),
				),
			},
			expectedAdapters: map[string]int{},
			expectedMax:      0,
			expectedErr:      nil, // Expect *no* error; just no adapters found
			spec:             loraSpec,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			metric, err := tc.spec.getLatestMetric(tc.metricFamilies)

			if tc.expectedErr != nil {
				assert.Error(t, err)
				assert.EqualError(t, err, tc.expectedErr.Error())
				return
			}

			require.NoError(t, err)

			if tc.spec == nil {
				assert.Nil(t, metric)
				return
			}

			if tc.expectedAdapters == nil && metric == nil {
				return
			}

			require.NotNil(t, metric, "expected non-nil metric")

			adaptersFound := make(map[string]int)
			maxLora := 0

			for _, label := range metric.GetLabel() {
				switch label.GetName() {
				case "running_lora_adapters", "waiting_lora_adapters":
					if label.GetValue() != "" {
						for _, adapter := range strings.Split(label.GetValue(), ",") {
							adaptersFound[adapter] = 0
						}
					}
				case "max_lora":
					parsed, parseErr := strconv.Atoi(label.GetValue())
					require.NoError(t, parseErr, "failed to parse max_lora")
					maxLora = parsed
				}
			}

			assert.Equal(t, tc.expectedAdapters, adaptersFound, "adapters mismatch")
			assert.Equal(t, tc.expectedMax, maxLora, "maxLora mismatch")
		})
	}
}

func TestLabelsMatch(t *testing.T) {
	tests := []struct {
		name         string
		metricLabels []*dto.LabelPair
		spec         *Spec
		want         bool
	}{
		{
			name:         "empty spec labels, should match",
			metricLabels: []*dto.LabelPair{{Name: proto.String("a"), Value: proto.String("b")}},
			spec:         &Spec{Labels: map[string]string{}},
			want:         true,
		},
		{
			name:         "nil spec labels, should match",
			metricLabels: []*dto.LabelPair{{Name: proto.String("a"), Value: proto.String("b")}},
			spec:         &Spec{},
			want:         true,
		},
		{
			name:         "exact match",
			metricLabels: []*dto.LabelPair{{Name: proto.String("a"), Value: proto.String("b")}},
			spec:         &Spec{Labels: map[string]string{"a": "b"}},
			want:         true,
		},
		{
			name:         "extra labels in metric",
			metricLabels: []*dto.LabelPair{{Name: proto.String("a"), Value: proto.String("b")}, {Name: proto.String("c"), Value: proto.String("d")}},
			spec:         &Spec{Labels: map[string]string{"a": "b"}},
			want:         true,
		},
		{
			name:         "missing label in metric",
			metricLabels: []*dto.LabelPair{{Name: proto.String("a"), Value: proto.String("b")}},
			spec:         &Spec{Labels: map[string]string{"a": "b", "c": "d"}},
			want:         false,
		},
		{
			name:         "value mismatch",
			metricLabels: []*dto.LabelPair{{Name: proto.String("a"), Value: proto.String("b")}},
			spec:         &Spec{Labels: map[string]string{"a": "c"}},
			want:         false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.spec.labelsMatch(tt.metricLabels)
			assert.Equal(t, tt.want, got, "labelsMatch() mismatch")
		})
	}
}

func TestExtractValue(t *testing.T) {
	tests := []struct {
		name   string
		metric *dto.Metric
		want   float64
	}{
		{
			name:   "nil metric",
			metric: nil,
			want:   0.0,
		},
		{
			name: "unsupported metric type",
			metric: &dto.Metric{
				Summary: &dto.Summary{},
			},
			want: 0.0,
		},
		{
			name: "gauge metric",
			metric: &dto.Metric{
				Gauge: &dto.Gauge{Value: ptr.To(42.5)},
			},
			want: 42.5,
		},
		{
			name: "counter metric",
			metric: &dto.Metric{
				Counter: &dto.Counter{Value: ptr.To(99.9)},
			},
			want: 99.9,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("extractValue panicked for test %q: %v", tt.name, r)
				}
			}()
			got := extractValue(tt.metric)
			assert.Equal(t, tt.want, got)
		})
	}
}
