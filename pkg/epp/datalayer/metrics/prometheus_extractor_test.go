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
	"testing"

	dto "github.com/prometheus/client_model/go"
	"google.golang.org/protobuf/proto"
	"k8s.io/utils/ptr"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

func TestPrometheusMetricPluginExtract(t *testing.T) {
	ctx := context.Background()
	metricName := "custom_metric"
	labels := map[string]string{"label1": "value1"}

	plugin := NewPrometheusMetricPlugin(metricName, labels)

	tests := []struct {
		name       string
		data       any
		expectVal  float64
		wantErr    bool
		wantUpdate bool
	}{
		{
			name:       "nil data",
			data:       nil,
			wantErr:    true,
			wantUpdate: false,
		},
		{
			name:       "missing metric",
			data:       fwkdl.PrometheusMetricMap{},
			wantErr:    true,
			wantUpdate: false,
		},
		{
			name: "valid metric with matching labels",
			data: fwkdl.PrometheusMetricMap{
				metricName: &dto.MetricFamily{
					Type: dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{
						{
							Label: []*dto.LabelPair{
								{Name: proto.String("label1"), Value: proto.String("value1")},
							},
							Gauge: &dto.Gauge{Value: ptr.To(123.45)},
						},
					},
				},
			},
			expectVal:  123.45,
			wantErr:    false,
			wantUpdate: true,
		},
		{
			name: "valid metric with extra labels (should match)",
			data: fwkdl.PrometheusMetricMap{
				metricName: &dto.MetricFamily{
					Type: dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{
						{
							Label: []*dto.LabelPair{
								{Name: proto.String("label1"), Value: proto.String("value1")},
								{Name: proto.String("extra"), Value: proto.String("foo")},
							},
							Gauge: &dto.Gauge{Value: ptr.To(67.89)},
						},
					},
				},
			},
			expectVal:  67.89,
			wantErr:    false,
			wantUpdate: true,
		},
		{
			name: "metric with mismatched labels",
			data: fwkdl.PrometheusMetricMap{
				metricName: &dto.MetricFamily{
					Type: dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{
						{
							Label: []*dto.LabelPair{
								{Name: proto.String("label1"), Value: proto.String("other")},
							},
							Gauge: &dto.Gauge{Value: ptr.To(0.0)},
						},
					},
				},
			},
			wantErr:    true,
			wantUpdate: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ep := fwkdl.NewEndpoint(nil, nil)
			err := plugin.Extract(ctx, tt.data, ep)
			if tt.wantErr && err == nil {
				t.Errorf("expected error but got nil")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if tt.wantUpdate {
				val, ok := ep.GetMetrics().Custom[metricName]
				if !ok {
					t.Errorf("custom metric not found in endpoint")
				} else if val != tt.expectVal {
					t.Errorf("expected value %v, got %v", tt.expectVal, val)
				}
			} else {
				val, ok := ep.GetMetrics().Custom[metricName]
				if ok {
					t.Errorf("expected no custom metric, but found %v", val)
				}
			}
		})
	}
}
