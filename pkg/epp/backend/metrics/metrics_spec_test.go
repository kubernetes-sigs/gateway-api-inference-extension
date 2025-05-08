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
	"reflect"
	"testing"
)

func TestStringToMetricSpec(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    *MetricSpec
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
			want: &MetricSpec{
				MetricName: "my_metric",
				Labels:     map[string]string{},
			},
			wantErr: false,
		},
		{
			name:  "one label",
			input: "my_metric{label1=value1}",
			want: &MetricSpec{
				MetricName: "my_metric",
				Labels: map[string]string{
					"label1": "value1",
				},
			},
			wantErr: false,
		},
		{
			name:  "multiple labels",
			input: "my_metric{label1=value1,label2=value2}",
			want: &MetricSpec{
				MetricName: "my_metric",
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
			want: &MetricSpec{
				MetricName: "my_metric",
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
			want:    nil, // Corrected expected value
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
			want: &MetricSpec{
				MetricName: "my_metric",
				Labels:     map[string]string{},
			},
			wantErr: false,
		},
		{
			name:  "no labels and just metric name with space before and after",
			input: "  my_metric  ",
			want: &MetricSpec{
				MetricName: "my_metric",
				Labels:     map[string]string{},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := stringToMetricSpec(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("stringToMetricSpec() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.want != nil && got != nil { // compare maps directly
				if tt.want.Labels == nil {
					tt.want.Labels = make(map[string]string)
				}
				if !reflect.DeepEqual(got.MetricName, tt.want.MetricName) {
					t.Errorf("stringToMetricSpec() got MetricName = %v, want %v", got.MetricName, tt.want.MetricName)
				}
				if !reflect.DeepEqual(got.Labels, tt.want.Labels) {
					t.Errorf("stringToMetricSpec() got Labels = %v, want %v", got.Labels, tt.want.Labels)
				}
			} else if tt.want != got { // handles if one is nil and the other isn't
				t.Errorf("stringToMetricSpec() = %v, want %v", got, tt.want)
			}
		})
	}
}
