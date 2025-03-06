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
	"reflect"
	"strings"
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

func TestNewMetricMappingAndValidate(t *testing.T) {
	tests := []struct {
		name           string
		allStr         string
		waitingStr     string
		runningStr     string
		usedStr        string
		maxStr         string
		usageStr       string
		loraReqInfoStr string
		wantErr        bool
		expectedErr    string // Added to check for specific error messages
	}{
		{
			name:           "valid vllm mapping",
			runningStr:     "running_metric",
			waitingStr:     "waiting_metric",
			usageStr:       "usage_metric",
			loraReqInfoStr: "lora_requests_info",
			wantErr:        false,
			expectedErr:    "",
		},
		{
			name:       "valid triton mapping",
			runningStr: "running_metric{label1=value1}",
			allStr:     "all_metric{label2=value2}",
			usedStr:    "used_blocks{label3=value3}",
			maxStr:     "max_blocks{label4=value4}",
			wantErr:    false,
		},
		{
			name:       "multiple labels mapping",
			runningStr: "running_metric{label1=value1,label5=value5}",
			allStr:     "all_metric{label2=value2,label6=value6}",
			usedStr:    "used_blocks{label3=value3}",
			maxStr:     "max_blocks{label4=value4}",
			wantErr:    false,
		},
		{
			name:        "missing running",
			waitingStr:  "waiting_metric",
			usageStr:    "usage_metric",
			wantErr:     true,
			expectedErr: "RunningRequests is required",
		},
		{
			name:        "missing both waiting and all",
			runningStr:  "running_metric",
			usageStr:    "usage_metric",
			wantErr:     true,
			expectedErr: "either WaitingRequests or AllRequests must be specified",
		},
		{
			name:        "missing usage and both block metrics",
			runningStr:  "running_metric",
			waitingStr:  "waiting_metric",
			wantErr:     true,
			expectedErr: "either KVCacheUsage or both UsedKVCacheBlocks and MaxKVCacheBlocks must be specified",
		},
		{
			name:        "missing max block metric",
			runningStr:  "running_metric",
			waitingStr:  "waiting_metric",
			usedStr:     "used_blocks",
			wantErr:     true,
			expectedErr: "either KVCacheUsage or both UsedKVCacheBlocks and MaxKVCacheBlocks must be specified",
		},
		{
			name:        "missing used block metric",
			runningStr:  "running_metric",
			waitingStr:  "waiting_metric",
			maxStr:      "max_blocks",
			wantErr:     true,
			expectedErr: "either KVCacheUsage or both UsedKVCacheBlocks and MaxKVCacheBlocks must be specified",
		},
		{
			name:        "invalid running metric format",
			runningStr:  "running_metric{invalid",
			waitingStr:  "waiting_metric",
			usageStr:    "usage_metric",
			wantErr:     true,
			expectedErr: "error parsing RunningRequests", // Check for part of the expected error
		},
		{
			name:           "lora metrics present",
			runningStr:     "running_metric",
			waitingStr:     "waiting_metric",
			usageStr:       "usage_metric",
			loraReqInfoStr: "lora_requests_info",

			wantErr:     false,
			expectedErr: "", // Check for part of the expected error
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewMetricMapping(tt.allStr, tt.waitingStr, tt.runningStr, tt.usedStr, tt.maxStr, tt.usageStr, tt.loraReqInfoStr)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewMetricMapping() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr && !strings.Contains(err.Error(), tt.expectedErr) {
				t.Errorf("NewMetricMapping() error = %v, expected to contain = %v", err, tt.expectedErr)
			}
		})
	}
}
