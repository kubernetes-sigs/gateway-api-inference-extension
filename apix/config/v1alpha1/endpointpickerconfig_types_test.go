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

package v1alpha1

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestStringers(t *testing.T) {
	tests := []struct {
		name string
		obj  fmt.Stringer
		want string
	}{
		{
			name: "PluginSpec",
			obj: PluginSpec{
				Name:       "test-plugin",
				Type:       "test-type",
				Parameters: json.RawMessage(`{"key":"value"}`),
			},
			want: "{Name: test-plugin, Type: test-type, Parameters: {\"key\":\"value\"}}",
		},
		{
			name: "SchedulingPlugin",
			obj: SchedulingPlugin{
				PluginRef: "test-ref",
				Weight:    ptr.To(2.5),
			},
			want: "{PluginRef: test-ref, Weight: 2.50}",
		},
		{
			name: "SaturationDetectorConfig",
			obj: &SaturationDetectorConfig{
				PluginRef: "test-plugin",
			},
			want: "{PluginRef: test-plugin}",
		},
		{
			name: "FlowControlConfig",
			obj: &FlowControlConfig{
				MaxBytes:          resource.NewQuantity(1024, resource.DecimalSI),
				DefaultRequestTTL: &metav1.Duration{Duration: 30 * time.Second},
				PriorityBands: []PriorityBandConfig{
					{Priority: 10, MaxBytes: resource.NewQuantity(512, resource.DecimalSI)},
				},
			},
			want: "{MaxBytes: 1024, DefaultRequestTTL: 30s, PriorityBands: [{Priority: 10, MaxBytes: 512}]}",
		},
		{
			name: "ParserConfig",
			obj: &ParserConfig{
				PluginRef: "test-parser",
			},
			want: "{PluginRef: test-parser}",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.obj.String()
			if got != tc.want {
				t.Errorf("String() = %v, want %v", got, tc.want)
			}
		})
	}
}
