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
			name: "SaturationDetector",
			obj: &SaturationDetector{
				QueueDepthThreshold:       10,
				KVCacheUtilThreshold:      0.8,
				MetricsStalenessThreshold: metav1.Duration{Duration: 100 * time.Millisecond},
			},
			want: "{QueueDepthThreshold: 10, KVCacheUtilThreshold: 0.80, MetricsStalenessThreshold: 100ms}",
		},
		{
			name: "FlowControlConfig",
			obj: &FlowControlConfig{
				MaxBytes:          ptr.To(int64(1024)),
				DefaultRequestTTL: &metav1.Duration{Duration: 30 * time.Second},
				PriorityBands: []PriorityBandConfig{
					{Priority: 10, MaxBytes: ptr.To(int64(512))},
				},
			},
			want: "{MaxBytes: 1024, DefaultRequestTTL: 30s, PriorityBands: [{Priority: 10, MaxBytes: 512}]}",
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
