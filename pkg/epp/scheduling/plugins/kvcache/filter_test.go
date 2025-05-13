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

package kvcache

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

func TestFilter(t *testing.T) {
	tests := []struct {
		name   string
		req    *types.LLMRequest
		filter plugins.Filter
		input  []types.Pod
		output []types.Pod
	}{

		{
			name:   "least kv cache empty input",
			filter: NewLeastKVCacheFilter(),
			input:  []types.Pod{},
			output: []types.Pod{},
		},
		{
			name:   "least kv cache",
			filter: NewLeastKVCacheFilter(),
			input: []types.Pod{
				&types.PodMetrics{
					MetricsState: &backendmetrics.MetricsState{
						KVCacheUsagePercent: 0,
					},
				},
				&types.PodMetrics{
					MetricsState: &backendmetrics.MetricsState{
						KVCacheUsagePercent: 0.3,
					},
				},
				&types.PodMetrics{
					MetricsState: &backendmetrics.MetricsState{
						KVCacheUsagePercent: 1.0,
					},
				},
			},
			output: []types.Pod{
				&types.PodMetrics{
					MetricsState: &backendmetrics.MetricsState{
						KVCacheUsagePercent: 0,
					},
				},
				&types.PodMetrics{
					MetricsState: &backendmetrics.MetricsState{
						KVCacheUsagePercent: 0.3,
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := types.NewSchedulingContext(context.Background(), test.req, nil, test.input)
			got := test.filter.Filter(ctx, test.input)

			if diff := cmp.Diff(test.output, got); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}
