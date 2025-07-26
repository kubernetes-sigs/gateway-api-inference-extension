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

package filter

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

func TestFilter(t *testing.T) {
	tests := []struct {
		name   string
		req    *types.LLMRequest
		input  []types.Pod
		output []types.Pod
	}{
		{
			name: "TestHeaderBasedFilter, header endpoint unset in request",
			req:  &types.LLMRequest{}, // Delieverately unset the header.
			input: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{
						Address: "test-endpoint",
					},
				},
			},
			output: []types.Pod{},
		},
		{
			name: "TestHeaderBasedFilter, header endpoint set in request but no match",
			req:  &types.LLMRequest{Headers: map[string]string{HeaderTestEppEndPointSelectionKey: "test-endpoint"}},
			input: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{
						Address: "test-endpoint-unmatch",
					},
				},
			},
			output: []types.Pod{},
		},
		{
			name: "TestHeaderBasedFilter, header endpoint set",
			req:  &types.LLMRequest{Headers: map[string]string{HeaderTestEppEndPointSelectionKey: "test-endpoint"}},
			input: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{
						Address: "test-endpoint",
					},
				},
			},
			output: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{
						Address: "test-endpoint",
					},
				},
			},
		},
		{
			name: "TestHeaderBasedFilter, multiple header endpoints set and multiple matches",
			req:  &types.LLMRequest{Headers: map[string]string{HeaderTestEppEndPointSelectionKey: "test-endpoint3,test-endpoint2"}},
			input: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{
						Address: "test-endpoint1",
					},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{
						Address: "test-endpoint2",
					},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{
						Address: "test-endpoint3",
					},
				},
			},
			output: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{
						Address: "test-endpoint3",
					},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{
						Address: "test-endpoint2",
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := NewHeaderBasedTestingFilter().Filter(context.Background(), types.NewCycleState(), test.req, test.input)

			if diff := cmp.Diff(test.output, got); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}
