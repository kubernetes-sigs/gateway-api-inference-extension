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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/test"
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
			name: "header unset in request",
			req:  &types.LLMRequest{}, // Deliberately unset
			input: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{Address: "10.0.0.1", Port: "3000"}},
			},
			output: []types.Pod{},
		},
		{
			name: "header set but no IP match",
			req:  &types.LLMRequest{Headers: map[string]string{test.HeaderTestEppEndPointSelectionKey: "10.0.0.99"}},
			input: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{Address: "10.0.0.1", Port: "3000"}},
			},
			output: []types.Pod{},
		},
		{
			name: "IP-only header matches pod (port-agnostic)",
			req:  &types.LLMRequest{Headers: map[string]string{test.HeaderTestEppEndPointSelectionKey: "10.0.0.1"}},
			input: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{Address: "10.0.0.1", Port: "3002"}},
			},
			output: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{Address: "10.0.0.1", Port: "3002"}},
			},
		},
		{
			name: "IP:port header matches exact port",
			req:  &types.LLMRequest{Headers: map[string]string{test.HeaderTestEppEndPointSelectionKey: "10.0.0.1:3002"}},
			input: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{Address: "10.0.0.1", Port: "3000"}},
				&types.PodMetrics{Pod: &backend.Pod{Address: "10.0.0.1", Port: "3002"}},
				&types.PodMetrics{Pod: &backend.Pod{Address: "10.0.0.2", Port: "3002"}},
			},
			output: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{Address: "10.0.0.1", Port: "3002"}},
			},
		},
		{
			name: "IP:port header with non-matching port produces no match",
			req:  &types.LLMRequest{Headers: map[string]string{test.HeaderTestEppEndPointSelectionKey: "10.0.0.1:9999"}},
			input: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{Address: "10.0.0.1", Port: "3002"}},
			},
			output: []types.Pod{},
		},
		{
			name: "multiple header values (IP and IP:port) produce multiple matches in order and deduped",
			req:  &types.LLMRequest{Headers: map[string]string{test.HeaderTestEppEndPointSelectionKey: "10.0.0.3:3004, 10.0.0.2, 10.0.0.3"}},
			input: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{Address: "10.0.0.1", Port: "3000"}},
				&types.PodMetrics{Pod: &backend.Pod{Address: "10.0.0.2", Port: "3002"}},
				&types.PodMetrics{Pod: &backend.Pod{Address: "10.0.0.3", Port: "3004"}},
			},
			output: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{Address: "10.0.0.3", Port: "3004"}},
				&types.PodMetrics{Pod: &backend.Pod{Address: "10.0.0.2", Port: "3002"}},
			},
		},
		{
			name: "IPv6 with brackets and port",
			req:  &types.LLMRequest{Headers: map[string]string{test.HeaderTestEppEndPointSelectionKey: "[fd00::1]:3002"}},
			input: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{Address: "fd00::1", Port: "3002"}},
				&types.PodMetrics{Pod: &backend.Pod{Address: "fd00::2", Port: "3002"}},
			},
			output: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{Address: "fd00::1", Port: "3002"}},
			},
		},
		{
			name: "IPv6 bare address (no port)",
			req:  &types.LLMRequest{Headers: map[string]string{test.HeaderTestEppEndPointSelectionKey: "fd00::2"}},
			input: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{Address: "fd00::1", Port: "3002"}},
				&types.PodMetrics{Pod: &backend.Pod{Address: "fd00::2", Port: "3004"}},
			},
			output: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{Address: "fd00::2", Port: "3004"}},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := NewHeaderBasedTestingFilter().Filter(context.Background(), types.NewCycleState(), tc.req, tc.input)
			if diff := cmp.Diff(tc.output, got); diff != "" {
				t.Fatalf("Unexpected output (-want +got): %s", diff)
			}
		})
	}
}
