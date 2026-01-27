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

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/test"
)

func TestFilter(t *testing.T) {
	tests := []struct {
		name   string
		req    *fwksched.LLMRequest
		input  []fwksched.Endpoint
		output []fwksched.Endpoint
	}{
		{
			name: "header unset in request",
			req:  &fwksched.LLMRequest{}, // Deliberately unset
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.1", Port: "3000"}, nil, nil),
			},
			output: []fwksched.Endpoint{},
		},
		{
			name: "header set but no IP match",
			req:  &fwksched.LLMRequest{Headers: map[string]string{test.HeaderTestEppEndPointSelectionKey: "10.0.0.99"}},
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.1", Port: "3000"}, nil, nil),
			},
			output: []fwksched.Endpoint{},
		},
		{
			name: "IP-only header matches pod (port-agnostic)",
			req:  &fwksched.LLMRequest{Headers: map[string]string{test.HeaderTestEppEndPointSelectionKey: "10.0.0.1"}},
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.1", Port: "3002"}, nil, nil),
			},
			output: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.1", Port: "3002"}, nil, nil),
			},
		},
		{
			name: "IP:port header matches exact port",
			req:  &fwksched.LLMRequest{Headers: map[string]string{test.HeaderTestEppEndPointSelectionKey: "10.0.0.1:3002"}},
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.1", Port: "3000"}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.1", Port: "3002"}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.2", Port: "3002"}, nil, nil),
			},
			output: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.1", Port: "3002"}, nil, nil),
			},
		},
		{
			name: "IP:port header with non-matching port produces no match",
			req:  &fwksched.LLMRequest{Headers: map[string]string{test.HeaderTestEppEndPointSelectionKey: "10.0.0.1:9999"}},
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.1", Port: "3002"}, nil, nil),
			},
			output: []fwksched.Endpoint{},
		},
		{
			name: "multiple header values (IP and IP:port) produce multiple matches in order and deduped",
			req:  &fwksched.LLMRequest{Headers: map[string]string{test.HeaderTestEppEndPointSelectionKey: "10.0.0.3:3004, 10.0.0.2, 10.0.0.3"}},
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.1", Port: "3000"}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.2", Port: "3002"}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.3", Port: "3004"}, nil, nil),
			},
			output: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.3", Port: "3004"}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.2", Port: "3002"}, nil, nil),
			},
		},
		{
			name: "IPv6 with brackets and port",
			req:  &fwksched.LLMRequest{Headers: map[string]string{test.HeaderTestEppEndPointSelectionKey: "[fd00::1]:3002"}},
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "fd00::1", Port: "3002"}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "fd00::2", Port: "3002"}, nil, nil),
			},
			output: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "fd00::1", Port: "3002"}, nil, nil),
			},
		},
		{
			name: "IPv6 bare address (no port)",
			req:  &fwksched.LLMRequest{Headers: map[string]string{test.HeaderTestEppEndPointSelectionKey: "fd00::2"}},
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "fd00::1", Port: "3002"}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "fd00::2", Port: "3004"}, nil, nil),
			},
			output: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: "fd00::2", Port: "3004"}, nil, nil),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := NewHeaderBasedTestingFilter().Filter(context.Background(), fwksched.NewCycleState(), tc.req, tc.input)
			if diff := cmp.Diff(tc.output, got, cmp.Comparer(fwksched.EndpointComparer)); diff != "" {
				t.Fatalf("Unexpected output (-want +got): %s", diff)
			}
		})
	}
}
