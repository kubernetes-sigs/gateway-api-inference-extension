/*
Copyright 2026 The Kubernetes Authors.

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

package bbr

import (
	"context"
	"testing"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/testing/protocmp"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	envoytest "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/test"
	epp "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

// fakeRequestPlugin implements framework.RequestProcessor for integration testing.
type fakeRequestPlugin struct {
	name     string
	mutateFn func(ctx context.Context, request *framework.InferenceRequest) error
}

func (p *fakeRequestPlugin) TypedName() epp.TypedName {
	return epp.TypedName{Type: "fake", Name: p.name}
}

func (p *fakeRequestPlugin) ProcessRequest(ctx context.Context, request *framework.InferenceRequest) error {
	return p.mutateFn(ctx, request)
}

var _ framework.RequestProcessor = &fakeRequestPlugin{}

// TestRequestPluginHeaderMutations_Unary validates header set/remove interactions
// across multiple request plugins in non-streaming (unary) mode via a real gRPC server.
func TestRequestPluginHeaderMutations_Unary(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		plugins        []framework.RequestProcessor
		reqHeaders     map[string]string
		wantSetHeaders map[string]string
		wantRemoved    []string
	}{
		{
			name: "set then remove same header - cancels out",
			plugins: []framework.RequestProcessor{
				&fakeRequestPlugin{
					name: "setter",
					mutateFn: func(_ context.Context, req *framework.InferenceRequest) error {
						req.SetHeader("X-Custom", "value1")
						return nil
					},
				},
				&fakeRequestPlugin{
					name: "remover",
					mutateFn: func(_ context.Context, req *framework.InferenceRequest) error {
						req.RemoveHeader("X-Custom")
						return nil
					},
				},
			},
			wantSetHeaders: map[string]string{},
			wantRemoved:    []string{"X-Custom"},
		},
		{
			name: "set then remove different headers - both apply",
			plugins: []framework.RequestProcessor{
				&fakeRequestPlugin{
					name: "setter",
					mutateFn: func(_ context.Context, req *framework.InferenceRequest) error {
						req.SetHeader("X-New", "hello")
						return nil
					},
				},
				&fakeRequestPlugin{
					name: "remover",
					mutateFn: func(_ context.Context, req *framework.InferenceRequest) error {
						req.RemoveHeader("X-Existing")
						return nil
					},
				},
			},
			reqHeaders: map[string]string{
				"X-Existing": "old-value",
			},
			wantSetHeaders: map[string]string{
				"X-New": "hello",
			},
			wantRemoved: []string{"X-Existing"},
		},
		{
			name: "remove non-existing header - no-op",
			plugins: []framework.RequestProcessor{
				&fakeRequestPlugin{
					name: "remover",
					mutateFn: func(_ context.Context, req *framework.InferenceRequest) error {
						req.RemoveHeader("X-Ghost")
						return nil
					},
				},
			},
			wantSetHeaders: map[string]string{},
			wantRemoved:    nil,
		},
		{
			name: "remove then set same header - new value wins",
			plugins: []framework.RequestProcessor{
				&fakeRequestPlugin{
					name: "remover",
					mutateFn: func(_ context.Context, req *framework.InferenceRequest) error {
						req.RemoveHeader("X-Reuse")
						return nil
					},
				},
				&fakeRequestPlugin{
					name: "setter",
					mutateFn: func(_ context.Context, req *framework.InferenceRequest) error {
						req.SetHeader("X-Reuse", "new-value")
						return nil
					},
				},
			},
			reqHeaders: map[string]string{
				"X-Reuse": "old-value",
			},
			wantSetHeaders: map[string]string{
				"X-Reuse": "new-value",
			},
			wantRemoved: nil,
		},
		{
			name: "two plugins set same header - last wins",
			plugins: []framework.RequestProcessor{
				&fakeRequestPlugin{
					name: "setter1",
					mutateFn: func(_ context.Context, req *framework.InferenceRequest) error {
						req.SetHeader("X-Shared", "first")
						return nil
					},
				},
				&fakeRequestPlugin{
					name: "setter2",
					mutateFn: func(_ context.Context, req *framework.InferenceRequest) error {
						req.SetHeader("X-Shared", "second")
						return nil
					},
				},
			},
			wantSetHeaders: map[string]string{
				"X-Shared": "second",
			},
			wantRemoved: nil,
		},
		{
			name: "two plugins set different headers - both apply",
			plugins: []framework.RequestProcessor{
				&fakeRequestPlugin{
					name: "setter-a",
					mutateFn: func(_ context.Context, req *framework.InferenceRequest) error {
						req.SetHeader("X-First", "aaa")
						return nil
					},
				},
				&fakeRequestPlugin{
					name: "setter-b",
					mutateFn: func(_ context.Context, req *framework.InferenceRequest) error {
						req.SetHeader("X-Second", "bbb")
						return nil
					},
				},
			},
			wantSetHeaders: map[string]string{
				"X-First":  "aaa",
				"X-Second": "bbb",
			},
			wantRemoved: nil,
		},
		{
			name: "two plugins remove same header - idempotent",
			plugins: []framework.RequestProcessor{
				&fakeRequestPlugin{
					name: "remover1",
					mutateFn: func(_ context.Context, req *framework.InferenceRequest) error {
						req.RemoveHeader("X-Dup")
						return nil
					},
				},
				&fakeRequestPlugin{
					name: "remover2",
					mutateFn: func(_ context.Context, req *framework.InferenceRequest) error {
						req.RemoveHeader("X-Dup")
						return nil
					},
				},
			},
			reqHeaders: map[string]string{
				"X-Dup": "value",
			},
			wantSetHeaders: map[string]string{},
			wantRemoved:    []string{"X-Dup"},
		},
		{
			name: "set existing header to same value - no mutation",
			plugins: []framework.RequestProcessor{
				&fakeRequestPlugin{
					name: "noop-setter",
					mutateFn: func(_ context.Context, req *framework.InferenceRequest) error {
						req.SetHeader("X-Keep", "original")
						return nil
					},
				},
			},
			reqHeaders: map[string]string{
				"X-Keep": "original",
			},
			wantSetHeaders: map[string]string{},
			wantRemoved:    nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx := context.Background()
			h := NewBBRHarnessWithPlugins(t, ctx, false, tc.plugins)

			// Build request: headers phase + body phase (unary style via ReqRaw)
			headers := map[string]string{}
			for k, v := range tc.reqHeaders {
				headers[k] = v
			}

			reqs := integration.ReqRaw(headers, `{"model":"test-model","prompt":"hello"}`)

			// Send headers
			require.NoError(t, h.Client.Send(reqs[0]), "failed to send headers")
			_, err := h.Client.Recv()
			require.NoError(t, err, "failed to receive headers response")

			// Send body and receive the response with header mutations
			require.NoError(t, h.Client.Send(reqs[1]), "failed to send body")
			res, err := h.Client.Recv()
			require.NoError(t, err, "failed to receive body response")

			// HandleRequestBody always sets BaseModelHeader via ds.GetBaseModel()
			wantSet := make(map[string]string, len(tc.wantSetHeaders)+1)
			for k, v := range tc.wantSetHeaders {
				wantSet[k] = v
			}
			wantSet["X-Gateway-Base-Model-Name"] = ""

			want := buildExpectedUnaryResponse(wantSet, tc.wantRemoved)
			envoytest.SortSetHeadersInResponses([]*extProcPb.ProcessingResponse{want})
			envoytest.SortSetHeadersInResponses([]*extProcPb.ProcessingResponse{res})

			if diff := cmp.Diff(want, res, protocmp.Transform()); diff != "" {
				t.Errorf("Response mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// buildExpectedUnaryResponse constructs the expected ProcessingResponse for a
// non-streaming integration test with the given header mutations.
func buildExpectedUnaryResponse(setHeaders map[string]string, removeHeaders []string) *extProcPb.ProcessingResponse {
	setHeaderOpts := make([]*envoyCorev3.HeaderValueOption, 0, len(setHeaders))
	for k, v := range setHeaders {
		setHeaderOpts = append(setHeaderOpts, &envoyCorev3.HeaderValueOption{
			Header: &envoyCorev3.HeaderValue{
				Key:      k,
				RawValue: []byte(v),
			},
		})
	}

	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestBody{
			RequestBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					ClearRouteCache: true,
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders:    setHeaderOpts,
						RemoveHeaders: removeHeaders,
					},
				},
			},
		},
	}
}
