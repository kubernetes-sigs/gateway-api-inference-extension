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

package handlers

import (
	"context"
	"encoding/json"
	"testing"

	basepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/testing/protocmp"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	envoytest "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/test"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	epp "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// fakeRequestPlugin implements framework.RequestProcessor for testing
// multi-plugin header mutation scenarios.
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

// TestHandleRequestBody_MultiPluginHeaderMutations tests the end-to-end behavior of
// HandleRequestBody when multiple request plugins set and/or remove headers.
// Each sub-test verifies the HeaderMutation in the resulting ProcessingResponse.
func TestHandleRequestBody_MultiPluginHeaderMutations(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	tests := []struct {
		name           string
		plugins        []framework.RequestProcessor
		initialHeaders map[string]string
		wantSetHeaders map[string]string
		wantRemoved    []string
	}{
		{
			// Plugin1 adds X-Custom, Plugin2 removes it.
			// The header was never in the original Envoy request, so Envoy treats
			// the removal as a no-op. The net visible effect is: nothing changed.
			// However, RemoveHeader() does record it in removedHeaders because
			// the key existed in Headers at the time of removal.
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
			// Plugin1 adds a new header, Plugin2 removes a pre-existing one.
			// Both mutations should appear in the response.
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
			initialHeaders: map[string]string{
				"X-Existing": "old-value",
			},
			wantSetHeaders: map[string]string{
				"X-New": "hello",
			},
			wantRemoved: []string{"X-Existing"},
		},
		{
			// RemoveHeader on a key that was never in Headers is a no-op:
			// the guard `if _, ok := r.Headers[key]; ok` prevents any mutation.
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
			// Plugin1 removes a pre-existing header, Plugin2 re-sets it.
			// SetHeader clears the key from removedHeaders, so the final result
			// is a set with the new value and no removal.
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
			initialHeaders: map[string]string{
				"X-Reuse": "old-value",
			},
			wantSetHeaders: map[string]string{
				"X-Reuse": "new-value",
			},
			wantRemoved: nil,
		},
		{
			// Both plugins set the same header key. Plugins run sequentially,
			// so the last writer wins.
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
			// Two plugins set different header keys. Both should appear in the response.
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
			// Two plugins both remove the same pre-existing header.
			// The second RemoveHeader is a no-op because the header is already gone.
			// The header should appear exactly once in removedHeaders.
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
			initialHeaders: map[string]string{
				"X-Dup": "value",
			},
			wantSetHeaders: map[string]string{},
			wantRemoved:    []string{"X-Dup"},
		},
		{
			// A plugin sets a header to the same value it already has.
			// The SetHeader optimization (types.go:43) should skip the mutation.
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
			initialHeaders: map[string]string{
				"X-Keep": "original",
			},
			wantSetHeaders: map[string]string{},
			wantRemoved:    nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := NewServer(false, &fakeDatastore{}, tc.plugins, []framework.ResponseProcessor{})
			reqCtx := &RequestContext{
				Request: framework.NewInferenceRequest(),
			}
			for k, v := range tc.initialHeaders {
				reqCtx.Request.Headers[k] = v
			}

			bodyBytes, err := json.Marshal(map[string]any{"model": "test-model", "prompt": "test"})
			if err != nil {
				t.Fatalf("Failed to marshal request body: %v", err)
			}

			resp, err := server.HandleRequestBody(ctx, reqCtx, bodyBytes)
			if err != nil {
				t.Fatalf("HandleRequestBody returned unexpected error: %v", err)
			}

			// HandleRequestBody always calls ds.GetBaseModel() and sets BaseModelHeader
			// after plugins run, regardless of which plugins are registered.
			wantSet := make(map[string]string, len(tc.wantSetHeaders)+1)
			for k, v := range tc.wantSetHeaders {
				wantSet[k] = v
			}
			wantSet[BaseModelHeader] = ""

			want := buildNonStreamingResponse(wantSet, tc.wantRemoved)
			envoytest.SortSetHeadersInResponses(want)
			envoytest.SortSetHeadersInResponses(resp)

			if diff := cmp.Diff(want, resp, protocmp.Transform()); diff != "" {
				t.Errorf("HandleRequestBody response mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// buildNonStreamingResponse constructs the expected ProcessingResponse for a
// non-streaming HandleRequestBody call with the given header mutations.
func buildNonStreamingResponse(setHeaders map[string]string, removeHeaders []string) []*extProcPb.ProcessingResponse {
	setHeaderOpts := make([]*basepb.HeaderValueOption, 0, len(setHeaders))
	for k, v := range setHeaders {
		setHeaderOpts = append(setHeaderOpts, &basepb.HeaderValueOption{
			Header: &basepb.HeaderValue{
				Key:      k,
				RawValue: []byte(v),
			},
		})
	}

	return []*extProcPb.ProcessingResponse{
		{
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
		},
	}
}
