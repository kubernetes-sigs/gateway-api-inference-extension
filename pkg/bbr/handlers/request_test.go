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

package handlers

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	basepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/testing/protocmp"
	metricsutils "k8s.io/component-base/metrics/testutil"
	crmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins/basemodelextractor"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins/bodyfieldtoheader"
	envoytest "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/test"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	epp "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const modelField = "model"

// === Request Headers Tests ===

func TestHandleRequestHeaders(t *testing.T) {
	tests := []struct {
		name         string
		headers      *extProcPb.HttpHeaders
		streaming    bool
		wantHeaders  map[string]string
		wantResponse []*extProcPb.ProcessingResponse
	}{
		{
			name: "headers response in non-streaming",
			headers: &extProcPb.HttpHeaders{
				Headers: &basepb.HeaderMap{
					Headers: []*basepb.HeaderValue{
						{Key: "content-type", RawValue: []byte("application/json")},
						{Key: "x-request-id", RawValue: []byte("abc-123")},
					},
				},
			},
			streaming: false,
			wantHeaders: map[string]string{
				"content-type": "application/json",
				"x-request-id": "abc-123",
			},
			wantResponse: []*extProcPb.ProcessingResponse{
				{Response: &extProcPb.ProcessingResponse_RequestHeaders{RequestHeaders: &extProcPb.HeadersResponse{}}},
			},
		},
		{
			name: "extracts headers in streaming, but not end of stream",
			headers: &extProcPb.HttpHeaders{
				Headers: &basepb.HeaderMap{
					Headers: []*basepb.HeaderValue{
						{Key: "content-type", RawValue: []byte("application/json")},
						{Key: "x-request-id", RawValue: []byte("abc-123")},
					},
				},
			},
			streaming: true,
			wantHeaders: map[string]string{
				"content-type": "application/json",
				"x-request-id": "abc-123",
			},
			wantResponse: nil,
		},
		{
			name: "extracts headers in streaming and end of stream",
			headers: &extProcPb.HttpHeaders{
				Headers: &basepb.HeaderMap{
					Headers: []*basepb.HeaderValue{
						{Key: "content-type", RawValue: []byte("application/json")},
						{Key: "x-request-id", RawValue: []byte("abc-123")},
					},
				},
				EndOfStream: true,
			},
			streaming: true,
			wantHeaders: map[string]string{
				"content-type": "application/json",
				"x-request-id": "abc-123",
			},
			wantResponse: []*extProcPb.ProcessingResponse{
				{Response: &extProcPb.ProcessingResponse_RequestHeaders{RequestHeaders: &extProcPb.HeadersResponse{}}},
			},
		},
		{
			name: "prefers RawValue over Value",
			headers: &extProcPb.HttpHeaders{
				Headers: &basepb.HeaderMap{
					Headers: []*basepb.HeaderValue{
						{Key: "x-test", RawValue: []byte("raw"), Value: "plain"},
					},
				},
			},
			streaming: false,
			wantHeaders: map[string]string{
				"x-test": "raw",
			},
			wantResponse: []*extProcPb.ProcessingResponse{
				{Response: &extProcPb.ProcessingResponse_RequestHeaders{RequestHeaders: &extProcPb.HeadersResponse{}}},
			},
		},
		{
			name: "falls back to Value when RawValue is empty",
			headers: &extProcPb.HttpHeaders{
				Headers: &basepb.HeaderMap{
					Headers: []*basepb.HeaderValue{
						{Key: "x-test", Value: "plain"},
					},
				},
			},
			streaming: false,
			wantHeaders: map[string]string{
				"x-test": "plain",
			},
			wantResponse: []*extProcPb.ProcessingResponse{
				{Response: &extProcPb.ProcessingResponse_RequestHeaders{RequestHeaders: &extProcPb.HeadersResponse{}}},
			},
		},
		{
			name:        "nil headers",
			headers:     nil,
			streaming:   false,
			wantHeaders: map[string]string{},
			wantResponse: []*extProcPb.ProcessingResponse{
				{Response: &extProcPb.ProcessingResponse_RequestHeaders{RequestHeaders: &extProcPb.HeadersResponse{}}},
			},
		},
		{
			name:        "nil header map",
			headers:     &extProcPb.HttpHeaders{},
			streaming:   false,
			wantHeaders: map[string]string{},
			wantResponse: []*extProcPb.ProcessingResponse{
				{Response: &extProcPb.ProcessingResponse_RequestHeaders{RequestHeaders: &extProcPb.HeadersResponse{}}},
			},
		},

		{
			name: "empty header map",
			headers: &extProcPb.HttpHeaders{
				Headers: &basepb.HeaderMap{
					Headers: []*basepb.HeaderValue{},
				},
			},
			streaming:   false,
			wantHeaders: map[string]string{},
			wantResponse: []*extProcPb.ProcessingResponse{
				{Response: &extProcPb.ProcessingResponse_RequestHeaders{RequestHeaders: &extProcPb.HeadersResponse{}}},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := NewServer(false, []framework.RequestProcessor{}, []framework.ResponseProcessor{})
			reqCtx := &RequestContext{
				Request: framework.NewInferenceRequest(),
			}

			resp := server.HandleRequestHeaders(context.Background(), reqCtx, tc.headers, tc.streaming)
			if reqCtx.RequestReceivedTimestamp.IsZero() {
				t.Error("RequestReceivedTimestamp was not set")
			}

			if diff := cmp.Diff(tc.wantResponse, resp, protocmp.Transform()); diff != "" {
				t.Errorf("HandleRequestHeaders response diff(-want, +got): %v", diff)
			}

			if diff := cmp.Diff(tc.wantHeaders, reqCtx.Request.Headers); diff != "" {
				t.Errorf("Extracted headers diff (-want, +got): %v", diff)
			}
		})
	}
}

// === Request Body Tests (built-in plugins) ===

func TestHandleRequestBody(t *testing.T) {
	metrics.Register()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	tests := []struct {
		name      string
		body      map[string]any
		streaming bool
		want      []*extProcPb.ProcessingResponse
		wantErr   bool
	}{
		{
			name: "model not found - skips gracefully",
			body: map[string]any{"prompt": "Tell me a joke"},
			want: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation:  &extProcPb.HeaderMutation{},
							},
						},
					},
				},
			},
		},
		{
			name:      "model not found with streaming - skips gracefully",
			body:      map[string]any{"prompt": "Tell me a joke"},
			streaming: true,
			want: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestHeaders{
						RequestHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*basepb.HeaderValueOption{
										{
											Header: &basepb.HeaderValue{
												Key:      "Content-Length",
												RawValue: []byte("27"),
											},
										},
									},
								},
							},
						},
					},
				},
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte(`{"prompt":"Tell me a joke"}`),
											EndOfStream: true,
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "model in body but empty - skips gracefully",
			body: map[string]any{"model": "", "prompt": "Tell me a joke"},
			want: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*basepb.HeaderValueOption{
										{
											Header: &basepb.HeaderValue{
												Key: basemodelextractor.BaseModelHeader,
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "model is not string, success after it's being auto converted to string",
			body: map[string]any{
				"model":  1,
				"prompt": "Tell me a joke",
			},
			want: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*basepb.HeaderValueOption{
										{
											Header: &basepb.HeaderValue{
												Key:      bodyfieldtoheader.ModelHeader,
												RawValue: []byte("1"),
											},
										},
										{
											Header: &basepb.HeaderValue{
												Key:      basemodelextractor.BaseModelHeader,
												RawValue: []byte(""),
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "success",
			body: map[string]any{
				"model":  "foo",
				"prompt": "Tell me a joke",
			},
			want: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*basepb.HeaderValueOption{
										{
											Header: &basepb.HeaderValue{
												Key:      bodyfieldtoheader.ModelHeader,
												RawValue: []byte("foo"),
											},
										},
										{
											Header: &basepb.HeaderValue{
												Key:      basemodelextractor.BaseModelHeader,
												RawValue: []byte(""),
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "success-with-streaming",
			body: map[string]any{
				"model":  "foo",
				"prompt": "Tell me a joke",
			},
			streaming: true,
			want: func() []*extProcPb.ProcessingResponse {
				b, _ := json.Marshal(map[string]any{"model": "foo", "prompt": "Tell me a joke"})
				return []*extProcPb.ProcessingResponse{
					{
						Response: &extProcPb.ProcessingResponse_RequestHeaders{
							RequestHeaders: &extProcPb.HeadersResponse{
								Response: &extProcPb.CommonResponse{
									ClearRouteCache: true,
									HeaderMutation: &extProcPb.HeaderMutation{
										SetHeaders: []*basepb.HeaderValueOption{
											{
												Header: &basepb.HeaderValue{
													Key:      contentLengthHeader,
													RawValue: []byte(strconv.Itoa(len(b))),
												},
											},
											{
												Header: &basepb.HeaderValue{
													Key:      bodyfieldtoheader.ModelHeader,
													RawValue: []byte("foo"),
												},
											},
											{
												Header: &basepb.HeaderValue{
													Key:      basemodelextractor.BaseModelHeader,
													RawValue: []byte(""),
												},
											},
										},
									},
								},
							},
						},
					},
					{
						Response: &extProcPb.ProcessingResponse_RequestBody{
							RequestBody: &extProcPb.BodyResponse{
								Response: &extProcPb.CommonResponse{
									BodyMutation: &extProcPb.BodyMutation{
										Mutation: &extProcPb.BodyMutation_StreamedResponse{
											StreamedResponse: &extProcPb.StreamedBodyResponse{
												Body:        b,
												EndOfStream: true,
											},
										},
									},
								},
							},
						},
					},
				}
			}(),
		},
		{
			name: "success-with-streaming-large-body",
			body: func() map[string]any {
				return map[string]any{
					"model":  "foo",
					"prompt": strings.Repeat("a", 70000),
				}
			}(),
			streaming: true,
			want: func() []*extProcPb.ProcessingResponse {
				m := map[string]any{
					"model":  "foo",
					"prompt": strings.Repeat("a", 70000),
				}
				b, _ := json.Marshal(m)
				limit := 62000
				return []*extProcPb.ProcessingResponse{
					{
						Response: &extProcPb.ProcessingResponse_RequestHeaders{
							RequestHeaders: &extProcPb.HeadersResponse{
								Response: &extProcPb.CommonResponse{
									ClearRouteCache: true,
									HeaderMutation: &extProcPb.HeaderMutation{
										SetHeaders: []*basepb.HeaderValueOption{
											{
												Header: &basepb.HeaderValue{
													Key:      contentLengthHeader,
													RawValue: []byte(strconv.Itoa(len(b))),
												},
											},
											{
												Header: &basepb.HeaderValue{
													Key:      bodyfieldtoheader.ModelHeader,
													RawValue: []byte("foo"),
												},
											},
											{
												Header: &basepb.HeaderValue{
													Key:      basemodelextractor.BaseModelHeader,
													RawValue: []byte(""),
												},
											},
										},
									},
								},
							},
						},
					},
					{
						Response: &extProcPb.ProcessingResponse_RequestBody{
							RequestBody: &extProcPb.BodyResponse{
								Response: &extProcPb.CommonResponse{
									BodyMutation: &extProcPb.BodyMutation{
										Mutation: &extProcPb.BodyMutation_StreamedResponse{
											StreamedResponse: &extProcPb.StreamedBodyResponse{
												Body:        b[:limit],
												EndOfStream: false,
											},
										},
									},
								},
							},
						},
					},
					{
						Response: &extProcPb.ProcessingResponse_RequestBody{
							RequestBody: &extProcPb.BodyResponse{
								Response: &extProcPb.CommonResponse{
									BodyMutation: &extProcPb.BodyMutation{
										Mutation: &extProcPb.BodyMutation_StreamedResponse{
											StreamedResponse: &extProcPb.StreamedBodyResponse{
												Body:        b[limit:],
												EndOfStream: true,
											},
										},
									},
								},
							},
						},
					},
				}
			}(),
		},
	}

	baseModelToHeaderPlugin := &basemodelextractor.BaseModelToHeaderPlugin{AdaptersStore: basemodelextractor.NewAdaptersStore()}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			modelToHeaderPlugin, _ := bodyfieldtoheader.NewBodyFieldToHeaderPlugin(modelField, bodyfieldtoheader.ModelHeader)
			server := NewServer(test.streaming, []framework.RequestProcessor{modelToHeaderPlugin, baseModelToHeaderPlugin}, []framework.ResponseProcessor{})
			reqCtx := &RequestContext{
				CycleState: framework.NewCycleState(),
				Request:    framework.NewInferenceRequest(),
			}
			bodyBytes, _ := json.Marshal(test.body)
			resp, err := server.HandleRequestBody(ctx, reqCtx, bodyBytes)
			if err != nil {
				if !test.wantErr {
					t.Fatalf("HandleRequestBody returned unexpected error: %v, want %v", err, test.wantErr)
				}
				return
			}

			// sort headers in responses for deterministic tests
			envoytest.SortSetHeadersInResponses(test.want)
			envoytest.SortSetHeadersInResponses(resp)
			if diff := cmp.Diff(test.want, resp, protocmp.Transform()); diff != "" {
				t.Errorf("HandleRequestBody returned unexpected response, diff(-want, +got): %v", diff)
			}
		})
	}

	// Assert BBR metrics: 2 model not in body, 1 model empty string, 7 successful model-from-body cases.
	wantMetrics := `
	# HELP bbr_body_field_empty_total [ALPHA] Count of times a field was found in a request body but was empty.
	# TYPE bbr_body_field_empty_total counter
	bbr_body_field_empty_total{field="model"} 1
	# HELP bbr_body_field_not_found_total [ALPHA] Count of times a field wasn't found in a request body.
	# TYPE bbr_body_field_not_found_total counter
	bbr_body_field_not_found_total{field="model"} 2
	# HELP bbr_success_total [ALPHA] Count of time the request was processed successfully.
	# TYPE bbr_success_total counter
	bbr_success_total{} 7
	`

	if err := metricsutils.GatherAndCompare(crmetrics.Registry, strings.NewReader(wantMetrics),
		"bbr_body_field_empty_total", "bbr_body_field_not_found_total", "bbr_success_total"); err != nil {
		t.Error(err)
	}
}

func TestHandleRequestBodyWithPluginMetrics(t *testing.T) {
	metrics.Register()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	modelToHeaderPlugin, _ := bodyfieldtoheader.NewBodyFieldToHeaderPlugin(modelField, bodyfieldtoheader.ModelHeader)
	baseModelToHeaderPlugin := &basemodelextractor.BaseModelToHeaderPlugin{AdaptersStore: basemodelextractor.NewAdaptersStore()}
	server := NewServer(false, []framework.RequestProcessor{modelToHeaderPlugin, baseModelToHeaderPlugin}, []framework.ResponseProcessor{})
	reqCtx := &RequestContext{
		CycleState: framework.NewCycleState(),
		Request:    framework.NewInferenceRequest(),
	}

	bodyBytes, _ := json.Marshal(map[string]any{
		"model":  "bar",
		"prompt": "test",
	})
	_, err := server.HandleRequestBody(ctx, reqCtx, bodyBytes)
	if err != nil {
		t.Fatalf("HandleRequestBody returned unexpected error: %v", err)
	}

	mfs, err := crmetrics.Registry.Gather()
	if err != nil {
		t.Fatalf("Failed to gather metrics: %v", err)
	}

	pluginsWithMetrics := 0
	for _, mf := range mfs {
		if mf.GetName() == "bbr_plugin_duration_seconds" {
			for _, m := range mf.GetMetric() {
				labels := map[string]string{}
				for _, lp := range m.GetLabel() {
					labels[lp.GetName()] = lp.GetValue()
				}
				if labels["extension_point"] == requestPluginExtensionPoint && m.GetHistogram().GetSampleCount() > 0 {
					pluginsWithMetrics++
				}
			}
		}
	}

	if pluginsWithMetrics != 2 {
		t.Errorf("Expected 2 request plugins with metrics observations, got %d", pluginsWithMetrics)
	}
}

// === Request Body Tests (multi-plugin header mutations) ===

// fakeRequestPlugin implements framework.RequestProcessor for testing
// multi-plugin header mutation scenarios.
type fakeRequestPlugin struct {
	name     string
	mutateFn func(ctx context.Context, request *framework.InferenceRequest) error
}

func (p *fakeRequestPlugin) TypedName() epp.TypedName {
	return epp.TypedName{Type: "fake", Name: p.name}
}

func (p *fakeRequestPlugin) ProcessRequest(ctx context.Context, _ *framework.CycleState, request *framework.InferenceRequest) error {
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
			server := NewServer(false, tc.plugins, []framework.ResponseProcessor{})
			reqCtx := &RequestContext{
				Request:    framework.NewInferenceRequest(),
				CycleState: framework.NewCycleState(),
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

			want := buildNonStreamingResponse(tc.wantSetHeaders, tc.wantRemoved)
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

// === Request Body Tests (body mutations) ===

type bodyMutatingPlugin struct {
	name     string
	mutateFn func(ctx context.Context, cycleState *framework.CycleState, request *framework.InferenceRequest) error
}

func (p *bodyMutatingPlugin) TypedName() epp.TypedName {
	return epp.TypedName{Type: "fake", Name: p.name}
}

func (p *bodyMutatingPlugin) ProcessRequest(ctx context.Context, cycleState *framework.CycleState, request *framework.InferenceRequest) error {
	return p.mutateFn(ctx, cycleState, request)
}

var _ framework.RequestProcessor = &bodyMutatingPlugin{}

func TestHandleRequestBody_BodyMutation(t *testing.T) {
	metrics.Register()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	plugin := &bodyMutatingPlugin{
		name: "body-mutator",
		mutateFn: func(_ context.Context, _ *framework.CycleState, request *framework.InferenceRequest) error {
			request.SetBodyField("injected", "value")
			return nil
		},
	}

	tests := []struct {
		name      string
		streaming bool
		body      map[string]any
		want      []*extProcPb.ProcessingResponse
	}{
		{
			name: "unary with body mutation",
			body: map[string]any{
				"prompt": "test",
			},
			want: func() []*extProcPb.ProcessingResponse {
				b, _ := json.Marshal(map[string]any{"prompt": "test", "injected": "value"})
				return []*extProcPb.ProcessingResponse{
					{
						Response: &extProcPb.ProcessingResponse_RequestBody{
							RequestBody: &extProcPb.BodyResponse{
								Response: &extProcPb.CommonResponse{
									ClearRouteCache: true,
									HeaderMutation: &extProcPb.HeaderMutation{
										SetHeaders: []*basepb.HeaderValueOption{
											{
												Header: &basepb.HeaderValue{
													Key:      contentLengthHeader,
													RawValue: []byte(strconv.Itoa(len(b))),
												},
											},
										},
									},
									BodyMutation: &extProcPb.BodyMutation{
										Mutation: &extProcPb.BodyMutation_Body{
											Body: b,
										},
									},
								},
							},
						},
					},
				}
			}(),
		},
		{
			name:      "streaming with body mutation",
			streaming: true,
			body: map[string]any{
				"prompt": "test",
			},
			want: func() []*extProcPb.ProcessingResponse {
				b, _ := json.Marshal(map[string]any{"prompt": "test", "injected": "value"})
				return []*extProcPb.ProcessingResponse{
					{
						Response: &extProcPb.ProcessingResponse_RequestHeaders{
							RequestHeaders: &extProcPb.HeadersResponse{
								Response: &extProcPb.CommonResponse{
									ClearRouteCache: true,
									HeaderMutation: &extProcPb.HeaderMutation{
										SetHeaders: []*basepb.HeaderValueOption{
											{
												Header: &basepb.HeaderValue{
													Key:      contentLengthHeader,
													RawValue: []byte(strconv.Itoa(len(b))),
												},
											},
										},
									},
								},
							},
						},
					},
					{
						Response: &extProcPb.ProcessingResponse_RequestBody{
							RequestBody: &extProcPb.BodyResponse{
								Response: &extProcPb.CommonResponse{
									BodyMutation: &extProcPb.BodyMutation{
										Mutation: &extProcPb.BodyMutation_StreamedResponse{
											StreamedResponse: &extProcPb.StreamedBodyResponse{
												Body:        b,
												EndOfStream: true,
											},
										},
									},
								},
							},
						},
					},
				}
			}(),
		},
	}

	baseModelPlugin := &basemodelextractor.BaseModelToHeaderPlugin{AdaptersStore: basemodelextractor.NewAdaptersStore()}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := NewServer(tc.streaming, []framework.RequestProcessor{plugin, baseModelPlugin}, []framework.ResponseProcessor{})
			reqCtx := &RequestContext{
				CycleState: framework.NewCycleState(),
				Request:    framework.NewInferenceRequest(),
			}
			bodyBytes, _ := json.Marshal(tc.body)
			resp, err := server.HandleRequestBody(ctx, reqCtx, bodyBytes)
			if err != nil {
				t.Fatalf("HandleRequestBody returned unexpected error: %v", err)
			}

			envoytest.SortSetHeadersInResponses(tc.want)
			envoytest.SortSetHeadersInResponses(resp)
			if diff := cmp.Diff(tc.want, resp, protocmp.Transform()); diff != "" {
				t.Errorf("HandleRequestBody returned unexpected response, diff(-want, +got): %v", diff)
			}
		})
	}
}
