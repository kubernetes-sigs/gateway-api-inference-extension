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

	// Assert BBR metrics: 2 model not in body, 1 model empty string, 4 successful model-from-body cases.
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
													Key:      basemodelextractor.BaseModelHeader,
													RawValue: []byte(""),
												},
											},
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
													Key:      basemodelextractor.BaseModelHeader,
													RawValue: []byte(""),
												},
											},
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
