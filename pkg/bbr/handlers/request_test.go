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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins"
	envoytest "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/test"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
)

func TestHandleRequestHeaders(t *testing.T) {
	tests := []struct {
		name        string
		headers     *extProcPb.HttpHeaders
		wantHeaders map[string]string
	}{
		{
			name: "extracts headers",
			headers: &extProcPb.HttpHeaders{
				Headers: &basepb.HeaderMap{
					Headers: []*basepb.HeaderValue{
						{Key: "content-type", RawValue: []byte("application/json")},
						{Key: "x-request-id", RawValue: []byte("abc-123")},
					},
				},
			},
			wantHeaders: map[string]string{
				"content-type": "application/json",
				"x-request-id": "abc-123",
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
			wantHeaders: map[string]string{
				"x-test": "raw",
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
			wantHeaders: map[string]string{
				"x-test": "plain",
			},
		},
		{
			name:        "nil headers",
			headers:     nil,
			wantHeaders: map[string]string{},
		},
		{
			name:        "nil header map",
			headers:     &extProcPb.HttpHeaders{},
			wantHeaders: map[string]string{},
		},
		{
			name: "empty header map",
			headers: &extProcPb.HttpHeaders{
				Headers: &basepb.HeaderMap{
					Headers: []*basepb.HeaderValue{},
				},
			},
			wantHeaders: map[string]string{},
		},
	}

	wantResp := []*extProcPb.ProcessingResponse{
		{Response: &extProcPb.ProcessingResponse_RequestHeaders{RequestHeaders: &extProcPb.HeadersResponse{}}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := NewServer(false, &fakeDatastore{}, []framework.RequestProcessor{}, []framework.ResponseProcessor{})
			reqCtx := &RequestContext{
				Request: framework.NewInferenceRequest(),
			}

			resp, err := server.HandleRequestHeaders(reqCtx, tc.headers)
			if err != nil {
				t.Fatalf("HandleRequestHeaders returned unexpected error: %v", err)
			}

			if reqCtx.RequestReceivedTimestamp.IsZero() {
				t.Error("RequestReceivedTimestamp was not set")
			}

			if diff := cmp.Diff(wantResp, resp, protocmp.Transform()); diff != "" {
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
			name: "model not found",
			body: map[string]any{
				"prompt": "Tell me a joke",
			},
			want: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{},
					},
				},
			},
		},
		{
			name: "model not found with streaming",
			body: map[string]any{
				"prompt": "Tell me a joke",
			},
			streaming: true,
			want: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestHeaders{
						RequestHeaders: &extProcPb.HeadersResponse{},
					},
				},
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body: mapToBytes(t, map[string]any{
												"prompt": "Tell me a joke",
											}),
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
			name: "model in body but empty",
			body: map[string]any{
				"model":  "",
				"prompt": "Tell me a joke",
			},
			want: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{},
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
			want: func() []*extProcPb.ProcessingResponse {
				b, _ := json.Marshal(map[string]any{"model": 1, "prompt": "Tell me a joke"})
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
													Key:      ModelHeader,
													RawValue: []byte("1"),
												},
											},
											{
												Header: &basepb.HeaderValue{
													Key:      BaseModelHeader,
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
			name: "success",
			body: map[string]any{
				"model":  "foo",
				"prompt": "Tell me a joke",
			},
			want: func() []*extProcPb.ProcessingResponse {
				b, _ := json.Marshal(map[string]any{"model": "foo", "prompt": "Tell me a joke"})
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
													Key:      ModelHeader,
													RawValue: []byte("foo"),
												},
											},
											{
												Header: &basepb.HeaderValue{
													Key:      BaseModelHeader,
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
													Key:      ModelHeader,
													RawValue: []byte("foo"),
												},
											},
											{
												Header: &basepb.HeaderValue{
													Key:      BaseModelHeader,
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
													Key:      ModelHeader,
													RawValue: []byte("foo"),
												},
											},
											{
												Header: &basepb.HeaderValue{
													Key:      BaseModelHeader,
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

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			modelToHeaderPlugin, _ := plugins.NewBodyFieldToHeaderPlugin(ModelField, ModelHeader)
			server := NewServer(test.streaming, &fakeDatastore{}, []framework.RequestProcessor{modelToHeaderPlugin}, []framework.ResponseProcessor{})
			reqCtx := &RequestContext{
				Request: framework.NewInferenceRequest(),
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
	bbr_success_total{} 4
	`

	if err := metricsutils.GatherAndCompare(crmetrics.Registry, strings.NewReader(wantMetrics),
		"bbr_body_field_empty_total", "bbr_body_field_not_found_total", "bbr_success_total"); err != nil {
		t.Error(err)
	}
}

func TestHandleRequestBodyWithPluginMetrics(t *testing.T) {
	metrics.Register()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	plugin, _ := plugins.NewBodyFieldToHeaderPlugin(ModelField, ModelHeader)
	server := NewServer(false, &fakeDatastore{}, []framework.RequestProcessor{plugin}, []framework.ResponseProcessor{})
	reqCtx := &RequestContext{
		Request: framework.NewInferenceRequest(),
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

	found := false
	for _, mf := range mfs {
		if mf.GetName() == "bbr_plugin_duration_seconds" {
			for _, m := range mf.GetMetric() {
				labels := map[string]string{}
				for _, lp := range m.GetLabel() {
					labels[lp.GetName()] = lp.GetValue()
				}
				if labels["extension_point"] == requestPluginExtensionPoint &&
					labels["plugin_type"] == plugins.BodyFieldToHeaderPluginType &&
					labels["plugin_name"] == plugins.BodyFieldToHeaderPluginType {
					if m.GetHistogram().GetSampleCount() > 0 {
						found = true
					}
				}
			}
		}
	}

	if !found {
		t.Error("Expected bbr_plugin_duration_seconds metric with extension_point=request, " +
			"plugin_type=body-field-to-header, plugin_name=body-field-to-header to have observations, but none found")
	}
}

func mapToBytes(t *testing.T, m map[string]any) []byte {
	// Convert map to JSON byte array
	bytes, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("Marshal(): %v", err)
	}
	return bytes
}
