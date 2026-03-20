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
	"errors"
	"strconv"
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

const testPluginValue = "done"

// fakeResponsePlugin implements framework.PayloadProcessor for testing response plugin execution.
type fakeResponsePlugin struct {
	name     string
	mutateFn func(ctx context.Context, cycleState *framework.CycleState, response *framework.InferenceResponse) error
}

func (p *fakeResponsePlugin) TypedName() epp.TypedName {
	return epp.TypedName{Type: "fake", Name: p.name}
}

func (p *fakeResponsePlugin) ProcessResponse(ctx context.Context, cycleState *framework.CycleState, response *framework.InferenceResponse) error {
	return p.mutateFn(ctx, cycleState, response)
}

var _ framework.ResponseProcessor = &fakeResponsePlugin{}

func newTestRequestContext() *RequestContext {
	return &RequestContext{
		CycleState: framework.NewCycleState(),
		Request:    framework.NewInferenceRequest(),
		Response:   framework.NewInferenceResponse(),
	}
}

func TestHandleResponseBody_NoPlugins(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	t.Run("unary", func(t *testing.T) {
		server := NewServer(false, []framework.RequestProcessor{}, []framework.ResponseProcessor{})
		responseBody := []byte(`{"choices":[{"text":"Hello!"}]}`)
		resp, err := server.HandleResponseBody(ctx, newTestRequestContext(), responseBody)
		if err != nil {
			t.Fatalf("HandleResponseBody returned unexpected error: %v", err)
		}

		want := []*extProcPb.ProcessingResponse{
			{
				Response: &extProcPb.ProcessingResponse_ResponseBody{
					ResponseBody: &extProcPb.BodyResponse{},
				},
			},
		}

		if diff := cmp.Diff(want, resp, protocmp.Transform()); diff != "" {
			t.Errorf("HandleResponseBody returned unexpected response, diff(-want, +got): %v", diff)
		}
	})

	t.Run("streaming", func(t *testing.T) {
		server := NewServer(true, []framework.RequestProcessor{}, []framework.ResponseProcessor{})
		responseBody := []byte(`{"choices":[{"text":"Hello!"}]}`)
		resp, err := server.HandleResponseBody(ctx, newTestRequestContext(), responseBody)
		if err != nil {
			t.Fatalf("HandleResponseBody returned unexpected error: %v", err)
		}

		want := []*extProcPb.ProcessingResponse{
			{
				Response: &extProcPb.ProcessingResponse_ResponseHeaders{
					ResponseHeaders: &extProcPb.HeadersResponse{},
				},
			},
			{
				Response: &extProcPb.ProcessingResponse_ResponseBody{
					ResponseBody: &extProcPb.BodyResponse{
						Response: &extProcPb.CommonResponse{
							BodyMutation: &extProcPb.BodyMutation{
								Mutation: &extProcPb.BodyMutation_StreamedResponse{
									StreamedResponse: &extProcPb.StreamedBodyResponse{
										Body:        responseBody,
										EndOfStream: true,
									},
								},
							},
						},
					},
				},
			},
		}

		if diff := cmp.Diff(want, resp, protocmp.Transform()); diff != "" {
			t.Errorf("HandleResponseBody returned unexpected response, diff(-want, +got): %v", diff)
		}
	})
}

func TestHandleResponseBody_SinglePlugin(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	mutatePlugin := &fakeResponsePlugin{
		name: "mutator",
		mutateFn: func(_ context.Context, _ *framework.CycleState, response *framework.InferenceResponse) error {
			response.SetBodyField("mutated", true)
			return nil
		},
	}

	server := NewServer(false, []framework.RequestProcessor{}, []framework.ResponseProcessor{mutatePlugin})
	responseBody := []byte(`{"choices":[{"text":"Hello!"}]}`)
	resp, err := server.HandleResponseBody(ctx, newTestRequestContext(), responseBody)
	if err != nil {
		t.Fatalf("HandleResponseBody returned unexpected error: %v", err)
	}

	wantBody, _ := json.Marshal(map[string]any{
		"choices": []any{map[string]any{"text": "Hello!"}},
		"mutated": true,
	})
	want := []*extProcPb.ProcessingResponse{
		expectedResponseBodyMutation(wantBody),
	}

	envoytest.SortSetHeadersInResponses(want)
	envoytest.SortSetHeadersInResponses(resp)
	if diff := cmp.Diff(want, resp, protocmp.Transform()); diff != "" {
		t.Errorf("HandleResponseBody returned unexpected response, diff(-want, +got): %v", diff)
	}
}

func TestHandleResponseBody_MultiplePlugins(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	plugin1 := &fakeResponsePlugin{
		name: "plugin1",
		mutateFn: func(_ context.Context, _ *framework.CycleState, response *framework.InferenceResponse) error {
			response.SetBodyField("p1", testPluginValue)
			return nil
		},
	}
	plugin2 := &fakeResponsePlugin{
		name: "plugin2",
		mutateFn: func(_ context.Context, _ *framework.CycleState, response *framework.InferenceResponse) error {
			response.SetBodyField("p2", testPluginValue)
			return nil
		},
	}

	server := NewServer(false, []framework.RequestProcessor{}, []framework.ResponseProcessor{plugin1, plugin2})
	responseBody := []byte(`{"original":true}`)
	resp, err := server.HandleResponseBody(ctx, newTestRequestContext(), responseBody)
	if err != nil {
		t.Fatalf("HandleResponseBody returned unexpected error: %v", err)
	}

	wantBody, _ := json.Marshal(map[string]any{
		"original": true,
		"p1":       testPluginValue,
		"p2":       testPluginValue,
	})
	want := []*extProcPb.ProcessingResponse{
		expectedResponseBodyMutation(wantBody),
	}

	envoytest.SortSetHeadersInResponses(want)
	envoytest.SortSetHeadersInResponses(resp)
	if diff := cmp.Diff(want, resp, protocmp.Transform()); diff != "" {
		t.Errorf("HandleResponseBody returned unexpected response, diff(-want, +got): %v", diff)
	}
}

func TestHandleResponseBody_PluginError(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	failingPlugin := &fakeResponsePlugin{
		name: "failing",
		mutateFn: func(_ context.Context, _ *framework.CycleState, _ *framework.InferenceResponse) error {
			return errors.New("failed to execute plugin")
		},
	}

	server := NewServer(false, []framework.RequestProcessor{}, []framework.ResponseProcessor{failingPlugin})
	responseBody := []byte(`{"choices":[{"text":"some response"}]}`)
	_, err := server.HandleResponseBody(ctx, newTestRequestContext(), responseBody)
	if err == nil {
		t.Fatal("HandleResponseBody should have returned an error")
	}

	if got := err.Error(); got == "" {
		t.Error("Expected non-empty error message")
	}
}

func TestHandleResponseBody_StreamingWithPlugin(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	mutatePlugin := &fakeResponsePlugin{
		name: "mutator",
		mutateFn: func(_ context.Context, _ *framework.CycleState, response *framework.InferenceResponse) error {
			response.SetBodyField("mutated", true)
			return nil
		},
	}

	server := NewServer(true, []framework.RequestProcessor{}, []framework.ResponseProcessor{mutatePlugin})
	responseBody := []byte(`{"choices":[{"text":"Hello!"}]}`)
	resp, err := server.HandleResponseBody(ctx, newTestRequestContext(), responseBody)
	if err != nil {
		t.Fatalf("HandleResponseBody returned unexpected error: %v", err)
	}

	wantBody, _ := json.Marshal(map[string]any{
		"choices": []any{map[string]any{"text": "Hello!"}},
		"mutated": true,
	})
	want := expectedStreamedResponseBodyMutation(wantBody)

	envoytest.SortSetHeadersInResponses(want)
	envoytest.SortSetHeadersInResponses(resp)
	if diff := cmp.Diff(want, resp, protocmp.Transform()); diff != "" {
		t.Errorf("HandleResponseBody returned unexpected response, diff(-want, +got): %v", diff)
	}
}

func TestProcessResponseBody_Streaming(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	server := NewServer(true, []framework.RequestProcessor{}, []framework.ResponseProcessor{})

	chunk1 := &extProcPb.HttpBody{
		Body: []byte(`{"choices":[{"te`),
	}
	chunk2 := &extProcPb.HttpBody{
		Body:        []byte(`xt":"Hello!"}]}`),
		EndOfStream: true,
	}

	reqCtx := newTestRequestContext()
	respStreamedBody := &streamedBody{}

	resp1, err := server.processResponseBody(ctx, reqCtx, chunk1, respStreamedBody)
	if err != nil {
		t.Fatalf("processResponseBody chunk1 returned unexpected error: %v", err)
	}
	if resp1 != nil {
		t.Fatalf("processResponseBody chunk1 should return nil while buffering, got: %v", resp1)
	}

	resp2, err := server.processResponseBody(ctx, reqCtx, chunk2, respStreamedBody)
	if err != nil {
		t.Fatalf("processResponseBody chunk2 returned unexpected error: %v", err)
	}
	if resp2 == nil {
		t.Fatal("processResponseBody chunk2 should return a response on EoS")
	}
}

func TestHandleResponseBody_PluginNoBodyMutation(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	headerOnlyPlugin := &fakeResponsePlugin{
		name: "header-only",
		mutateFn: func(_ context.Context, _ *framework.CycleState, response *framework.InferenceResponse) error {
			response.SetHeader("X-Custom-Response", "added")
			return nil
		},
	}

	tests := []struct {
		name      string
		streaming bool
		want      []*extProcPb.ProcessingResponse
	}{
		{
			name: "unary - header-only plugin skips body mutation",
			want: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_ResponseBody{
						ResponseBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*basepb.HeaderValueOption{
										{
											Header: &basepb.HeaderValue{
												Key:      "X-Custom-Response",
												RawValue: []byte("added"),
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
			name:      "streaming - header-only plugin passes original body",
			streaming: true,
			want: func() []*extProcPb.ProcessingResponse {
				responseBody := []byte(`{"choices":[{"text":"Hello!"}]}`)
				return []*extProcPb.ProcessingResponse{
					{
						Response: &extProcPb.ProcessingResponse_ResponseHeaders{
							ResponseHeaders: &extProcPb.HeadersResponse{
								Response: &extProcPb.CommonResponse{
									ClearRouteCache: true,
									HeaderMutation: &extProcPb.HeaderMutation{
										SetHeaders: []*basepb.HeaderValueOption{
											{
												Header: &basepb.HeaderValue{
													Key:      "X-Custom-Response",
													RawValue: []byte("added"),
												},
											},
										},
									},
								},
							},
						},
					},
					{
						Response: &extProcPb.ProcessingResponse_ResponseBody{
							ResponseBody: &extProcPb.BodyResponse{
								Response: &extProcPb.CommonResponse{
									BodyMutation: &extProcPb.BodyMutation{
										Mutation: &extProcPb.BodyMutation_StreamedResponse{
											StreamedResponse: &extProcPb.StreamedBodyResponse{
												Body:        responseBody,
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

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := NewServer(tc.streaming, []framework.RequestProcessor{}, []framework.ResponseProcessor{headerOnlyPlugin})
			responseBody := []byte(`{"choices":[{"text":"Hello!"}]}`)
			resp, err := server.HandleResponseBody(ctx, newTestRequestContext(), responseBody)
			if err != nil {
				t.Fatalf("HandleResponseBody returned unexpected error: %v", err)
			}

			envoytest.SortSetHeadersInResponses(tc.want)
			envoytest.SortSetHeadersInResponses(resp)
			if diff := cmp.Diff(tc.want, resp, protocmp.Transform()); diff != "" {
				t.Errorf("HandleResponseBody returned unexpected response, diff(-want, +got): %v", diff)
			}
		})
	}
}

// expectedResponseBodyMutation builds the expected unary response for a mutated body,
// including the content-length header mutation.
func expectedResponseBodyMutation(bodyBytes []byte) *extProcPb.ProcessingResponse {
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseBody{
			ResponseBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					ClearRouteCache: true,
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: []*basepb.HeaderValueOption{
							{
								Header: &basepb.HeaderValue{
									Key:      contentLengthHeader,
									RawValue: []byte(strconv.Itoa(len(bodyBytes))),
								},
							},
						},
					},
					BodyMutation: &extProcPb.BodyMutation{
						Mutation: &extProcPb.BodyMutation_Body{
							Body: bodyBytes,
						},
					},
				},
			},
		},
	}
}

// expectedStreamedResponseBodyMutation builds the expected streamed response for a mutated body:
// a deferred ResponseHeaders with header mutation, then a ResponseBody with StreamedBodyResponse.
func expectedStreamedResponseBodyMutation(bodyBytes []byte) []*extProcPb.ProcessingResponse {
	return []*extProcPb.ProcessingResponse{
		{
			Response: &extProcPb.ProcessingResponse_ResponseHeaders{
				ResponseHeaders: &extProcPb.HeadersResponse{
					Response: &extProcPb.CommonResponse{
						ClearRouteCache: true,
						HeaderMutation: &extProcPb.HeaderMutation{
							SetHeaders: []*basepb.HeaderValueOption{
								{
									Header: &basepb.HeaderValue{
										Key:      contentLengthHeader,
										RawValue: []byte(strconv.Itoa(len(bodyBytes))),
									},
								},
							},
						},
					},
				},
			},
		},
		{
			Response: &extProcPb.ProcessingResponse_ResponseBody{
				ResponseBody: &extProcPb.BodyResponse{
					Response: &extProcPb.CommonResponse{
						BodyMutation: &extProcPb.BodyMutation{
							Mutation: &extProcPb.BodyMutation_StreamedResponse{
								StreamedResponse: &extProcPb.StreamedBodyResponse{
									Body:        bodyBytes,
									EndOfStream: true,
								},
							},
						},
					},
				},
			},
		},
	}
}
