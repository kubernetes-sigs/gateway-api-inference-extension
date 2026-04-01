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

// Package bbr contains integration tests for the body-based routing extension.
package bbr

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/testing/protocmp"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins/basemodelextractor"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins/bodyfieldtoheader"
	envoytest "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/test"
	epp "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

// TestBodyBasedRouting validates the "Unary" (Non-Streaming) request-phase behavior of BBR.
// This simulates scenarios where Envoy buffers the body before sending it to ext_proc.
func TestBodyBasedRouting(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name             string
		req              *extProcPb.ProcessingRequest
		wantResponse     *extProcPb.ProcessingResponse
		wantStatusCode   envoyTypePb.StatusCode
		wantBodyContains string
	}{
		{
			name:         "success: extracts model and sets header",
			req:          integration.ReqLLMUnary(logger, "test", "llama"),
			wantResponse: ExpectBBRUnaryResponse("llama", "llama", "test"),
		},
		{
			name: "no model parameter in body - skips gracefully",
			req:  integration.ReqLLMUnary(logger, "test1", ""),
			wantResponse: &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_RequestBody{
					RequestBody: &extProcPb.BodyResponse{
						Response: &extProcPb.CommonResponse{
							ClearRouteCache: true,
							HeaderMutation: &extProcPb.HeaderMutation{
								SetHeaders: []*envoyCorev3.HeaderValueOption{
									{
										Header: &envoyCorev3.HeaderValue{
											Key: "X-Gateway-Base-Model-Name",
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx := context.Background()
			h := NewBBRHarness(t, ctx, false)

			res, err := integration.SendRequest(t, h.Client, tc.req)
			require.NoError(t, err, "unexpected error during request processing")

			if tc.wantStatusCode != 0 {
				ir := res.GetImmediateResponse()
				require.NotNil(t, ir, "expected ImmediateResponse")
				require.Equal(t, tc.wantStatusCode, ir.GetStatus().GetCode())
				if tc.wantBodyContains != "" {
					require.True(t, strings.Contains(string(ir.GetBody()), tc.wantBodyContains),
						"ImmediateResponse body %s should contain %s", string(ir.GetBody()), tc.wantBodyContains)
				}
				return
			}

			envoytest.SortSetHeadersInResponses([]*extProcPb.ProcessingResponse{tc.wantResponse})
			envoytest.SortSetHeadersInResponses([]*extProcPb.ProcessingResponse{res})
			if diff := cmp.Diff(tc.wantResponse, res, protocmp.Transform()); diff != "" {
				t.Errorf("Response mismatch (-want +got): %v", diff)
			}
		})
	}
}

// TestResponsePlugins validates the full request→response lifecycle in unary mode,
// testing that response plugins can mutate the response body and that responses
// pass through unchanged when no response plugins are configured.
func TestResponsePlugins(t *testing.T) {
	t.Parallel()

	responsePlugin := &testResponsePlugin{
		name: "guardrail",
		mutateFn: func(_ context.Context, response *framework.InferenceResponse) error {
			response.SetBodyField("guardrail", "applied")
			return nil
		},
	}

	respHeaders := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{
					Headers: []*envoyCorev3.HeaderValue{
						{Key: "content-type", Value: "application/json"},
					},
				},
			},
		},
	}
	respBodyReq := func(body map[string]any) *extProcPb.ProcessingRequest {
		b, _ := json.Marshal(body)
		return &extProcPb.ProcessingRequest{
			Request: &extProcPb.ProcessingRequest_ResponseBody{
				ResponseBody: &extProcPb.HttpBody{Body: b, EndOfStream: true},
			},
		}
	}

	tests := []struct {
		name          string
		createHarness func(t *testing.T, ctx context.Context) *BBRHarness
		reqs          []*extProcPb.ProcessingRequest
		wantResponses []*extProcPb.ProcessingResponse
	}{
		{
			name: "no plugins: response passes through unchanged",
			createHarness: func(t *testing.T, ctx context.Context) *BBRHarness {
				return NewBBRHarness(t, ctx, false)
			},
			reqs: []*extProcPb.ProcessingRequest{
				integration.ReqLLMUnary(logger, "test", "llama"),
				respHeaders,
				respBodyReq(map[string]any{"choices": []any{map[string]any{"text": "Hi there!"}}}),
			},
			wantResponses: []*extProcPb.ProcessingResponse{
				ExpectBBRUnaryResponse("llama", "llama", "test"),
				ExpectResponseHeadersPassThrough(),
				ExpectResponseBodyPassThrough(),
			},
		},
		{
			name: "response plugin mutates response body",
			createHarness: func(t *testing.T, ctx context.Context) *BBRHarness {
				t.Helper()
				modelToHeaderPlugin, err := bodyfieldtoheader.NewBodyFieldToHeaderPlugin(modelField, bodyfieldtoheader.ModelHeader)
				require.NoError(t, err, "failed to create body-field-to-header plugin")
				baseModelPlugin := &basemodelextractor.BaseModelToHeaderPlugin{AdaptersStore: basemodelextractor.NewAdaptersStore()}
				return NewBBRHarnessWithPlugins(t, ctx, false,
					[]framework.RequestProcessor{modelToHeaderPlugin, baseModelPlugin},
					[]framework.ResponseProcessor{responsePlugin},
				)
			},
			reqs: []*extProcPb.ProcessingRequest{
				integration.ReqLLMUnary(logger, "hello", "test-model"),
				respHeaders,
				respBodyReq(map[string]any{"choices": []any{map[string]any{"text": "Hello!"}}}),
			},
			wantResponses: []*extProcPb.ProcessingResponse{
				ExpectBBRUnaryResponse("test-model", "", "hello"),
				ExpectResponseHeadersPassThrough(),
				ExpectResponseBodyMutation(map[string]any{
					"choices":   []any{map[string]any{"text": "Hello!"}},
					"guardrail": "applied",
				}),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx := context.Background()
			h := tc.createHarness(t, ctx)

			responses, err := integration.StreamedRequest(t, h.Client, tc.reqs, len(tc.wantResponses))
			require.NoError(t, err, "unexpected error during streamed request")
			require.Len(t, responses, len(tc.wantResponses))

			envoytest.SortSetHeadersInResponses(tc.wantResponses)
			envoytest.SortSetHeadersInResponses(responses)
			if diff := cmp.Diff(tc.wantResponses, responses, protocmp.Transform()); diff != "" {
				t.Errorf("Response mismatch (-want +got): %v", diff)
			}
		})
	}
}

// TestFullDuplexStreamed_BodyBasedRouting validates the "Streaming" behavior of BBR.
// This validates that BBR correctly buffers streamed chunks, inspects the body, and injects the header.
func TestFullDuplexStreamed_BodyBasedRouting(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name             string
		reqs             []*extProcPb.ProcessingRequest
		wantResponses    []*extProcPb.ProcessingResponse
		wantErr          bool
		wantStatusCode   envoyTypePb.StatusCode
		wantBodyContains string
	}{
		{
			name: "success: adds model header from simple body",
			reqs: integration.ReqLLM(logger, "test", "foo", "bar"),
			wantResponses: []*extProcPb.ProcessingResponse{
				ExpectBBRHeader("foo", "llama", "64"),
				ExpectBBRBodyPassThrough("test", "foo"),
			},
		},
		{
			name: "success: buffers split chunks and extracts model",
			reqs: integration.ReqRaw(
				map[string]string{"hi": "mom"},
				`{"max_tokens":100,"model":"sql-lo`,
				`ra-sheddable","prompt":"test","temperature":0}`,
			),
			wantResponses: []*extProcPb.ProcessingResponse{
				ExpectBBRHeader("sql-lora-sheddable", "llama", "79"),
				ExpectBBRBodyPassThrough("test", "sql-lora-sheddable"),
			},
		},
		{
			name: "missing model field - skips gracefully",
			reqs: integration.ReqLLM(logger, "test", "", ""),
			wantResponses: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestHeaders{
						RequestHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*envoyCorev3.HeaderValueOption{
										{
											Header: &envoyCorev3.HeaderValue{
												Key:      "Content-Length",
												RawValue: []byte("50"),
											},
										},
										{
											Header: &envoyCorev3.HeaderValue{
												Key: "X-Gateway-Base-Model-Name",
											},
										},
									},
								},
							},
						},
					},
				},
				ExpectBBRBodyPassThrough("test", ""),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx := context.Background()
			h := NewBBRHarness(t, ctx, true)

			expectedCount := len(tc.wantResponses)
			if tc.wantStatusCode != 0 || tc.wantErr {
				expectedCount = 1
			}

			responses, err := integration.StreamedRequest(t, h.Client, tc.reqs, expectedCount)

			if tc.wantErr {
				require.Error(t, err, "expected stream error")
				return
			}
			require.NoError(t, err, "unexpected stream error")

			if tc.wantStatusCode != 0 {
				require.Len(t, responses, 1)
				ir := responses[0].GetImmediateResponse()
				require.NotNil(t, ir, "expected ImmediateResponse")
				require.Equal(t, tc.wantStatusCode, ir.GetStatus().GetCode())
				if tc.wantBodyContains != "" {
					require.True(t, strings.Contains(string(ir.GetBody()), tc.wantBodyContains),
						"ImmediateResponse body %s should contain %s", string(ir.GetBody()), tc.wantBodyContains)
				}
				return
			}

			// sort headers in responses for deterministic tests
			envoytest.SortSetHeadersInResponses(tc.wantResponses)
			envoytest.SortSetHeadersInResponses(responses)
			if diff := cmp.Diff(tc.wantResponses, responses, protocmp.Transform()); diff != "" {
				t.Errorf("Response mismatch (-want +got): %v", diff)
			}
		})
	}
}

// testResponsePlugin implements framework.ResponseProcessor for integration tests.
type testResponsePlugin struct {
	name     string
	mutateFn func(ctx context.Context, response *framework.InferenceResponse) error
}

func (p *testResponsePlugin) TypedName() epp.TypedName {
	return epp.TypedName{Type: "test", Name: p.name}
}

func (p *testResponsePlugin) ProcessResponse(ctx context.Context, _ *framework.CycleState, response *framework.InferenceResponse) error {
	return p.mutateFn(ctx, response)
}

var _ framework.ResponseProcessor = &testResponsePlugin{}
