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
	"testing"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/testing/protocmp"

	envoytest "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/test"
	"sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

// TestBodyBasedRouting validates the "Unary" (Non-Streaming) behavior of BBR.
// This simulates scenarios where Envoy buffers the body before sending it to ext_proc.
func TestBodyBasedRouting(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		req          *extProcPb.ProcessingRequest
		wantResponse *extProcPb.ProcessingResponse
		wantErr      bool
	}{
		{
			name:         "success: extracts model and sets header",
			req:          integration.ReqLLMUnary(logger, "test", "llama"),
			wantResponse: ExpectBBRUnaryResponse("llama"),
			wantErr:      false,
		},
		{
			name:         "noop: no model parameter in body",
			req:          integration.ReqLLMUnary(logger, "test1", ""),
			wantResponse: ExpectBBRUnaryResponse(""), // Expect no headers.
			wantErr:      false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx := context.Background()
			h := NewBBRHarness(t, ctx, false)

			res, err := integration.SendRequest(t, h.Client, tc.req)

			if tc.wantErr {
				require.Error(t, err, "expected error during request processing")
			} else {
				require.NoError(t, err, "unexpected error during request processing")
			}

			// sort headers in responses for deterministic tests
			envoytest.SortSetHeadersInResponses([]*extProcPb.ProcessingResponse{tc.wantResponse})
			envoytest.SortSetHeadersInResponses([]*extProcPb.ProcessingResponse{res})
			if diff := cmp.Diff(tc.wantResponse, res, protocmp.Transform()); diff != "" {
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
		skipExactCompare bool
	}{
		{
			name: "success: adds model header from simple body",
			reqs: integration.ReqLLM(logger, "test", "foo", "bar"),
			wantResponses: []*extProcPb.ProcessingResponse{
				ExpectBBRNoOpHeader(),
				ExpectBBRHeader("foo"),
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
				ExpectBBRNoOpHeader(),
				ExpectBBRHeader("sql-lora-sheddable"),
				ExpectBBRBodyPassThrough("test", "sql-lora-sheddable"),
			},
		},
		{
			name: "noop: handles missing model field gracefully",
			reqs: integration.ReqLLM(logger, "test", "", ""),
			wantResponses: []*extProcPb.ProcessingResponse{
				ExpectBBRNoOpHeader(),
				ExpectBBRNoOpHeader(),
				ExpectBBRBodyPassThrough("test", ""),
			},
		},
		{
			name: "audio transcriptions: multipart form sets model header and passes body through",
			reqs: func() []*extProcPb.ProcessingRequest {
				headers, body := integration.BuildMultipartTranscriptionsRequest("whisper-1", "audio.mp3", []byte("test audio"))
				return integration.ReqRaw(headers, string(body))
			}(),
			wantResponses: func() []*extProcPb.ProcessingResponse {
				_, body := integration.BuildMultipartTranscriptionsRequest("whisper-1", "audio.mp3", []byte("test audio"))
				return []*extProcPb.ProcessingResponse{
					ExpectBBRNoOpHeader(),
					ExpectBBRHeader("whisper-1"),
					ExpectBBRBodyPassThroughRaw(body),
				}
			}(),
			skipExactCompare: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx := context.Background()
			h := NewBBRHarness(t, ctx, true)

			responses, err := integration.StreamedRequest(t, h.Client, tc.reqs, len(tc.wantResponses))

			if tc.wantErr {
				require.Error(t, err, "expected stream error")
			} else {
				require.NoError(t, err, "unexpected stream error")
			}

			// sort headers in responses for deterministic tests
			envoytest.SortSetHeadersInResponses(tc.wantResponses)
			envoytest.SortSetHeadersInResponses(responses)
			if tc.skipExactCompare {
				require.Len(t, responses, len(tc.wantResponses), "response count")
				var gotModelHeader string
				var gotBody []byte
				for _, r := range responses {
					if rh := r.GetRequestHeaders(); rh != nil && rh.Response != nil && rh.Response.HeaderMutation != nil {
						for _, h := range rh.Response.HeaderMutation.SetHeaders {
							if h.GetHeader().GetKey() == "X-Gateway-Model-Name" {
								gotModelHeader = string(h.GetHeader().GetRawValue())
								break
							}
						}
					}
					if rb := r.GetRequestBody(); rb != nil && rb.Response != nil && rb.Response.BodyMutation != nil {
						if sr := rb.Response.BodyMutation.GetStreamedResponse(); sr != nil {
							gotBody = sr.Body
						}
					}
				}
				require.Equal(t, "whisper-1", gotModelHeader, "X-Gateway-Model-Name")
				_, wantBody := integration.BuildMultipartTranscriptionsRequest("whisper-1", "audio.mp3", []byte("test audio"))
				require.Equal(t, wantBody, gotBody, "body pass-through")
			} else if diff := cmp.Diff(tc.wantResponses, responses, protocmp.Transform()); diff != "" {
				t.Errorf("Response mismatch (-want +got): %v", diff)
			}
		})
	}
}
