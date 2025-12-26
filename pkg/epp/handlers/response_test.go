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
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	body = `
	{
		"id": "cmpl-573498d260f2423f9e42817bbba3743a",
		"object": "text_completion",
		"created": 1732563765,
		"model": "meta-llama/Llama-3.1-8B-Instruct",
		"choices": [
			{
				"index": 0,
				"text": " Chronicle\nThe San Francisco Chronicle has a new book review section, and it's a good one. The reviews are short, but they're well-written and well-informed. The Chronicle's book review section is a good place to start if you're looking for a good book review.\nThe Chronicle's book review section is a good place to start if you're looking for a good book review. The Chronicle's book review section",
				"logprobs": null,
				"finish_reason": "length",
				"stop_reason": null,
				"prompt_logprobs": null
			}
		],
		"usage": {
			"prompt_tokens": 11,
			"total_tokens": 111,
			"completion_tokens": 100
		}
	}
	`
	bodyWithCachedTokens = `
	{
		"id": "cmpl-573498d260f2423f9e42817bbba3743a",
		"object": "text_completion",
		"created": 1732563765,
		"model": "meta-llama/Llama-3.1-8B-Instruct",
		"choices": [
			{
				"index": 0,
				"text": " Chronicle\nThe San Francisco Chronicle has a new book review section, and it's a good one. The reviews are short, but they're well-written and well-informed. The Chronicle's book review section is a good place to start if you're looking for a good book review.\nThe Chronicle's book review section is a good place to start if you're looking for a good book review. The Chronicle's book review section",
				"logprobs": null,
				"finish_reason": "length",
				"stop_reason": null,
				"prompt_logprobs": null
			}
		],
		"usage": {
			"prompt_tokens": 11,
			"total_tokens": 111,
			"completion_tokens": 100,
			"prompt_token_details": {
				"cached_tokens": 10
			}
		}
	}
	`

	streamingBodyWithoutUsage = `data: {"id":"cmpl-41764c93-f9d2-4f31-be08-3ba04fa25394","object":"text_completion","created":1740002445,"model":"food-review-0","choices":[],"usage":null}
	`

	streamingBodyWithUsage = `data: {"id":"cmpl-41764c93-f9d2-4f31-be08-3ba04fa25394","object":"text_completion","created":1740002445,"model":"food-review-0","choices":[],"usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}
data: [DONE]
	`
	streamingBodyWithUsageAndCachedTokens = `data: {"id":"cmpl-41764c93-f9d2-4f31-be08-3ba04fa25394","object":"text_completion","created":1740002445,"model":"food-review-0","choices":[],"usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10,"prompt_token_details":{"cached_tokens":5}}}
data: [DONE]
	`
)

type mockDirector struct{}

func (m *mockDirector) HandleResponseBodyStreaming(ctx context.Context, reqCtx *RequestContext) (*RequestContext, error) {
	return reqCtx, nil
}
func (m *mockDirector) HandleResponseBodyComplete(ctx context.Context, reqCtx *RequestContext) (*RequestContext, error) {
	return reqCtx, nil
}
func (m *mockDirector) HandleResponseReceived(ctx context.Context, reqCtx *RequestContext) (*RequestContext, error) {
	return reqCtx, nil
}
func (m *mockDirector) HandlePreRequest(ctx context.Context, reqCtx *RequestContext) (*RequestContext, error) {
	return reqCtx, nil
}
func (m *mockDirector) GetRandomPod() *backend.Pod {
	return &backend.Pod{}
}
func (m *mockDirector) HandleRequest(ctx context.Context, reqCtx *RequestContext) (*RequestContext, error) {
	return reqCtx, nil
}

func TestHandleResponseBody(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	tests := []struct {
		name    string
		body    []byte
		reqCtx  *RequestContext
		want    Usage
		wantErr bool
	}{
		{
			name: "success",
			body: []byte(body),
			want: Usage{
				PromptTokens:     11,
				TotalTokens:      111,
				CompletionTokens: 100,
			},
		},
		{
			name: "success with cached tokens",
			body: []byte(bodyWithCachedTokens),
			want: Usage{
				PromptTokens:     11,
				TotalTokens:      111,
				CompletionTokens: 100,
				PromptTokenDetails: &PromptTokenDetails{
					CachedTokens: 10,
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := &StreamingServer{}
			server.director = &mockDirector{}
			reqCtx := test.reqCtx
			if reqCtx == nil {
				reqCtx = &RequestContext{}
			}
			var responseMap map[string]any
			marshalErr := json.Unmarshal(test.body, &responseMap)
			if marshalErr != nil {
				t.Error(marshalErr, "Error unmarshaling request body")
			}
			_, err := server.HandleResponseBody(ctx, reqCtx, responseMap)
			if err != nil {
				if !test.wantErr {
					t.Fatalf("HandleResponseBody returned unexpected error: %v, want %v", err, test.wantErr)
				}
				return
			}

			if diff := cmp.Diff(test.want, reqCtx.Usage); diff != "" {
				t.Errorf("HandleResponseBody returned unexpected response, diff(-want, +got): %v", diff)
			}
		})
	}
}

func TestHandleStreamedResponseBody(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	tests := []struct {
		name    string
		body    string
		reqCtx  *RequestContext
		want    Usage
		wantErr bool
	}{
		{
			name: "streaming request without usage",
			body: streamingBodyWithoutUsage,
			reqCtx: &RequestContext{
				modelServerStreaming: true,
			},
			wantErr: false,
			// In the middle of streaming response, so request context response is not set yet.
		},
		{
			name: "streaming request with usage",
			body: streamingBodyWithUsage,
			reqCtx: &RequestContext{
				modelServerStreaming: true,
			},
			wantErr: false,
			want: Usage{
				PromptTokens:     7,
				TotalTokens:      17,
				CompletionTokens: 10,
			},
		},
		{
			name: "streaming request with usage and cached tokens",
			body: streamingBodyWithUsageAndCachedTokens,
			reqCtx: &RequestContext{
				modelServerStreaming: true,
			},
			wantErr: false,
			want: Usage{
				PromptTokens:     7,
				TotalTokens:      17,
				CompletionTokens: 10,
				PromptTokenDetails: &PromptTokenDetails{
					CachedTokens: 5,
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := &StreamingServer{}
			server.director = &mockDirector{}
			reqCtx := test.reqCtx
			if reqCtx == nil {
				reqCtx = &RequestContext{}
			}
			server.HandleResponseBodyModelStreaming(ctx, reqCtx, test.body)

			if diff := cmp.Diff(test.want, reqCtx.Usage); diff != "" {
				t.Errorf("HandleResponseBody returned unexpected response, diff(-want, +got): %v", diff)
			}
		})
	}
}

func TestHandleResponseBodyModelStreaming_TokenAccumulation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		chunks    []string
		wantUsage Usage
	}{
		{
			name: "Standard: Usage and DONE in same chunk",
			chunks: []string{
				`data: {"usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}` + "\n" + `data: [DONE]`,
			},
			wantUsage: Usage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
		},
		{
			name: "Split: Usage in Chunk 1, DONE in Chunk 2",
			chunks: []string{
				// Chunk 1: Usage data arrives
				`data: {"usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}` + "\n",
				// Chunk 2: Stream termination. Should NOT overwrite the usage from Chunk 1.
				`data: [DONE]`,
			},
			wantUsage: Usage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
		},
		{
			name: "Fragmented: Content -> Usage -> DONE",
			chunks: []string{
				`data: {"choices":[{"text":"Hello"}]}` + "\n",
				`data: {"usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}` + "\n",
				`data: [DONE]`,
			},
			wantUsage: Usage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
		},
		{
			name: "No Usage Data",
			chunks: []string{
				`data: {"choices":[{"text":"Hello"}]}` + "\n",
				`data: [DONE]`,
			},
			wantUsage: Usage{}, // Zero values
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := &StreamingServer{
				director: &mockDirector{},
			}
			reqCtx := &RequestContext{}

			for _, chunk := range tc.chunks {
				server.HandleResponseBodyModelStreaming(context.Background(), reqCtx, chunk)
			}

			assert.Equal(t, tc.wantUsage, reqCtx.Usage, "Usage data should match expected accumulation")
			assert.True(t, reqCtx.ResponseComplete, "Response should be marked complete after [DONE]")
		})
	}
}

func TestGenerateResponseHeaders_Sanitization(t *testing.T) {
	server := &StreamingServer{}
	reqCtx := &RequestContext{
		Response: &Response{
			Headers: map[string]string{
				"x-backend-server":              "vllm-v0.6.3",            // should passthrough
				metadata.ObjectiveKey:           "sensitive-objective-id", // should be stripped
				metadata.DestinationEndpointKey: "10.2.0.5:8080",          // should be stripped
				"content-length":                "500",                    // hould be stripped
			},
		},
	}

	results := server.generateResponseHeaders(reqCtx)

	gotHeaders := make(map[string]string)
	for _, h := range results {
		gotHeaders[h.Header.Key] = string(h.Header.RawValue)
	}

	assert.Contains(t, gotHeaders, "x-backend-server")
	assert.Contains(t, gotHeaders, "x-went-into-resp-headers")
	assert.NotContains(t, gotHeaders, metadata.ObjectiveKey)
	assert.NotContains(t, gotHeaders, metadata.DestinationEndpointKey)
	assert.NotContains(t, gotHeaders, "content-length")
}
