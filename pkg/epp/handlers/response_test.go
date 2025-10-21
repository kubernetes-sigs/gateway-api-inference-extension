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
	"testing"

	"github.com/google/go-cmp/cmp"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
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

	streamingBodyWithoutUsage = `
		    data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}}]} 

			data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"}}]} 

			data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"}}]} 

			data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]} 

			data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[],"usage":null} 

			data: [DONE]
	  		`

	streamingBodyWithUsage = `
			data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}}]} 

			data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"}}]} 

			data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"}}]} 

			data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]} 

			data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}} 

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
		want    *types.Usage
		wantErr bool
	}{
		{
			name: "success",
			body: []byte(body),
			want: &types.Usage{
				PromptTokens:     11,
				TotalTokens:      111,
				CompletionTokens: 100,
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
			_, err := server.HandleResponseBody(ctx, reqCtx, test.body)
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
		want    *types.Usage
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
			want: &types.Usage{
				PromptTokens:     5,
				TotalTokens:      12,
				CompletionTokens: 7,
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
			server.HandleResponseBodyModelStreaming(ctx, reqCtx, []byte(test.body))
			server.HandleResponseBodyModelStreamingComplete(ctx, reqCtx, []byte(test.body))

			if diff := cmp.Diff(test.want, reqCtx.Usage); diff != "" {
				t.Errorf("HandleResponseBody returned unexpected response, diff(-want, +got): %v", diff)
			}
		})
	}
}
