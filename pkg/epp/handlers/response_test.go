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

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	body = `
	{
		"id": "cmpl-573498d260f2423f9e42817bbba3743a",
		"object": "text_completion",
		"created": 1732563765,
		"model": "meta-llama/Llama-2-7b-hf",
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

	streamingBodyWithoutUsage = `data: {"id":"cmpl-41764c93-f9d2-4f31-be08-3ba04fa25394","object":"text_completion","created":1740002445,"model":"tweet-summary-0","choices":[],"usage":null}
	`

	streamingBodyWithUsage = `data: {"id":"cmpl-41764c93-f9d2-4f31-be08-3ba04fa25394","object":"text_completion","created":1740002445,"model":"tweet-summary-0","choices":[],"usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}
data: [DONE]
	`
)

func TestHandleResponseBody(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	tests := []struct {
		name    string
		req     *extProcPb.ProcessingRequest_ResponseBody
		reqCtx  *RequestContext
		want    Usage
		wantErr bool
	}{
		{
			name: "success",
			req: &extProcPb.ProcessingRequest_ResponseBody{
				ResponseBody: &extProcPb.HttpBody{
					Body: []byte(body),
				},
			},
			want: Usage{
				PromptTokens:     11,
				TotalTokens:      111,
				CompletionTokens: 100,
			},
		},
		{
			name: "malformed response",
			req: &extProcPb.ProcessingRequest_ResponseBody{
				ResponseBody: &extProcPb.HttpBody{
					Body: []byte("malformed json"),
				},
			},
			wantErr: true,
		},
		{
			name: "streaming request without usage",
			req: &extProcPb.ProcessingRequest_ResponseBody{
				ResponseBody: &extProcPb.HttpBody{
					Body: []byte(streamingBodyWithoutUsage),
				},
			},
			reqCtx: &RequestContext{
				modelServerStreaming: true,
			},
			wantErr: false,
			// In the middle of streaming response, so request context response is not set yet.
		},
		{
			name: "streaming request with usage",
			req: &extProcPb.ProcessingRequest_ResponseBody{
				ResponseBody: &extProcPb.HttpBody{
					Body: []byte(streamingBodyWithUsage),
				},
			},
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
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := &Server{}
			reqCtx := test.reqCtx
			if reqCtx == nil {
				reqCtx = &RequestContext{}
			}
			_, err := server.HandleResponseBody(ctx, reqCtx, &extProcPb.ProcessingRequest{Request: test.req})
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
