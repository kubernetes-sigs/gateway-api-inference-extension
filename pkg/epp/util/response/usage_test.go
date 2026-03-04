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

package response

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	fwkrc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
)

func TestExtractUsage(t *testing.T) {
	tests := []struct {
		name    string
		body    []byte
		want    *fwkrc.Usage
		wantErr bool
	}{
		{
			name: "Chat Completion (uses prompt_tokens)",
			body: []byte(`{
				"object": "chat.completion",
				"usage": {
					"prompt_tokens": 10,
					"completion_tokens": 20,
					"total_tokens": 30
				}
			}`),
			want: &fwkrc.Usage{
				PromptTokens:     10,
				CompletionTokens: 20,
				TotalTokens:      30,
			},
		},
		{
			name: "Conversations API (uses input_tokens)",
			body: []byte(`{
				"object": "conversation",
				"usage": {
					"input_tokens": 15,
					"output_tokens": 25,
					"total_tokens": 40
				}
			}`),
			want: &fwkrc.Usage{
				PromptTokens:     15,
				CompletionTokens: 25,
				TotalTokens:      40,
			},
		},
		{
			name: "Full Usage with Cached Token details",
			body: []byte(`{
				"object": "chat.completion",
				"usage": {
					"prompt_tokens": 100,
					"completion_tokens": 50,
					"total_tokens": 150,
					"prompt_token_details": {
						"cached_tokens": 40
					}
				}
			}`),
			want: &fwkrc.Usage{
				PromptTokens:     100,
				CompletionTokens: 50,
				TotalTokens:      150,
				PromptTokenDetails: &fwkrc.PromptTokenDetails{
					CachedTokens: 40,
				},
			},
		},
		{
			name: "Fallback logic (unknown object type)",
			body: []byte(`{
				"object": "unknown_type",
				"usage": {
					"input_tokens": 5,
					"completion_tokens": 5,
					"total_tokens": 10
				}
			}`),
			want: &fwkrc.Usage{
				PromptTokens:     5,
				CompletionTokens: 5,
				TotalTokens:      10,
			},
		},
		{
			name:    "Missing usage field returns error",
			body:    []byte(`{"object": "chat.completion"}`),
			wantErr: true,
		},
		{
			name:    "Invalid JSON returns error",
			body:    []byte(`{malformed`),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExtractUsage(tt.body)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ExtractUsage() error = %v, wantErr %v", err, tt.wantErr)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ExtractUsage() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestExtractUsageStreaming(t *testing.T) {
	tests := []struct {
		name     string
		response string
		want     ResponseBody
	}{
		{
			name:     "Single data chunk with usage",
			response: "data: {\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":5,\"total_tokens\":10}}\n",
			want: ResponseBody{
				Usage: &fwkrc.Usage{PromptTokens: 5, CompletionTokens: 5, TotalTokens: 10},
			},
		},
		{
			name:     "Usage and DONE in the same multi-line response",
			response: "data: {\"usage\":{\"prompt_tokens\":10,\"total_tokens\":10}}\ndata: [DONE]",
			want: ResponseBody{
				Usage: &fwkrc.Usage{PromptTokens: 10, TotalTokens: 10},
			},
		},
		{
			name:     "Chunk without usage (ignored)",
			response: "data: {\"choices\":[{\"text\":\"hi\"}]}",
			want:     ResponseBody{},
		},
		{
			name:     "Just DONE signal",
			response: "data: [DONE]",
			want:     ResponseBody{},
		},
		{
			name:     "Malformed JSON in stream (skipped)",
			response: "data: {bad-json}\ndata: {\"usage\":{\"total_tokens\":5}}",
			want: ResponseBody{
				Usage: &fwkrc.Usage{TotalTokens: 5},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ExtractUsageStreaming(tt.response)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ExtractUsageStreaming() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
