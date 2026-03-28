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

package openai

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
)

func TestNewOpenAIParser(t *testing.T) {
	parser := NewOpenAIParser()

	expectedName := fwkplugin.TypedName{
		Type: OpenAIParserType,
		Name: OpenAIParserType,
	}

	if diff := cmp.Diff(expectedName, parser.TypedName()); diff != "" {
		t.Errorf("TypedName() mismatch (-want +got):\n%s", diff)
	}
}

func TestOpenAIParser_ParseRequest(t *testing.T) {
	parser := NewOpenAIParser()

	tests := []struct {
		name    string
		headers map[string]string
		body    map[string]any
		want    *fwkrh.RequestBody
		wantErr bool
	}{
		{
			name:    "completions request body",
			headers: map[string]string{":path": "/v1/completions"},
			body: map[string]any{
				"model":  "test",
				"prompt": "test prompt",
			},
			want: &fwkrh.RequestBody{
				Completions: &fwkrh.CompletionsRequest{
					Prompt: "test prompt",
				},
			},
		},
		{
			name:    "chat completions request body",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{
						"role": "system", "content": "this is a system message",
					},
					map[string]any{
						"role": "user", "content": "hello",
					},
				},
			},
			want: &fwkrh.RequestBody{
				ChatCompletions: &fwkrh.ChatCompletionsRequest{
					Messages: []fwkrh.Message{
						{Role: "system", Content: fwkrh.Content{Raw: "this is a system message"}},
						{Role: "user", Content: fwkrh.Content{Raw: "hello"}},
					},
				},
			},
		},
		{
			name:    "chat completions request body with multi-modal content",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{
						"role": "system",
						"content": []map[string]any{
							{
								"type": "text",
								"text": "Describe this image in one sentence.",
							},
						},
					},
					map[string]any{
						"role": "user",
						"content": []map[string]any{
							{
								"type": "image_url",
								"image_url": map[string]any{
									"url": "https://example.com/images/dui.jpg.",
								},
							},
						},
					},
				},
			},
			want: &fwkrh.RequestBody{
				ChatCompletions: &fwkrh.ChatCompletionsRequest{
					Messages: []fwkrh.Message{
						{Role: "system", Content: fwkrh.Content{
							Structured: []fwkrh.ContentBlock{
								{
									Text: "Describe this image in one sentence.",
									Type: "text",
								},
							},
						}},
						{Role: "user", Content: fwkrh.Content{
							Structured: []fwkrh.ContentBlock{
								{
									Type:     "image_url",
									ImageURL: fwkrh.ImageBlock{Url: "https://example.com/images/dui.jpg."},
								},
							},
						}},
					},
				},
			},
		},
		{
			name:    "chat completions request body with audio and video content",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{
						"role": "user",
						"content": []map[string]any{
							{
								"type": "input_audio",
								"input_audio": map[string]any{
									"data":   "base64data",
									"format": "wav",
								},
							},
							{
								"type": "video_url",
								"video_url": map[string]any{
									"url": "https://example.com/video.mp4",
								},
							},
						},
					},
				},
			},
			want: &fwkrh.RequestBody{
				ChatCompletions: &fwkrh.ChatCompletionsRequest{
					Messages: []fwkrh.Message{
						{Role: "user", Content: fwkrh.Content{
							Structured: []fwkrh.ContentBlock{
								{
									Type:       "input_audio",
									InputAudio: fwkrh.AudioBlock{Data: "base64data", Format: "wav"},
								},
								{
									Type:     "video_url",
									VideoURL: fwkrh.VideoBlock{Url: "https://example.com/video.mp4"},
								},
							},
						}},
					},
				},
			},
		},
		{
			name:    "chat completions with all optional fields",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{"role": "user", "content": "hello"},
				},
				"tools":                        []any{map[string]any{"type": "function"}},
				"documents":                    []any{map[string]any{"content": "doc"}},
				"chat_template":                "custom template",
				"return_assistant_tokens_mask": true,
				"continue_final_message":       true,
				"add_generation_prompt":        true,
				"chat_template_kwargs":         map[string]any{"key": "value"},
			},
			want: &fwkrh.RequestBody{
				ChatCompletions: &fwkrh.ChatCompletionsRequest{
					Messages:                  []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Raw: "hello"}}},
					Tools:                     []any{map[string]any{"type": "function"}},
					Documents:                 []any{map[string]any{"content": "doc"}},
					ChatTemplate:              "custom template",
					ReturnAssistantTokensMask: true,
					ContinueFinalMessage:      true,
					AddGenerationPrompt:       true,
					ChatTemplateKWArgs:        map[string]any{"key": "value"},
				},
			},
		},
		{
			name:    "nil body",
			headers: map[string]string{":path": "/v1/completions"},
			body:    nil,
			wantErr: true,
		},
		{
			name:    "invalid prompt format",
			headers: map[string]string{":path": "/v1/completions"},
			body: map[string]any{
				"model":  "test",
				"prompt": 123,
			},
			wantErr: true,
		},
		{
			name:    "invalid messages format",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model":    "test",
				"messages": "invalid",
			},
			wantErr: true,
		},
		{
			name:    "neither prompt nor messages",
			headers: map[string]string{":path": "/v1/completions"},
			body: map[string]any{
				"model": "test",
			},
			wantErr: true,
		},
		{
			name:    "empty messages array",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model":    "test",
				"messages": []any{},
			},
			wantErr: true,
		},
		{
			name:    "message with non-string role",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{"role": 123, "content": "hello"},
				},
			},
			wantErr: true,
		},
		{
			name:    "message with non-string content",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{"role": "user", "content": 123},
				},
			},
			wantErr: true,
		},
		{
			name:    "invalid tools format",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{"role": "user", "content": "hello"},
				},
				"tools": "invalid",
			},
			wantErr: true,
		},
		{
			name:    "invalid documents format",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{"role": "user", "content": "hello"},
				},
				"documents": "invalid",
			},
			wantErr: true,
		},
		{
			name:    "invalid chat_template format",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{"role": "user", "content": "hello"},
				},
				"chat_template": 123,
			},
			wantErr: true,
		},
		{
			name:    "invalid return_assistant_tokens_mask format",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{"role": "user", "content": "hello"},
				},
				"return_assistant_tokens_mask": "invalid",
			},
			wantErr: true,
		},
		{
			name:    "invalid continue_final_message format",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{"role": "user", "content": "hello"},
				},
				"continue_final_message": "invalid",
			},
			wantErr: true,
		},
		{
			name:    "invalid add_generation_prompt format",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{"role": "user", "content": "hello"},
				},
				"add_generation_prompt": "invalid",
			},
			wantErr: true,
		},
		{
			name:    "invalid chat_template_kwargs format",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{"role": "user", "content": "hello"},
				},
				"chat_template_kwargs": "invalid",
			},
			wantErr: true,
		},
		{
			name:    "completions request with cache_salt",
			headers: map[string]string{":path": "/v1/completions"},
			body: map[string]any{
				"model":      "test",
				"prompt":     "test prompt",
				"cache_salt": "Z3V2bmV3aGxza3ZubGFoZ3Zud3V3ZWZ2bmd0b3V2bnZmc2xpZ3RoZ2x2aQ==",
			},
			want: &fwkrh.RequestBody{
				Completions: &fwkrh.CompletionsRequest{
					Prompt:    "test prompt",
					CacheSalt: "Z3V2bmV3aGxza3ZubGFoZ3Zud3V3ZWZ2bmd0b3V2bnZmc2xpZ3RoZ2x2aQ==",
				},
			},
		},
		{
			name:    "chat completions request with cache_salt",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{
						"role": "system", "content": "this is a system message",
					},
					map[string]any{
						"role": "user", "content": "hello",
					},
				},
				"cache_salt": "Z3V2bmV3aGxza3ZubGFoZ3Zud3V3ZWZ2bmd0b3V2bnZmc2xpZ3RoZ2x2aQ==",
			},
			want: &fwkrh.RequestBody{
				ChatCompletions: &fwkrh.ChatCompletionsRequest{
					Messages: []fwkrh.Message{
						{Role: "system", Content: fwkrh.Content{Raw: "this is a system message"}},
						{Role: "user", Content: fwkrh.Content{Raw: "hello"}},
					},
					CacheSalt: "Z3V2bmV3aGxza3ZubGFoZ3Zud3V3ZWZ2bmd0b3V2bnZmc2xpZ3RoZ2x2aQ==",
				},
			},
		},
		{
			name:    "responses request body",
			headers: map[string]string{":path": "/v1/responses"},
			body: map[string]any{
				"model":        "gpt-4o",
				"input":        "How do I check if a Python object is an instance of a class?",
				"instructions": "You are a coding assistant that talks like a pirate.",
			},
			want: &fwkrh.RequestBody{
				Responses: &fwkrh.ResponsesRequest{
					Input:        "How do I check if a Python object is an instance of a class?",
					Instructions: "You are a coding assistant that talks like a pirate.",
				},
			},
		},
		{
			name:    "responses request with cache_salt",
			headers: map[string]string{":path": "/v1/responses"},
			body: map[string]any{
				"model":      "gpt-4o",
				"input":      "test input",
				"cache_salt": "abc123",
			},
			want: &fwkrh.RequestBody{
				Responses: &fwkrh.ResponsesRequest{
					Input:     "test input",
					CacheSalt: "abc123",
				},
			},
		},
		{
			name:    "responses request missing input",
			headers: map[string]string{":path": "/v1/responses"},
			body: map[string]any{
				"model":        "gpt-4o",
				"instructions": "test instructions",
			},
			wantErr: true,
		},
		// Path-based detection tests
		{
			name:    "conversations API via path",
			headers: map[string]string{":path": "/v1/conversations"},
			body: map[string]any{
				"model": "gpt-4o",
				"items": []map[string]interface{}{
					{"type": "message", "role": "user", "content": "Hello"},
				},
			},
			want: &fwkrh.RequestBody{
				Conversations: &fwkrh.ConversationsRequest{
					Items: []fwkrh.ConversationItem{
						{Type: "message", Role: "user", Content: "Hello"},
					},
				},
			},
		},
		{
			name:    "path from x-original-path header",
			headers: map[string]string{"x-original-path": "/v1/conversations"},
			body: map[string]any{
				"model": "gpt-4o",
				"items": []map[string]interface{}{
					{"type": "message", "role": "user", "content": "Hello"},
				},
			},
			want: &fwkrh.RequestBody{
				Conversations: &fwkrh.ConversationsRequest{
					Items: []fwkrh.ConversationItem{
						{Type: "message", Role: "user", Content: "Hello"},
					},
				},
			},
		},
		{
			name:    "defaults to completions API when no path header",
			headers: map[string]string{},
			body: map[string]any{
				"model":  "gpt-4o",
				"prompt": "test prompt",
			},
			want: &fwkrh.RequestBody{
				Completions: &fwkrh.CompletionsRequest{
					Prompt: "test prompt",
				},
			},
		},
		{
			name:    "chat completions request body with stream",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{"role": "user", "content": "hello"},
				},
				"stream": true,
			},
			want: &fwkrh.RequestBody{
				ChatCompletions: &fwkrh.ChatCompletionsRequest{
					Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Raw: "hello"}}},
				},
				ParsedBody: map[string]any{
					"model": "test",
					"messages": []any{
						map[string]any{"role": "user", "content": "hello"},
					},
					"stream": true,
				},
				Stream: true,
			},
		},
		// Embeddings API tests
		{
			name:    "embeddings request body with string input",
			headers: map[string]string{":path": "/v1/embeddings"},
			body: map[string]any{
				"model": "text-embedding-3-small",
				"input": "The food was delicious and the waiter...",
			},
			want: &fwkrh.RequestBody{
				Embeddings: &fwkrh.EmbeddingsRequest{
					Input: "The food was delicious and the waiter...",
				},
				ParsedBody: map[string]any{
					"model": "text-embedding-3-small",
					"input": "The food was delicious and the waiter...",
				},
			},
		},
		{
			name:    "embeddings request body with array input",
			headers: map[string]string{":path": "/v1/embeddings"},
			body: map[string]any{
				"model": "text-embedding-3-small",
				"input": []any{"First document", "Second document"},
			},
			want: &fwkrh.RequestBody{
				Embeddings: &fwkrh.EmbeddingsRequest{
					Input: []any{"First document", "Second document"},
				},
				ParsedBody: map[string]any{
					"model": "text-embedding-3-small",
					"input": []any{"First document", "Second document"},
				},
			},
		},
		{
			name:    "embeddings request with cache_salt",
			headers: map[string]string{":path": "/v1/embeddings"},
			body: map[string]any{
				"model":      "text-embedding-3-small",
				"input":      "embed this text",
				"cache_salt": "embeddings-salt-123",
			},
			want: &fwkrh.RequestBody{
				Embeddings: &fwkrh.EmbeddingsRequest{
					Input:     "embed this text",
					CacheSalt: "embeddings-salt-123",
				},
				ParsedBody: map[string]any{
					"model":      "text-embedding-3-small",
					"input":      "embed this text",
					"cache_salt": "embeddings-salt-123",
				},
			},
		},
		{
			name:    "embeddings API via x-original-path header",
			headers: map[string]string{"x-original-path": "/v1/embeddings"},
			body: map[string]any{
				"model": "text-embedding-3-small",
				"input": "text to embed",
			},
			want: &fwkrh.RequestBody{
				Embeddings: &fwkrh.EmbeddingsRequest{
					Input: "text to embed",
				},
				ParsedBody: map[string]any{
					"model": "text-embedding-3-small",
					"input": "text to embed",
				},
			},
		},
		{
			name:    "embeddings request missing input",
			headers: map[string]string{":path": "/v1/embeddings"},
			body: map[string]any{
				"model": "text-embedding-3-small",
			},
			wantErr: true,
		},
		{
			name:    "embeddings request with null input",
			headers: map[string]string{":path": "/v1/embeddings"},
			body: map[string]any{
				"model": "text-embedding-3-small",
				"input": nil,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bodyBytes, err := json.Marshal(tt.body)
			if err != nil {
				t.Fatalf("Invalid tt.body %v: cannot convert to bytes", tt.body)
			}
			got, err := parser.ParseRequest(context.Background(), bodyBytes, tt.headers)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseRequest() error = %v, wantErr %v", err, tt.wantErr)
			}

			if tt.wantErr {
				return
			}

			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ParseRequest() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestOpenAIParser_ParseResponse(t *testing.T) {
	parser := NewOpenAIParser()

	tests := []struct {
		name    string
		body    []byte
		want    *fwkrh.ParsedResponse
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
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     10,
					CompletionTokens: 20,
					TotalTokens:      30,
				},
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
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     15,
					CompletionTokens: 25,
					TotalTokens:      40,
				},
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
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     100,
					CompletionTokens: 50,
					TotalTokens:      150,
					PromptTokenDetails: &fwkrh.PromptTokenDetails{
						CachedTokens: 40,
					},
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
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     5,
					CompletionTokens: 5,
					TotalTokens:      10,
				},
			},
		},
		{
			name: "Missing usage field returns error",
			body: []byte(`{"object": "chat.completion"}`),
			want: &fwkrh.ParsedResponse{
				Usage: nil,
			},
		},
		{
			name:    "Invalid JSON returns error",
			body:    []byte(`{malformed`),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parser.ParseResponse(context.Background(), tt.body, map[string]string{}, false)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseResponse() error = %v, wantErr %v", err, tt.wantErr)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ParseResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestOpenAIParser_ParseResponse_Streaming(t *testing.T) {
	parser := NewOpenAIParser()

	tests := []struct {
		name  string
		chunk []byte
		want  *fwkrh.ParsedResponse
	}{
		{
			name:  "Single data chunk with usage",
			chunk: []byte("data: {\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":10,\"total_tokens\":17}}\n"),
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     7,
					CompletionTokens: 10,
					TotalTokens:      17,
				},
			},
		},
		{
			name:  "Usage and DONE in the same multi-line response",
			chunk: []byte("data: {\"usage\":{\"prompt_tokens\":10,\"prompt_token_details\":{\"cached_tokens\":10}}}\ndata: [DONE]"),
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens: 10,
					PromptTokenDetails: &fwkrh.PromptTokenDetails{
						CachedTokens: 10,
					},
				},
			},
		},
		{
			name:  "Chunk without usage returns ParsedResponse with nil usage",
			chunk: []byte(`data: {"choices":[{"text":"hello"}]}`),
			want: &fwkrh.ParsedResponse{
				Usage: nil,
			},
		},
		{
			name:  "DONE message returns error",
			chunk: []byte(`data: [DONE]`),
			want: &fwkrh.ParsedResponse{
				Usage: nil,
			},
		},
		{
			name:  "Malformed JSON in stream (skipped)",
			chunk: []byte(`data: {bad-json}\ndata: {\"usage\":{\"total_tokens\":5}}`),
			want: &fwkrh.ParsedResponse{
				Usage: nil,
			},
		},
		{
			name:  "ResponsesAPI streaming with full response",
			chunk: []byte("event: response.completed\ndata: {\"response\":{\"id\":\"resp_8e38bd02b4f56572\",\"model\":\"Qwen/Qwen3-32B\",\"object\":\"response\",\"usage\":{\"input_tokens\":31,\"input_tokens_details\":{\"cached_tokens\":16},\"output_tokens\":3,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":34}},\"type\":\"response.completed\"}"),
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     31,
					CompletionTokens: 3,
					TotalTokens:      34,
				},
			},
		},
		{
			name:  "ResponsesAPI without response.completed type returns nil",
			chunk: []byte("event: response.in_progress\ndata: {\"response\":{\"usage\":{\"input_tokens\":31,\"output_tokens\":3}},\"type\":\"response.in_progress\"}"),
			want: &fwkrh.ParsedResponse{
				Usage: nil,
			},
		},
		{
			name:  "ResponsesAPI with multiple events extracts from completed",
			chunk: []byte("event: response.output_text.delta\ndata: {\"delta\":\"Hello\",\"type\":\"response.output_text.delta\"}\n\nevent: response.completed\ndata: {\"response\":{\"usage\":{\"input_tokens\":39,\"output_tokens\":10,\"total_tokens\":49}},\"type\":\"response.completed\"}"),
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     39,
					CompletionTokens: 10,
					TotalTokens:      49,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parser.ParseResponse(context.Background(), tt.chunk, map[string]string{contentType: eventStreamType}, true)
			if err != nil {
				t.Fatalf("ParseStreamResponse() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ParseStreamResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// Benchmark tests for performance comparison
func BenchmarkExtractRequestData_Completions(b *testing.B) {
	body := map[string]any{
		"model":  "test",
		"prompt": "test prompt",
	}
	headers := map[string]string{":path": "/v1/completions"}
	parser := NewOpenAIParser()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonBytes, err := json.Marshal(body)
		if err != nil {
			b.Errorf("body cannot be marshalled to JSON bytes")
		}
		_, err = parser.ParseRequest(context.Background(), jsonBytes, headers)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkExtractRequestData_ChatCompletions(b *testing.B) {
	body := map[string]any{
		"model": "test",
		"messages": []any{
			map[string]any{"role": "user", "content": "hello"},
		},
	}
	headers := map[string]string{":path": "/v1/chat/completions"}
	parser := NewOpenAIParser()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonBytes, err := json.Marshal(body)
		if err != nil {
			b.Errorf("body cannot be marshalled to JSON bytes")
		}
		_, err = parser.ParseRequest(context.Background(), jsonBytes, headers)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkExtractRequestData_ChatCompletionsWithOptionals(b *testing.B) {
	body := map[string]any{
		"model": "test",
		"messages": []any{
			map[string]any{"role": "user", "content": "hello"},
		},
		"tools":                        []any{map[string]any{"type": "function"}},
		"documents":                    []any{map[string]any{"content": "doc"}},
		"chat_template":                "custom template",
		"return_assistant_tokens_mask": true,
		"continue_final_message":       true,
		"add_generation_prompt":        true,
		"chat_template_kwargs":         map[string]any{"key": "value"},
	}
	headers := map[string]string{":path": "/v1/chat/completions"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonBytes, err := json.Marshal(body)
		if err != nil {
			b.Errorf("body cannot be marshalled to JSON bytes")
		}
		_, err = extractRequestBody(jsonBytes, headers)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkExtractRequestData_Responses(b *testing.B) {
	body := map[string]any{
		"model":        "gpt-4o",
		"input":        "How do I check if a Python object is an instance of a class?",
		"instructions": "You are a coding assistant that talks like a pirate.",
	}
	headers := map[string]string{":path": "/v1/responses"}
	parser := NewOpenAIParser()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonBytes, err := json.Marshal(body)
		if err != nil {
			b.Errorf("body cannot be marshalled to JSON bytes")
		}
		_, err = parser.ParseRequest(context.Background(), jsonBytes, headers)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkExtractRequestData_Conversations(b *testing.B) {
	body := map[string]any{
		"model": "gpt-4o",
		"items": []map[string]interface{}{
			{"type": "message", "role": "user", "content": "Hello"},
		},
	}
	headers := map[string]string{":path": "/v1/conversations"}
	parser := NewOpenAIParser()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonBytes, err := json.Marshal(body)
		if err != nil {
			b.Errorf("body cannot be marshalled to JSON bytes")
		}
		_, err = parser.ParseRequest(context.Background(), jsonBytes, headers)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkExtractRequestData_Embeddings(b *testing.B) {
	body := map[string]any{
		"model": "text-embedding-3-small",
		"input": "The food was delicious and the waiter...",
	}
	headers := map[string]string{":path": "/v1/embeddings"}
	jsonBytes, err := json.Marshal(body)
	if err != nil {
		b.Fatal(err)
	}
	parser := NewOpenAIParser()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err = parser.ParseRequest(context.Background(), jsonBytes, headers)
		if err != nil {
			b.Fatal(err)
		}
	}
}
