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
package payloadprocess

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/payloadprocess"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkrc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestNewOpenAIParser(t *testing.T) {
	parser := NewOpenAIParser()

	expectedName := fwkplugin.TypedName{
		Type: payloadprocess.ParserType,
		Name: OpenAIParserName,
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
		body    []byte
		want    *scheduling.LLMRequestBody
		wantErr bool
	}{
		{
			name:    "Path: /v1/chat/completions",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body:    []byte(`{"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}`),
			want: &scheduling.LLMRequestBody{
				ParsedBody: map[string]any{
					"model": "gpt-4",
					"messages": []any{
						map[string]any{"role": "user", "content": "hello"},
					},
				},
				ChatCompletions: &scheduling.ChatCompletionsRequest{
					Messages: []scheduling.Message{
						{Role: "user", Content: scheduling.Content{Raw: "hello"}},
					},
				},
			},
		},
		{
			name:    "Path: /v1/completions",
			headers: map[string]string{":path": "/v1/completions"},
			body:    []byte(`{"model": "text-davinci-003", "prompt": "Once upon a time"}`),
			want: &scheduling.LLMRequestBody{
				ParsedBody: map[string]any{
					"model":  "text-davinci-003",
					"prompt": "Once upon a time",
				},
				Completions: &scheduling.CompletionsRequest{
					Prompt: "Once upon a time",
				},
			},
		},
		{
			name:    "Path: /v1/responses",
			headers: map[string]string{":path": "/v1/responses"},
			body:    []byte(`{"model": "gpt-4o", "input": "test input", "instructions": "be brief"}`),
			want: &scheduling.LLMRequestBody{
				ParsedBody: map[string]any{
					"model":        "gpt-4o",
					"input":        "test input",
					"instructions": "be brief",
				},
				Responses: &scheduling.ResponsesRequest{
					Input:        "test input",
					Instructions: "be brief",
				},
			},
		},
		{
			name:    "Path: /v1/conversations",
			headers: map[string]string{":path": "/v1/conversations"},
			body:    []byte(`{"model": "gpt-4o", "items": [{"type": "message", "role": "user", "content": "Hi"}]}`),
			want: &scheduling.LLMRequestBody{
				ParsedBody: map[string]any{
					"model": "gpt-4o",
					"items": []any{
						map[string]any{"type": "message", "role": "user", "content": "Hi"},
					},
				},
				Conversations: &scheduling.ConversationsRequest{
					Items: []scheduling.ConversationItem{
						{Type: "message", Role: "user", Content: "Hi"},
					},
				},
			},
		},
		{
			name:    "Error: Missing prompt in completions",
			headers: map[string]string{":path": "/v1/completions"},
			body:    []byte(`{"model": "text-davinci-003"}`),
			wantErr: true,
		},
		{
			name:    "Error: Invalid JSON syntax",
			headers: map[string]string{":path": "/v1/chat/completions"},
			body:    []byte(`{"model": "gpt-4", "messages": `), // Malformed
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parser.ParseRequest(tt.headers, tt.body)

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
		want    *payloadprocess.ParsedResponse
		wantErr bool
	}{
		{
			name: "Successful usage extraction",
			body: []byte(`{
				"usage": {
					"prompt_tokens": 10,
					"completion_tokens": 20,
					"total_tokens": 30
				}
			}`),
			want: &payloadprocess.ParsedResponse{
				Usage: &fwkrc.Usage{
					PromptTokens:     10,
					CompletionTokens: 20,
					TotalTokens:      30,
				},
			},
			wantErr: false,
		},
		{
			name: "Success with cached tokens",
			body: []byte(`{
				"usage": {
					"prompt_tokens": 15,
					"completion_tokens": 5,
					"total_tokens": 20,
					"prompt_token_details": { "cached_tokens": 10 }
				}
			}`),
			want: &payloadprocess.ParsedResponse{
				Usage: &fwkrc.Usage{
					PromptTokens:     15,
					CompletionTokens: 5,
					TotalTokens:      20,
					PromptTokenDetails: &fwkrc.PromptTokenDetails{
						CachedTokens: 10,
					},
				},
			},
			wantErr: false,
		},
		{
			name:    "Nil usage returns error",
			body:    []byte(`{"choices": []}`), // No usage field
			want:    nil,
			wantErr: true,
		},
		{
			name:    "Invalid JSON returns error",
			body:    []byte(`{invalid}`),
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parser.ParseResponse(tt.body)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseResponse() error = %v, wantErr %v", err, tt.wantErr)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ParseResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestOpenAIParser_ParseStreamResponse(t *testing.T) {
	parser := NewOpenAIParser()

	tests := []struct {
		name    string
		chunk   []byte
		want    *payloadprocess.ParsedResponse
		wantErr bool
	}{
		{
			name:  "Stream chunk with usage",
			chunk: []byte(`data: {"usage":{"prompt_tokens":7,"completion_tokens":10,"total_tokens":17}}`),
			want: &payloadprocess.ParsedResponse{
				Usage: &fwkrc.Usage{
					PromptTokens:     7,
					CompletionTokens: 10,
					TotalTokens:      17,
				},
			},
			wantErr: false,
		},
		{
			name:  "Stream chunk with cached tokens",
			chunk: []byte(`data: {"usage":{"prompt_tokens":10,"prompt_token_details":{"cached_tokens":5}}}`),
			want: &payloadprocess.ParsedResponse{
				Usage: &fwkrc.Usage{
					PromptTokens: 10,
					PromptTokenDetails: &fwkrc.PromptTokenDetails{
						CachedTokens: 5,
					},
				},
			},
			wantErr: false,
		},
		{
			name:    "Chunk without usage returns error",
			chunk:   []byte(`data: {"choices":[{"text":"hello"}]}`),
			want:    nil,
			wantErr: true,
		},
		{
			name:    "DONE message returns error",
			chunk:   []byte(`data: [DONE]`),
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parser.ParseStreamResponse(tt.chunk)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseStreamResponse() error = %v, wantErr %v", err, tt.wantErr)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ParseStreamResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
