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

package types

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestNewLLMResponseFromBytes(t *testing.T) {
	chatCompletionJSON := `{
		"choices": [
			{
				"message": {
					"role": "assistant",
					"content": "Hello!"
				},
				"finish_reason": "stop"
			}
		],
		"usage": {
			"prompt_tokens": 1,
			"completion_tokens": 2,
			"total_tokens": 3
		}
	}`

	legacyCompletionJSON := `{
		"choices": [
			{
				"text": "Hello there!",
				"finish_reason": "stop"
			}
		],
		"usage": {
			"prompt_tokens": 4,
			"completion_tokens": 5,
			"total_tokens": 9
		}
	}`

	chatCompletionEmptyChoicesJSON := `{
		"choices": [],
		"usage": {
			"prompt_tokens": 1,
			"completion_tokens": 2,
			"total_tokens": 3
		}
	}`

	legacyCompletionEmptyChoicesJSON := `{
		"choices": [],
		"usage": {
			"prompt_tokens": 4,
			"completion_tokens": 5,
			"total_tokens": 9
		}
	}`

	chatCompletionEmptyUsageJSON := `{
		"choices": [
			{
				"message": {
					"role": "assistant",
					"content": "Hello!"
				},
				"finish_reason": "stop"
			}
		]
	}`

	legacyCompletionEmptyUsageJSON := `{
		"choices": [
			{
				"text": "Hello there!",
				"finish_reason": "stop"
			}
		]
	}`

	invalidJSON := `{"invalid": json}`
	unstructuredJSON := `{"foo": "bar"}`

	testCases := []struct {
		name      string
		input     []byte
		want      *LLMResponse
		wantError bool
	}{
		{
			name:  "valid chat completion response",
			input: []byte(chatCompletionJSON),
			want: &LLMResponse{
				ChatCompletion: &ChatCompletionResponse{
					Choices: []ChatChoice{
						{
							Message: Message{
								Role: "assistant",
								Content: Content{
									Raw: "Hello!",
								},
							},
							FinishReason: "stop",
						},
					},
					Usage: &Usage{
						PromptTokens:     1,
						CompletionTokens: 2,
						TotalTokens:      3,
					},
				},
			},
			wantError: false,
		},
		{
			name:  "valid legacy completion response",
			input: []byte(legacyCompletionJSON),
			want: &LLMResponse{
				Completion: &CompletionResponse{
					Choices: []CompletionChoice{
						{
							Text:         "Hello there!",
							FinishReason: "stop",
						},
					},
					Usage: &Usage{
						PromptTokens:     4,
						CompletionTokens: 5,
						TotalTokens:      9,
					},
				},
			},
			wantError: false,
		},
		{
			name:      "invalid json",
			input:     []byte(invalidJSON),
			want:      nil,
			wantError: true,
		},
		{
			name:      "empty input",
			input:     []byte{},
			want:      nil,
			wantError: true,
		},
		{
			name:      "unstructured json",
			input:     []byte(unstructuredJSON),
			want:      nil,
			wantError: true,
		},
		{
			name:      "chat completion with empty choices",
			input:     []byte(chatCompletionEmptyChoicesJSON),
			want:      nil,
			wantError: true,
		},
		{
			name:      "legacy completion with empty choices",
			input:     []byte(legacyCompletionEmptyChoicesJSON),
			want:      nil,
			wantError: true,
		},
		{
			name:  "chat completion with empty usage",
			input: []byte(chatCompletionEmptyUsageJSON),
			want: &LLMResponse{
				ChatCompletion: &ChatCompletionResponse{
					Choices: []ChatChoice{
						{
							Message: Message{
								Role: "assistant",
								Content: Content{
									Raw: "Hello!",
								},
							},
							FinishReason: "stop",
						},
					},
				},
			},
			wantError: false,
		},
		{
			name:  "legacy completion with empty usage",
			input: []byte(legacyCompletionEmptyUsageJSON),
			want: &LLMResponse{
				Completion: &CompletionResponse{
					Choices: []CompletionChoice{
						{
							Text:         "Hello there!",
							FinishReason: "stop",
						},
					},
				},
			},
			wantError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := NewLLMResponseFromBytes(tc.input)

			if (err != nil) != tc.wantError {
				t.Errorf("NewLLMResponseFromBytes() error = %v, wantError %v", err, tc.wantError)
				return
			}

			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("NewLLMResponseFromBytes() (-want +got): %v", diff)
			}
		})
	}
}

func TestNewLLMResponseFromStream(t *testing.T) {
	testCases := []struct {
		name        string
		streamData  []byte
		want        *LLMResponse
		wantErr     bool
		errContains string
	}{
		{
			name: "valid chat stream with content and usage",
			streamData: []byte(`
			data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}}]} 

			data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"}}]} 

			data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"}}]} 

			data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]} 

			data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}} 

			data: [DONE]
			`),
			want: &LLMResponse{
				ChatCompletion: &ChatCompletionResponse{
					Choices: []ChatChoice{
						{
							Message: Message{
								Role:    "assistant",
								Content: Content{Raw: "Hello world"},
							},
							FinishReason: "stop",
						},
					},
					Usage: &Usage{
						PromptTokens:     5,
						CompletionTokens: 7,
						TotalTokens:      12,
					},
				},
			},
			wantErr: false,
		},
		{
			name: "valid completion stream with content and usage",
			streamData: []byte(`
			data: {"id":"cmpl-1","object":"text_completion","choices":[{"index":0,"text":"Hello"}]} 

			data: {"id":"cmpl-1","object":"text_completion","choices":[{"index":0,"text":" world"}]} 

			data: {"id":"cmpl-1","object":"text_completion","choices":[{"index":0,"text":"","finish_reason":"stop"}]} 

			data: {"id":"cmpl-1","object":"text_completion","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}} 

			data: [DONE]
			`),
			want: &LLMResponse{
				Completion: &CompletionResponse{
					Choices: []CompletionChoice{
						{
							Text:         "Hello world",
							FinishReason: "stop",
						},
					},
					Usage: &Usage{
						PromptTokens:     5,
						CompletionTokens: 7,
						TotalTokens:      12,
					},
				},
			},
		},
		{
			name:        "empty stream data",
			streamData:  []byte(""),
			wantErr:     true,
			errContains: "input bytes are empty",
		},
		{
			name:        "stream with no choices",
			streamData:  []byte(`data: [DONE]`),
			wantErr:     true,
			errContains: "failed to determine stream type",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := NewLLMResponseFromStream(tc.streamData)

			if tc.wantErr {
				if err == nil {
					t.Errorf("Expected an error, but got nil")
				}
				if err != nil && tc.errContains != "" && !strings.Contains(err.Error(), tc.errContains) {
					t.Errorf("Expected error to contain '%s', but got '%s'", tc.errContains, err.Error())
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if diff := cmp.Diff(tc.want, got); diff != "" {
					t.Errorf("NewLLMResponseFromStream() mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func TestFirstChoiceContent(t *testing.T) {
	testCases := []struct {
		name      string
		res       *LLMResponse
		want      []byte
		wantError bool
	}{
		{
			name: "chatCompletion with choice",
			res: &LLMResponse{
				ChatCompletion: &ChatCompletionResponse{
					Choices: []ChatChoice{
						{Message: Message{Role: "assistant",
							Content: Content{
								Raw: "Hello from Chat",
							}}},
					},
				},
			},
			want: []byte(`{"role":"assistant","content":"Hello from Chat"},`),
		},
		{
			name: "legacyCompletion with choice",
			res: &LLMResponse{
				Completion: &CompletionResponse{
					Choices: []CompletionChoice{
						{Text: "Hello from Legacy"},
					},
				},
			},
			want: []byte(`Hello from Legacy`),
		},
		{
			name: "chatCompletion with no choices",
			res: &LLMResponse{
				ChatCompletion: &ChatCompletionResponse{
					Choices: []ChatChoice{},
				},
			},
			wantError: true,
		},
		{
			name: "legacyCompletion with no choices",
			res: &LLMResponse{
				Completion: &CompletionResponse{
					Choices: []CompletionChoice{},
				},
			},
			wantError: true,
		},
		{
			name: "LLMResponse with all fields nil",
			res: &LLMResponse{
				ChatCompletion: nil,
				Completion:     nil,
			},
			wantError: true,
		},
		{
			name:      "Empty LLMResponse struct",
			res:       &LLMResponse{},
			wantError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.res.FirstChoiceContent()
			if tc.wantError != (err != nil) {
				t.Errorf("FirstChoiceContent() wantError is %v, but got error: %v", tc.wantError, err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("FirstChoiceContent() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestUsage_String(t *testing.T) {
	var nilUsage *Usage
	tests := []struct {
		name string
		u    *Usage
		want string
	}{
		{
			name: "nil usage",
			u:    nilUsage,
			want: nilString,
		},
		{
			name: "non-nil usage",
			u:    &Usage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3},
			want: "{Prompt: 1, Completion: 2, Total: 3}",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.u.String(); got != tt.want {
				t.Errorf("Usage.String() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestChatCompletionResponse_String(t *testing.T) {
	var nilResp *ChatCompletionResponse
	tests := []struct {
		name string
		r    *ChatCompletionResponse
		want string
	}{
		{
			name: "nil response",
			r:    nilResp,
			want: nilString,
		},
		{
			name: "response with no choices",
			r:    &ChatCompletionResponse{Choices: []ChatChoice{}, Usage: &Usage{}},
			want: "{ContentLength: 0, Usage: {Prompt: 0, Completion: 0, Total: 0}}",
		},
		{
			name: "response with choices",
			r: &ChatCompletionResponse{
				Choices: []ChatChoice{
					{Message: Message{Content: Content{Raw: "hello"}}},
				},
				Usage: &Usage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3},
			},
			want: "{ContentLength: 5, Usage: {Prompt: 1, Completion: 2, Total: 3}}",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.r.String(); got != tt.want {
				t.Errorf("ChatCompletionResponse.String() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLegacyCompletionResponse_String(t *testing.T) {
	var nilResp *CompletionResponse
	tests := []struct {
		name string
		r    *CompletionResponse
		want string
	}{
		{
			name: "nil response",
			r:    nilResp,
			want: nilString,
		},
		{
			name: "response with no choices",
			r:    &CompletionResponse{Choices: []CompletionChoice{}, Usage: &Usage{}},
			want: "{TextLength: 0, Usage: {Prompt: 0, Completion: 0, Total: 0}}",
		},
		{
			name: "response with choices",
			r: &CompletionResponse{
				Choices: []CompletionChoice{
					{Text: "hello world"},
				},
				Usage: &Usage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3},
			},
			want: "{TextLength: 11, Usage: {Prompt: 1, Completion: 2, Total: 3}}",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.r.String(); got != tt.want {
				t.Errorf("LegacyCompletionResponse.String() = %v, want %v", got, tt.want)
			}
		})
	}
}
