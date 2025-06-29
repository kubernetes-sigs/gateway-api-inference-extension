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

package request

import (
	"testing"
)

func TestExtractPromptFromRequestBody(t *testing.T) {
	tests := []struct {
		name    string
		body    map[string]any
		want    string
		wantErr bool
		errType error
	}{
		{
			name: "chat completions request body",
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{
						"role": "system", "content": "this is a system message",
					},
					map[string]any{
						"role": "user", "content": "hello",
					},
					map[string]any{
						"role": "assistant", "content": "hi, what can I do for you?",
					},
				},
			},
			want: "<|im_start|>system\nthis is a system message<|im_end|>\n" +
				"<|im_start|>user\nhello<|im_end|>\n" +
				"<|im_start|>assistant\nhi, what can I do for you?<|im_end|>\n",
		},
		{
			name: "completions request body",
			body: map[string]any{
				"model":  "test",
				"prompt": "test prompt",
			},
			want: "test prompt",
		},
		{
			name: "invalid prompt format",
			body: map[string]any{
				"model": "test",
				"prompt": []any{
					map[string]any{
						"role": "system", "content": "this is a system message",
					},
					map[string]any{
						"role": "user", "content": "hello",
					},
					map[string]any{
						"role": "assistant", "content": "hi, what can I",
					},
				},
			},
			wantErr: true,
		},
		{
			name: "invalid messaged format",
			body: map[string]any{
				"model": "test",
				"messages": map[string]any{
					"role": "system", "content": "this is a system message",
				},
			},
			wantErr: true,
		},
		{
			name: "prompt does not exist",
			body: map[string]any{
				"model": "test",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExtractPromptFromRequestBody(tt.body)
			if (err != nil) != tt.wantErr {
				t.Errorf("ExtractPromptFromRequestBody() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("ExtractPromptFromRequestBody() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestExtractPromptField(t *testing.T) {
	tests := []struct {
		name    string
		body    map[string]any
		want    string
		wantErr bool
	}{
		{
			name: "valid prompt",
			body: map[string]any{
				"prompt": "test prompt",
			},
			want: "test prompt",
		},
		{
			name:    "prompt not found",
			body:    map[string]any{},
			wantErr: true,
		},
		{
			name: "non-string prompt",
			body: map[string]any{
				"prompt": 123,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := extractPromptField(tt.body)
			if (err != nil) != tt.wantErr {
				t.Errorf("extractPromptField() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("extractPromptField() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestExtractPromptFromMessagesField(t *testing.T) {
	tests := []struct {
		name    string
		body    map[string]any
		want    string
		wantErr bool
	}{
		{
			name: "valid messages",
			body: map[string]any{
				"messages": []any{
					map[string]any{"role": "user", "content": "test1"},
					map[string]any{"role": "assistant", "content": "test2"},
				},
			},
			want: "<|im_start|>user\ntest1<|im_end|>\n<|im_start|>assistant\ntest2<|im_end|>\n",
		},
		{
			name: "invalid messages format",
			body: map[string]any{
				"messages": "invalid",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := extractPromptFromMessagesField(tt.body)
			if (err != nil) != tt.wantErr {
				t.Errorf("extractPromptFromMessagesField() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("extractPromptFromMessagesField() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestConstructChatMessage(t *testing.T) {
	tests := []struct {
		role    string
		content string
		want    string
	}{
		{"user", "hello", "<|im_start|>user\nhello<|im_end|>\n"},
		{"assistant", "hi", "<|im_start|>assistant\nhi<|im_end|>\n"},
	}

	for _, tt := range tests {
		if got := constructChatMessage(tt.role, tt.content); got != tt.want {
			t.Errorf("constructChatMessage() = %v, want %v", got, tt.want)
		}
	}
}
