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

package scheduling

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLLMRequestBody_PromptText(t *testing.T) {
	tests := []struct {
		name     string
		body     *LLMRequestBody
		expected string
	}{
		{
			name: "completions request returns prompt directly",
			body: &LLMRequestBody{
				Completions: &CompletionsRequest{
					Prompt: Prompt{Raw: "What is the meaning of life?"},
				},
			},
			expected: "What is the meaning of life?",
		},
		{
			name: "completions request with array of strings prompt",
			body: &LLMRequestBody{
				Completions: &CompletionsRequest{
					Prompt: Prompt{Strings: []string{"Why is", "the sky blue?"}},
				},
			},
			expected: "Why is the sky blue?",
		},
		{
			name: "chat completions with single raw message",
			body: &LLMRequestBody{
				ChatCompletions: &ChatCompletionsRequest{
					Messages: []Message{
						{Role: "user", Content: Content{Raw: "Hello, how are you?"}},
					},
				},
			},
			expected: "Hello, how are you? ",
		},
		{
			name: "chat completions with multiple messages",
			body: &LLMRequestBody{
				ChatCompletions: &ChatCompletionsRequest{
					Messages: []Message{
						{Role: "system", Content: Content{Raw: "You are a helpful assistant."}},
						{Role: "user", Content: Content{Raw: "Tell me a joke."}},
					},
				},
			},
			expected: "You are a helpful assistant. Tell me a joke. ",
		},
		{
			name: "chat completions with structured content blocks",
			body: &LLMRequestBody{
				ChatCompletions: &ChatCompletionsRequest{
					Messages: []Message{
						{
							Role: "user",
							Content: Content{
								Structured: []ContentBlock{
									{Type: "text", Text: "Describe this image:"},
									{Type: "image_url", ImageURL: ImageBlock{Url: "http://example.com/img.png"}},
								},
							},
						},
					},
				},
			},
			expected: "Describe this image:  ",
		},
		{
			name: "responses request with string input",
			body: &LLMRequestBody{
				Responses: &ResponsesRequest{
					Input: "Some response input",
				},
			},
			expected: "Some response input",
		},
		{
			name: "responses request with non-string input",
			body: &LLMRequestBody{
				Responses: &ResponsesRequest{
					Input: map[string]any{"key": "value"},
				},
			},
			expected: `{"key":"value"}`,
		},
		{
			name: "conversations request",
			body: &LLMRequestBody{
				Conversations: &ConversationsRequest{
					Items: []ConversationItem{
						{Type: "message", Role: "user", Content: "Hello"},
					},
				},
			},
			expected: `[{"type":"message","role":"user","content":"Hello"}]`,
		},
		{
			name:     "empty body returns empty string",
			body:     &LLMRequestBody{},
			expected: "",
		},
		{
			name: "chat completions with no messages",
			body: &LLMRequestBody{
				ChatCompletions: &ChatCompletionsRequest{
					Messages: []Message{},
				},
			},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.body.PromptText()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestPrompt_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    Prompt
		wantErr bool
	}{
		{
			name:  "string prompt",
			input: `"hello world"`,
			want:  Prompt{Raw: "hello world"},
		},
		{
			name:  "array of strings prompt",
			input: `["hello","world"]`,
			want:  Prompt{Strings: []string{"hello", "world"}},
		},
		{
			name:  "single-element array prompt",
			input: `["hello world"]`,
			want:  Prompt{Strings: []string{"hello world"}},
		},
		{
			name:    "integer prompt is rejected",
			input:   `123`,
			wantErr: true,
		},
		{
			name:    "object prompt is rejected",
			input:   `{"key":"value"}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var p Prompt
			err := p.UnmarshalJSON([]byte(tt.input))
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, p)
			}
		})
	}
}

func TestPrompt_PlainText(t *testing.T) {
	tests := []struct {
		name string
		p    Prompt
		want string
	}{
		{name: "raw string", p: Prompt{Raw: "hello"}, want: "hello"},
		{name: "strings joined", p: Prompt{Strings: []string{"a", "b", "c"}}, want: "a b c"},
		{name: "single string in array", p: Prompt{Strings: []string{"hello"}}, want: "hello"},
		{name: "zero value", p: Prompt{}, want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.p.PlainText())
		})
	}
}

func TestPrompt_IsEmpty(t *testing.T) {
	assert.True(t, Prompt{}.IsEmpty())
	assert.True(t, Prompt{Strings: []string{}}.IsEmpty())
	assert.False(t, Prompt{Raw: "x"}.IsEmpty())
	assert.False(t, Prompt{Strings: []string{"x"}}.IsEmpty())
}

func TestPrompt_MarshalJSON(t *testing.T) {
	raw, _ := Prompt{Raw: "hello"}.MarshalJSON()
	assert.Equal(t, `"hello"`, string(raw))

	arr, _ := Prompt{Strings: []string{"a", "b"}}.MarshalJSON()
	assert.Equal(t, `["a","b"]`, string(arr))

	empty, _ := Prompt{}.MarshalJSON()
	assert.Equal(t, `""`, string(empty))
}
