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

package requesthandling

import (
	"testing"

	"github.com/stretchr/testify/assert"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
)

func TestLLMRequestBody_PromptText(t *testing.T) {
	tests := []struct {
		name     string
		body     *fwkrh.InferenceRequestBody
		expected string
	}{
		{
			name: "completions request returns prompt directly",
			body: &fwkrh.InferenceRequestBody{
				Completions: &fwkrh.CompletionsRequest{
					Prompt: fwkrh.Prompt{Raw: "What is the meaning of life?"},
				},
			},
			expected: "What is the meaning of life?",
		},
		{
			name: "completions request with array of strings prompt",
			body: &fwkrh.InferenceRequestBody{
				Completions: &fwkrh.CompletionsRequest{
					Prompt: fwkrh.Prompt{Strings: []string{"Why is", "the sky blue?"}},
				},
			},
			expected: "Why is the sky blue?",
		},
		{
			name: "chat completions with single raw message",
			body: &fwkrh.InferenceRequestBody{
				ChatCompletions: &fwkrh.ChatCompletionsRequest{
					Messages: []fwkrh.Message{
						{Role: "user", Content: fwkrh.Content{Raw: "Hello, how are you?"}},
					},
				},
			},
			expected: "Hello, how are you? ",
		},
		{
			name: "chat completions with multiple messages",
			body: &fwkrh.InferenceRequestBody{
				ChatCompletions: &fwkrh.ChatCompletionsRequest{
					Messages: []fwkrh.Message{
						{Role: "system", Content: fwkrh.Content{Raw: "You are a helpful assistant."}},
						{Role: "user", Content: fwkrh.Content{Raw: "Tell me a joke."}},
					},
				},
			},
			expected: "You are a helpful assistant. Tell me a joke. ",
		},
		{
			name: "chat completions with structured content blocks",
			body: &fwkrh.InferenceRequestBody{
				ChatCompletions: &fwkrh.ChatCompletionsRequest{
					Messages: []fwkrh.Message{
						{
							Role: "user",
							Content: fwkrh.Content{
								Structured: []fwkrh.ContentBlock{
									{Type: "text", Text: "Describe this image:"},
									{Type: "image_url", ImageURL: fwkrh.ImageBlock{Url: "http://example.com/img.png"}},
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
			body: &fwkrh.InferenceRequestBody{
				Responses: &fwkrh.ResponsesRequest{
					Input: "Some response input",
				},
			},
			expected: "Some response input",
		},
		{
			name: "responses request with non-string input",
			body: &fwkrh.InferenceRequestBody{
				Responses: &fwkrh.ResponsesRequest{
					Input: map[string]any{"key": "value"},
				},
			},
			expected: `{"key":"value"}`,
		},
		{
			name: "conversations request",
			body: &fwkrh.InferenceRequestBody{
				Conversations: &fwkrh.ConversationsRequest{
					Items: []fwkrh.ConversationItem{
						{Type: "message", Role: "user", Content: "Hello"},
					},
				},
			},
			expected: `[{"type":"message","role":"user","content":"Hello"}]`,
		},
		{
			name:     "empty body returns empty string",
			body:     &fwkrh.InferenceRequestBody{},
			expected: "",
		},
		{
			name: "chat completions with no messages",
			body: &fwkrh.InferenceRequestBody{
				ChatCompletions: &fwkrh.ChatCompletionsRequest{
					Messages: []fwkrh.Message{},
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
		want    fwkrh.Prompt
		wantErr bool
	}{
		{
			name:  "string prompt",
			input: `"hello world"`,
			want:  fwkrh.Prompt{Raw: "hello world"},
		},
		{
			name:  "array of strings prompt",
			input: `["hello","world"]`,
			want:  fwkrh.Prompt{Strings: []string{"hello", "world"}},
		},
		{
			name:  "single-element array prompt",
			input: `["hello world"]`,
			want:  fwkrh.Prompt{Strings: []string{"hello world"}},
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
			var p fwkrh.Prompt
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
		p    fwkrh.Prompt
		want string
	}{
		{name: "raw string", p: fwkrh.Prompt{Raw: "hello"}, want: "hello"},
		{name: "strings joined", p: fwkrh.Prompt{Strings: []string{"a", "b", "c"}}, want: "a b c"},
		{name: "single string in array", p: fwkrh.Prompt{Strings: []string{"hello"}}, want: "hello"},
		{name: "zero value", p: fwkrh.Prompt{}, want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.p.PlainText())
		})
	}
}

func TestPrompt_IsEmpty(t *testing.T) {
	assert.True(t, fwkrh.Prompt{}.IsEmpty())
	assert.True(t, fwkrh.Prompt{Strings: []string{}}.IsEmpty())
	assert.False(t, fwkrh.Prompt{Raw: "x"}.IsEmpty())
	assert.False(t, fwkrh.Prompt{Strings: []string{"x"}}.IsEmpty())
}

func TestPrompt_MarshalJSON(t *testing.T) {
	raw, _ := fwkrh.Prompt{Raw: "hello"}.MarshalJSON()
	assert.Equal(t, `"hello"`, string(raw))

	arr, _ := fwkrh.Prompt{Strings: []string{"a", "b"}}.MarshalJSON()
	assert.Equal(t, `["a","b"]`, string(arr))

	empty, _ := fwkrh.Prompt{}.MarshalJSON()
	assert.Equal(t, `""`, string(empty))
}
