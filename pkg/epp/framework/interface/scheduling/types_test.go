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
					Prompt: "What is the meaning of life?",
				},
			},
			expected: "What is the meaning of life?",
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
					Input: map[string]interface{}{"key": "value"},
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
