/*
Copyright 2026 The Kubernetes Authors.

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

package inflightload

import (
	"testing"

	"github.com/stretchr/testify/require"

	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestSimpleTokenEstimator_Estimate(t *testing.T) {
	estimator := NewSimpleTokenEstimator()

	testCases := []struct {
		name     string
		request  *framework.InferenceRequest
		expected int64
	}{
		{
			name:     "Nil request",
			request:  nil,
			expected: 0,
		},
		{
			name:     "Empty request",
			request:  &framework.InferenceRequest{},
			expected: 0,
		},
		{
			name: "Body nil",
			request: &framework.InferenceRequest{
				Body: nil,
			},
			expected: 0,
		},
		{
			name: "Less than 4 characters",
			request: &framework.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{Raw: "123"},
					},
				},
			},
			expected: 3, // 3/4 (input tokens) + 3/4*1.5 (output tokens) = 3
		},
		{
			name: "Completions Request",
			request: &framework.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{Raw: "Hello, world!"},
					},
				},
			},
			expected: 8, // 8/4 (input tokens) + 8/4*1.5 (output tokens) = 8
		},
		{
			name: "Completions with empty prompt",
			request: &framework.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{},
					},
				},
			},
			expected: 3,
		},
		{
			name: "Completions with exactly 4 characters",
			request: &framework.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{Raw: "1234"},
					},
				},
			},
			expected: 3,
		},
		{
			name: "Chat Completions Request with Structured content",
			request: &framework.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					ChatCompletions: &fwkrh.ChatCompletionsRequest{
						Messages: []fwkrh.Message{
							{
								Role: "user",
								Content: fwkrh.Content{
									Structured: []fwkrh.ContentBlock{
										{
											Type: "text",
											Text: "This is a longer message.",
										},
									},
								},
							},
						},
					},
				},
			},
			expected: 18,
		},
		{
			name: "Chat Completions with Raw content",
			request: &framework.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					ChatCompletions: &fwkrh.ChatCompletionsRequest{
						Messages: []fwkrh.Message{
							{
								Role: "user",
								Content: fwkrh.Content{
									Raw: "This is raw content.",
								},
							},
						},
					},
				},
			},
			expected: 13,
		},
		{
			name: "Chat Completions with multiple messages",
			request: &framework.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					ChatCompletions: &fwkrh.ChatCompletionsRequest{
						Messages: []fwkrh.Message{
							{
								Role: "user",
								Content: fwkrh.Content{
									Structured: []fwkrh.ContentBlock{
										{Type: "text", Text: "Hi"},
									},
								},
							},
							{
								Role: "assistant",
								Content: fwkrh.Content{
									Structured: []fwkrh.ContentBlock{
										{Type: "text", Text: "Hello"},
									},
								},
							},
						},
					},
				},
			},
			// PromptText() joins messages with a trailing space separator ("Hi Hello " = 9 chars),
			// so the estimate is higher than summing per-message lengths individually (7 chars).
			expected: 8,
		},
		{
			name: "Chat Completions with empty messages",
			request: &framework.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					ChatCompletions: &fwkrh.ChatCompletionsRequest{
						Messages: []fwkrh.Message{},
					},
				},
			},
			expected: 3,
		},
		{
			name: "Responses API with string input",
			request: &framework.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					Responses: &fwkrh.ResponsesRequest{
						Input: "Tell me a story about a brave knight.",
					},
				},
			},
			expected: 23,
		},
		{
			name: "Responses API with structured input",
			request: &framework.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					Responses: &fwkrh.ResponsesRequest{
						Input: []any{
							map[string]any{"role": "user", "content": "Hello"},
						},
					},
				},
			},
			expected: 23,
		},
		{
			name: "Conversations API",
			request: &framework.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					Conversations: &fwkrh.ConversationsRequest{
						Items: []fwkrh.ConversationItem{
							{Type: "message", Role: "user", Content: "Hi there"},
						},
					},
				},
			},
			expected: 35,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := estimator.Estimate(tc.request)
			require.Equal(t, tc.expected, actual)
		})
	}
}

func TestSimpleTokenEstimator_Estimate_CustomConfig(t *testing.T) {
	estimator := &SimpleTokenEstimator{
		CharactersPerToken: 2.0,
		OutputRatio:        2.0,
	}

	testCases := []struct {
		name     string
		request  *framework.InferenceRequest
		expected int64
	}{
		{
			name: "Empty prompt with custom config",
			request: &framework.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{},
					},
				},
			},
			expected: 3,
		},
		{
			name: "4 chars with custom config",
			request: &framework.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{Raw: "1234"},
					},
				},
			},
			expected: 6,
		},
		{
			name: "More than 4 chars with custom config",
			request: &framework.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{Raw: "This is a longer message."},
					},
				},
			},
			expected: 39,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := estimator.Estimate(tc.request)
			require.Equal(t, tc.expected, actual)
		})
	}
}
