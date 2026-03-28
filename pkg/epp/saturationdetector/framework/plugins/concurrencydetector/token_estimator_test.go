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

package concurrencydetector

import (
	"testing"

	"github.com/stretchr/testify/require"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
)

func TestSimpleTokenEstimator_Estimate(t *testing.T) {
	estimator := NewSimpleTokenEstimator()

	testCases := []struct {
		name     string
		request  *fwkrh.LLMRequest
		expected int64
	}{
		{
			name:     "Nil request",
			request:  nil,
			expected: 0,
		},
		{
			name:     "Empty request",
			request:  &fwkrh.LLMRequest{},
			expected: 0,
		},
		{
			name: "Body nil",
			request: &fwkrh.LLMRequest{
				Body: nil,
			},
			expected: 0,
		},
		{
			name: "Less than 4 characters",
			request: &fwkrh.LLMRequest{
				Body: &fwkrh.RequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: "123",
					},
				},
			},
			expected: 3, // 3/4 (input tokens) + 3/4*1.5 (output tokens) = 3
		},
		{
			name: "Completions Request",
			request: &fwkrh.LLMRequest{
				Body: &fwkrh.RequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: "Hello, world!",
					},
				},
			},
			expected: 8, // 8/4 (input tokens) + 8/4*1.5 (output tokens) = 8
		},
		{
			name: "Completions with empty prompt",
			request: &fwkrh.LLMRequest{
				Body: &fwkrh.RequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: "",
					},
				},
			},
			expected: 3,
		},
		{
			name: "Completions with exactly 4 characters",
			request: &fwkrh.LLMRequest{
				Body: &fwkrh.RequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: "1234",
					},
				},
			},
			expected: 3,
		},
		{
			name: "Chat Completions Request with Structured content",
			request: &fwkrh.LLMRequest{
				Body: &fwkrh.RequestBody{
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
			request: &fwkrh.LLMRequest{
				Body: &fwkrh.RequestBody{
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
			request: &fwkrh.LLMRequest{
				Body: &fwkrh.RequestBody{
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
			expected: 5,
		},
		{
			name: "Chat Completions with empty messages",
			request: &fwkrh.LLMRequest{
				Body: &fwkrh.RequestBody{
					ChatCompletions: &fwkrh.ChatCompletionsRequest{
						Messages: []fwkrh.Message{},
					},
				},
			},
			expected: 3,
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
		request  *fwkrh.LLMRequest
		expected int64
	}{
		{
			name: "Empty prompt with custom config",
			request: &fwkrh.LLMRequest{
				Body: &fwkrh.RequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: "",
					},
				},
			},
			expected: 3,
		},
		{
			name: "4 chars with custom config",
			request: &fwkrh.LLMRequest{
				Body: &fwkrh.RequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: "1234",
					},
				},
			},
			expected: 6,
		},
		{
			name: "More than 4 chars with custom config",
			request: &fwkrh.LLMRequest{
				Body: &fwkrh.RequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: "This is a longer message.",
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
