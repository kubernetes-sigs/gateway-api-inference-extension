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

	"github.com/google/go-cmp/cmp"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

func TestExtractRequestData(t *testing.T) {
	tests := []struct {
		name    string
		body    map[string]any
		want    *types.LLMRequestBody
		wantErr bool
	}{
		{
			name: "completions request body",
			body: map[string]any{
				"model":  "test",
				"prompt": "test prompt",
			},
			want: &types.LLMRequestBody{
				Completions: &types.CompletionsRequest{
					Prompt: "test prompt",
				},
			},
		},
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
				},
			},
			want: &types.LLMRequestBody{
				ChatCompletions: &types.ChatCompletionsRequest{
					Messages: []types.Message{
						{Role: "system", Content: types.Content{Raw: "this is a system message"}},
						{Role: "user", Content: types.Content{Raw: "hello"}},
					},
				},
			},
		},
		{
			name: "chat completions request body with multi-modal content",
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
			want: &types.LLMRequestBody{
				ChatCompletions: &types.ChatCompletionsRequest{
					Messages: []types.Message{
						{Role: "system", Content: types.Content{
							Structured: []types.ContentBlock{
								{
									Text: "Describe this image in one sentence.",
									Type: "text",
								},
							},
						}},
						{Role: "user", Content: types.Content{
							Structured: []types.ContentBlock{
								{
									Type:     "image_url",
									ImageURL: types.ImageBlock{Url: "https://example.com/images/dui.jpg."},
								},
							},
						}},
					},
				},
			},
		},
		{
			name: "chat completions with all optional fields",
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
			want: &types.LLMRequestBody{
				ChatCompletions: &types.ChatCompletionsRequest{
					Messages:                  []types.Message{{Role: "user", Content: types.Content{Raw: "hello"}}},
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
			body:    nil,
			wantErr: true,
		},
		{
			name: "invalid prompt format",
			body: map[string]any{
				"model":  "test",
				"prompt": 123,
			},
			wantErr: true,
		},
		{
			name: "invalid messages format",
			body: map[string]any{
				"model":    "test",
				"messages": "invalid",
			},
			wantErr: true,
		},
		{
			name: "neither prompt nor messages",
			body: map[string]any{
				"model": "test",
			},
			wantErr: true,
		},
		{
			name: "empty messages array",
			body: map[string]any{
				"model":    "test",
				"messages": []any{},
			},
			wantErr: true,
		},
		{
			name: "message with non-string role",
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{"role": 123, "content": "hello"},
				},
			},
			wantErr: true,
		},
		{
			name: "message with non-string content",
			body: map[string]any{
				"model": "test",
				"messages": []any{
					map[string]any{"role": "user", "content": 123},
				},
			},
			wantErr: true,
		},
		{
			name: "invalid tools format",
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
			name: "invalid documents format",
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
			name: "invalid chat_template format",
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
			name: "invalid return_assistant_tokens_mask format",
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
			name: "invalid continue_final_message format",
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
			name: "invalid add_generation_prompt format",
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
			name: "invalid chat_template_kwargs format",
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
			name: "completions request with cache_salt",
			body: map[string]any{
				"model":      "test",
				"prompt":     "test prompt",
				"cache_salt": "Z3V2bmV3aGxza3ZubGFoZ3Zud3V3ZWZ2bmd0b3V2bnZmc2xpZ3RoZ2x2aQ==",
			},
			want: &types.LLMRequestBody{
				Completions: &types.CompletionsRequest{
					Prompt:    "test prompt",
					CacheSalt: "Z3V2bmV3aGxza3ZubGFoZ3Zud3V3ZWZ2bmd0b3V2bnZmc2xpZ3RoZ2x2aQ==",
				},
			},
		},
		{
			name: "chat completions request with cache_salt",
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
			want: &types.LLMRequestBody{
				ChatCompletions: &types.ChatCompletionsRequest{
					Messages: []types.Message{
						{Role: "system", Content: types.Content{Raw: "this is a system message"}},
						{Role: "user", Content: types.Content{Raw: "hello"}},
					},
					CacheSalt: "Z3V2bmV3aGxza3ZubGFoZ3Zud3V3ZWZ2bmd0b3V2bnZmc2xpZ3RoZ2x2aQ==",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExtractRequestBody(tt.body)
			if (err != nil) != tt.wantErr {
				t.Errorf("ExtractRequestBody() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}

			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ExtractRequestBody() mismatch (-want +got):\n%s", diff)
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

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ExtractRequestBody(body)
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

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ExtractRequestBody(body)
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

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ExtractRequestBody(body)
		if err != nil {
			b.Fatal(err)
		}
	}
}
