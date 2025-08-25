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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

func TestExtractRequestData(t *testing.T) {
	tests := []struct {
		name    string
		body    map[string]any
		want    *types.LLMRequestData
		wantErr bool
		errType error
	}{
		{
			name: "completions request body",
			body: map[string]any{
				"model":  "test",
				"prompt": "test prompt",
			},
			want: &types.LLMRequestData{
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
			want: &types.LLMRequestData{
				ChatCompletions: &types.ChatCompletionsRequest{
					Messages: []types.Message{
						{Role: "system", Content: "this is a system message"},
						{Role: "user", Content: "hello"},
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
			want: &types.LLMRequestData{
				ChatCompletions: &types.ChatCompletionsRequest{
					Messages:                  []types.Message{{Role: "user", Content: "hello"}},
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExtractRequestData(tt.body)
			if (err != nil) != tt.wantErr {
				t.Errorf("ExtractRequestData() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}

			// Compare the results
			switch {
			case got.Completions != nil && tt.want.Completions != nil:
				if got.Completions.Prompt != tt.want.Completions.Prompt {
					t.Errorf("ExtractRequestData() Completions.Prompt = %v, want %v", got.Completions.Prompt, tt.want.Completions.Prompt)
				}
			case got.ChatCompletions != nil && tt.want.ChatCompletions != nil:
				// Compare messages
				if len(got.ChatCompletions.Messages) != len(tt.want.ChatCompletions.Messages) {
					t.Errorf("ExtractRequestData() Messages length = %v, want %v", len(got.ChatCompletions.Messages), len(tt.want.ChatCompletions.Messages))
				}
				for i, msg := range got.ChatCompletions.Messages {
					if i < len(tt.want.ChatCompletions.Messages) {
						wantMsg := tt.want.ChatCompletions.Messages[i]
						if msg.Role != wantMsg.Role || msg.Content != wantMsg.Content {
							t.Errorf("ExtractRequestData() Message[%d] = %v, want %v", i, msg, wantMsg)
						}
					}
				}
				// Compare other fields
				if got.ChatCompletions.ChatTemplate != tt.want.ChatCompletions.ChatTemplate {
					t.Errorf("ExtractRequestData() ChatTemplate = %v, want %v", got.ChatCompletions.ChatTemplate, tt.want.ChatCompletions.ChatTemplate)
				}
				if got.ChatCompletions.ReturnAssistantTokensMask != tt.want.ChatCompletions.ReturnAssistantTokensMask {
					t.Errorf("ExtractRequestData() ReturnAssistantTokensMask = %v, want %v", got.ChatCompletions.ReturnAssistantTokensMask, tt.want.ChatCompletions.ReturnAssistantTokensMask)
				}
			default:
				t.Errorf("ExtractRequestData() result type mismatch")
			}
		})
	}
}

func TestExtractPromptField(t *testing.T) {
	tests := []struct {
		name      string
		body      map[string]any
		wantValue string
		wantFound bool
		wantErr   bool
	}{
		{
			name: "valid prompt",
			body: map[string]any{
				"prompt": "test prompt",
			},
			wantValue: "test prompt",
			wantFound: true,
		},
		{
			name:      "prompt not found",
			body:      map[string]any{},
			wantFound: false,
			wantErr:   true, // extractPromptField returns found=true but with an error if prompt is missing
		},
		{
			name: "non-string prompt",
			body: map[string]any{
				"prompt": 123,
			},
			wantFound: true,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotValue, gotFound, err := extractPromptField(tt.body)
			if (err != nil) != tt.wantErr {
				t.Errorf("extractPromptField() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if gotValue != tt.wantValue {
				t.Errorf("extractPromptField() value = %v, want %v", gotValue, tt.wantValue)
			}
			if gotFound != tt.wantFound {
				t.Errorf("extractPromptField() found = %v, want %v", gotFound, tt.wantFound)
			}
		})
	}
}

func TestExtractMessagesField(t *testing.T) {
	tests := []struct {
		name    string
		body    map[string]any
		want    []types.Message
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
			want: []types.Message{
				{Role: "user", Content: "test1"},
				{Role: "assistant", Content: "test2"},
			},
		},
		{
			name:    "messages not found",
			body:    map[string]any{},
			wantErr: true,
		},
		{
			name: "invalid messages format",
			body: map[string]any{
				"messages": "invalid",
			},
			wantErr: true,
		},
		{
			name: "invalid message object",
			body: map[string]any{
				"messages": []any{"invalid"},
			},
			wantErr: true,
		},
		{
			name: "missing role",
			body: map[string]any{
				"messages": []any{
					map[string]any{"content": "test"},
				},
			},
			wantErr: true,
		},
		{
			name: "missing content",
			body: map[string]any{
				"messages": []any{
					map[string]any{"role": "user"},
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := extractMessagesField(tt.body)
			if (err != nil) != tt.wantErr {
				t.Errorf("extractMessagesField() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}
			if len(got) != len(tt.want) {
				t.Errorf("extractMessagesField() length = %v, want %v", len(got), len(tt.want))
				return
			}
			for i, msg := range got {
				if msg.Role != tt.want[i].Role || msg.Content != tt.want[i].Content {
					t.Errorf("extractMessagesField() message[%d] = %v, want %v", i, msg, tt.want[i])
				}
			}
		})
	}
}
