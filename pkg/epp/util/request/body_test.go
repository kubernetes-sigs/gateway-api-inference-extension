package request

import (
	"testing"
)

func TestExtractPromptFromRequestBody(t *testing.T) {
	tests := []struct {
		name    string
		body    map[string]interface{}
		want    string
		wantErr bool
		errType error
	}{
		{
			name: "chat completions request body",
			body: map[string]interface{}{
				"model": "test",
				"messages": []interface{}{
					map[string]interface{}{
						"role": "system", "content": "this is a system message",
					},
					map[string]interface{}{
						"role": "user", "content": "hello",
					},
					map[string]interface{}{
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
			body: map[string]interface{}{
				"model":  "test",
				"prompt": "test prompt",
			},
			want: "test prompt",
		},
		{
			name: "invalid prompt format",
			body: map[string]interface{}{
				"model": "test",
				"prompt": []interface{}{
					map[string]interface{}{
						"role": "system", "content": "this is a system message",
					},
					map[string]interface{}{
						"role": "user", "content": "hello",
					},
					map[string]interface{}{
						"role": "assistant", "content": "hi, what can I",
					},
				},
			},
			wantErr: true,
		},
		{
			name: "invalid messaged format",
			body: map[string]interface{}{
				"model": "test",
				"messages": map[string]interface{}{
					"role": "system", "content": "this is a system message",
				},
			},
			wantErr: true,
		},
		{
			name: "prompt does not exist",
			body: map[string]interface{}{
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
		body    map[string]interface{}
		want    string
		wantErr bool
	}{
		{
			name: "valid prompt",
			body: map[string]interface{}{
				"prompt": "test prompt",
			},
			want: "test prompt",
		},
		{
			name:    "prompt not found",
			body:    map[string]interface{}{},
			wantErr: true,
		},
		{
			name: "non-string prompt",
			body: map[string]interface{}{
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
		body    map[string]interface{}
		want    string
		wantErr bool
	}{
		{
			name: "valid messages",
			body: map[string]interface{}{
				"messages": []interface{}{
					map[string]interface{}{"role": "user", "content": "test1"},
					map[string]interface{}{"role": "assistant", "content": "test2"},
				},
			},
			want: "<|im_start|>user\ntest1<|im_end|>\n<|im_start|>assistant\ntest2<|im_end|>\n",
		},
		{
			name: "invalid messages format",
			body: map[string]interface{}{
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
