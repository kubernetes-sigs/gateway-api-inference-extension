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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
)

const (
	// StreamDone is the special string indicating the end of a streaming response.
	StreamDone = "[DONE]"
)

// LLMResponse is a structured representation of a parsed LLM response body.
// An LLMResponse must contain exactly one of ChatCompletion or LegacyCompletion.
type LLMResponse struct {
	// ChatCompletion is the representation of the OpenAI /vv1/chat/completions response body.
	ChatCompletion *ChatCompletionResponse `json:"chat_completion,omitempty"`
	// Completion is the representation of the OpenAI /v1/completions response body.
	Completion *CompletionResponse `json:"legacy_completion,omitempty"`
}

// FirstChoiceContent extracts the first choice of the response.
func (res *LLMResponse) FirstChoiceContent() ([]byte, error) {
	if res.ChatCompletion != nil && len(res.ChatCompletion.Choices) > 0 {
		return MarshalMessagesToJSON(res.ChatCompletion.Choices[0].Message)
	}
	if res.Completion != nil && len(res.Completion.Choices) > 0 {
		return []byte(res.Completion.Choices[0].Text), nil
	}
	return nil, errors.New("no choices found in the LLM response")
}

func (res *LLMResponse) Usage() *Usage {
	if res.ChatCompletion != nil {
		return res.ChatCompletion.Usage
	}
	return res.Completion.Usage
}

// ChatCompletionResponse represents the full response body for the chat completions API.
type ChatCompletionResponse struct {
	Choices []ChatChoice `json:"choices"`
	Usage   *Usage       `json:"usage,omitempty"`
}

func (r *ChatCompletionResponse) String() string {
	if r == nil {
		return nilString
	}
	contentLen := 0
	if len(r.Choices) > 0 {
		contentLen = len(r.Choices[0].Message.Content.Raw)
	}
	return fmt.Sprintf("{ContentLength: %d, Usage: %s}", contentLen, r.Usage)
}

// ChatChoice represents a single choice in the chat completion response.
type ChatChoice struct {
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// ChatMessage represents the message object within a choice.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CompletionResponse represents the full response body for the legacy completions API.
type CompletionResponse struct {
	Choices []CompletionChoice `json:"choices"`
	Usage   *Usage             `json:"usage,omitempty"`
}

func (r *CompletionResponse) String() string {
	if r == nil {
		return nilString
	}
	textLen := 0
	if len(r.Choices) > 0 {
		textLen = len(r.Choices[0].Text)
	}
	return fmt.Sprintf("{TextLength: %d, Usage: %v}", textLen, r.Usage)
}

// CompletionChoice represents a single choice in the legacy completion response.
type CompletionChoice struct {
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason"`
}

// Usage represents the token usage data common to all response formats.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (u *Usage) String() string {
	if u == nil {
		return nilString
	}
	return fmt.Sprintf("{Prompt: %d, Completion: %d, Total: %d}", u.PromptTokens, u.CompletionTokens, u.TotalTokens)
}

// ChatCompletionStreamChoiceDelta represents the delta in a streaming choice.
type ChatCompletionStreamChoiceDelta struct {
	Content string `json:"content,omitempty"`
	Role    string `json:"role,omitempty"`
}

// ChatCompletionStreamChoice represents a choice in a streaming response.
type ChatCompletionStreamChoice struct {
	Index        int                             `json:"index"`
	Delta        ChatCompletionStreamChoiceDelta `json:"delta"`
	FinishReason string                          `json:"finish_reason,omitempty"`
}

// ChatCompletionChunk represents a chunk of a streaming chat completion response.
type ChatCompletionChunk struct {
	Choices []ChatCompletionStreamChoice `json:"choices"`
	Usage   *Usage                       `json:"usage,omitempty"`
}

// CompletionStreamChoice represents a choice in a streaming completion response.
type CompletionStreamChoice struct {
	Text         string `json:"text"`
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// CompletionChunk represents a chunk of a streaming completion response.
type CompletionChunk struct {
	Choices []CompletionStreamChoice `json:"choices"`
	Usage   *Usage                   `json:"usage,omitempty"`
}

// NewLLMResponseFromStream initializes an LLMResponse from a streaming response.
func NewLLMResponseFromStream(body []byte) (*LLMResponse, error) {
	if len(body) == 0 {
		return nil, errors.New("input bytes are empty")
	}

	lines := bytes.Split(body, []byte("data: "))

	// Determine stream type from the first data chunk.
	for _, line := range lines {
		line = bytes.TrimSpace(line)

		jsonData := bytes.TrimPrefix(line, []byte("data: "))
		if len(jsonData) == 0 || string(jsonData) == StreamDone {
			continue
		}

		if bytes.Contains(jsonData, []byte(`"delta":`)) {
			return processChatStream(lines)
		}
		if bytes.Contains(jsonData, []byte(`"text":`)) {
			return processCompletionStream(lines)
		}
	}

	return nil, errors.New("failed to determine stream type or find choices")
}

func processChatStream(lines [][]byte) (*LLMResponse, error) {
	chatChoices := make(map[int]*ChatChoice)
	var chatUsage *Usage

	for _, line := range lines {
		line = bytes.TrimSpace(line)
		jsonData := bytes.TrimPrefix(line, []byte("data: "))
		if len(jsonData) == 0 || string(jsonData) == StreamDone {
			continue
		}

		var chunk ChatCompletionChunk
		if err := json.Unmarshal(jsonData, &chunk); err != nil {
			continue // Ignore malformed chunks
		}

		if chunk.Usage != nil {
			chatUsage = chunk.Usage
		}
		for _, choiceChunk := range chunk.Choices {
			if _, ok := chatChoices[choiceChunk.Index]; !ok {
				chatChoices[choiceChunk.Index] = &ChatChoice{Message: Message{}}
			}
			choice := chatChoices[choiceChunk.Index]
			choice.Message.Role += choiceChunk.Delta.Role
			choice.Message.Content.Raw += choiceChunk.Delta.Content
			if choiceChunk.FinishReason != "" {
				choice.FinishReason = choiceChunk.FinishReason
			}
		}
	}

	if len(chatChoices) == 0 && chatUsage == nil {
		return nil, errors.New("no choices or usage found in chat stream")
	}

	return aggregateChatStream(chatChoices, chatUsage), nil
}

func processCompletionStream(lines [][]byte) (*LLMResponse, error) {
	completionChoices := make(map[int]*CompletionChoice)
	var completionUsage *Usage

	for _, line := range lines {
		line = bytes.TrimSpace(line)
		jsonData := bytes.TrimPrefix(line, []byte("data: "))
		if len(jsonData) == 0 || string(jsonData) == StreamDone {
			continue
		}

		var chunk CompletionChunk
		if err := json.Unmarshal(jsonData, &chunk); err != nil {
			continue // Ignore malformed chunks
		}

		if chunk.Usage != nil {
			completionUsage = chunk.Usage
		}
		for _, choiceChunk := range chunk.Choices {
			if _, ok := completionChoices[choiceChunk.Index]; !ok {
				completionChoices[choiceChunk.Index] = &CompletionChoice{}
			}
			choice := completionChoices[choiceChunk.Index]
			choice.Text += choiceChunk.Text
			if choiceChunk.FinishReason != "" {
				choice.FinishReason = choiceChunk.FinishReason
			}
		}
	}

	if len(completionChoices) == 0 && completionUsage == nil {
		return nil, errors.New("no choices or usage found in completion stream")
	}

	return aggregateCompletionStream(completionChoices, completionUsage), nil
}

func aggregateChatStream(choices map[int]*ChatChoice, usage *Usage) *LLMResponse {
	resp := &ChatCompletionResponse{Usage: usage}
	keys := make([]int, 0, len(choices))
	for k := range choices {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	finalChoices := make([]ChatChoice, len(keys))
	for i, k := range keys {
		finalChoices[i] = *choices[k]
	}
	resp.Choices = finalChoices

	return &LLMResponse{ChatCompletion: resp}
}

func aggregateCompletionStream(choices map[int]*CompletionChoice, usage *Usage) *LLMResponse {
	resp := &CompletionResponse{Usage: usage}
	keys := make([]int, 0, len(choices))
	for k := range choices {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	finalChoices := make([]CompletionChoice, len(keys))
	for i, k := range keys {
		finalChoices[i] = *choices[k]
	}
	resp.Choices = finalChoices
	return &LLMResponse{Completion: resp}
}

// NewLLMResponseFromBytes initializes an LLMResponse by trying to parse the data
// as a chat completion and then as a legacy completion response.
func NewLLMResponseFromBytes(body []byte) (*LLMResponse, error) {
	if len(body) == 0 {
		return nil, errors.New("input bytes are empty")
	}

	// Attempt to unmarshal as a ChatCompletionResponse first.
	var chatResp ChatCompletionResponse
	if err := json.Unmarshal(body, &chatResp); err == nil {
		// Check if the role is set to distinguish ChatCompletion and LegacyCompletion.
		if len(chatResp.Choices) > 0 && chatResp.Choices[0].Message.Role != "" {
			return &LLMResponse{ChatCompletion: &chatResp}, nil
		}
	}

	// Try to unmarshal as a LegacyCompletionResponse.
	var legacyResp CompletionResponse
	if err := json.Unmarshal(body, &legacyResp); err == nil {
		if len(legacyResp.Choices) > 0 {
			return &LLMResponse{Completion: &legacyResp}, nil
		}
	}

	return nil, errors.New("failed to unmarshal body into any known LLM response format")
}
