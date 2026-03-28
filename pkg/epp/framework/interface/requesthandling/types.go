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
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const nilString = "<nil>"

// RequestBody contains the request-body fields that we parse out as user input,
// to be used in forming scheduling decisions.
// An RequestBody must contain exactly one of CompletionsRequest, ChatCompletionsRequest, ResponsesRequest, or ConversationsRequest.
type RequestBody struct {
	// CompletionsRequest is the representation of the OpenAI /v1/completions request body.
	Completions *CompletionsRequest `json:"completions,omitempty"`
	// ChatCompletionsRequest is the representation of the OpenAI /v1/chat/completions request body.
	ChatCompletions *ChatCompletionsRequest `json:"chat_completions,omitempty"`
	// ResponsesRequest is the representation of the OpenAI /v1/responses request body.
	Responses *ResponsesRequest `json:"responses,omitempty"`
	// ConversationsRequest is the representation of the OpenAI /v1/conversations request body.
	Conversations *ConversationsRequest `json:"conversations,omitempty"`
	// EmbeddingsRequest is the representation of the OpenAI /v1/embeddings request body.
	Embeddings *EmbeddingsRequest `json:"embeddings,omitempty"`
	// ParsedBody contains the unmarshaled request payload.
	// Note: Because this handles multiple protocols, this field is strictly expected
	// to be either a map[string]any (for HTTP/JSON) or a proto.Message (for gRPC).
	ParsedBody any `json:"-"`

	// Stream indicates whether the request specifies a streaming response (e.g., via a stream field).
	// This typically implies the model server's response will be streamed.
	Stream bool `json:"-"`
}

func (r *RequestBody) CacheSalt() string {
	if r.Conversations != nil {
		return r.Conversations.CacheSalt
	}
	if r.Responses != nil {
		return r.Responses.CacheSalt
	}
	if r.ChatCompletions != nil {
		return r.ChatCompletions.CacheSalt
	}
	if r.Completions != nil {
		return r.Completions.CacheSalt
	}
	return ""
}

// PromptText returns the aggregated prompt or message content of the request.
// It simplifies logic for extracting text from Completions vs ChatCompletions Requests.
func (r *RequestBody) PromptText() string {
	if r == nil {
		return ""
	}
	if r.Completions != nil {
		return r.Completions.Prompt
	}
	if r.ChatCompletions != nil {
		var contentBuilder strings.Builder
		for idx, msg := range r.ChatCompletions.Messages {
			contentBuilder.WriteString(msg.Content.PlainText())
			if idx < len(r.ChatCompletions.Messages)-1 {
				contentBuilder.WriteString(" ")
			}
		}
		return contentBuilder.String()
	}
	return ""
}

// CompletionsRequest is a structured representation of the fields we parse out of the /v1/completions request
// body. For detailed body fields, please refer to https://platform.openai.com/docs/api-reference/completions.
// This struct includes fields usable for plugins and scheduling decisions - and not the entire
// API spec.
type CompletionsRequest struct {
	// Prompt is the prompt that was sent in the request body.
	Prompt string `json:"prompt,omitempty"`
	// CacheSalt is an optional request parameter to isolate prefix caches for security reasons.
	CacheSalt string `json:"cache_salt,omitempty"`
}

func (r *CompletionsRequest) String() string {
	if r == nil {
		return nilString
	}

	return fmt.Sprintf("{PromptLength: %d}", len(r.Prompt))
}

// ChatCompletionsRequest is a structured representation of the fields we parse out of the v1/chat/completions
// request body. For detailed body fields, please refer to https://platform.openai.com/docs/api-reference/chat.
// This struct includes fields usable for plugins and scheduling decisions - and not the entire
// API spec.
type ChatCompletionsRequest struct {
	/* parameters from the official OpenAI chat-completions API */
	Messages []Message     `json:"messages,omitempty"`
	Tools    []interface{} `json:"tools,omitempty"`
	/* parameters from the HuggingFace transformers chat-templates API */
	Documents                 []interface{}          `json:"documents,omitempty"`
	ChatTemplate              string                 `json:"chat_template,omitempty"`
	ReturnAssistantTokensMask bool                   `json:"return_assistant_tokens_mask,omitempty"`
	ContinueFinalMessage      bool                   `json:"continue_final_message,omitempty"`
	AddGenerationPrompt       bool                   `json:"add_generation_prompt,omitempty"`
	ChatTemplateKWArgs        map[string]interface{} `json:"chat_template_kwargs,omitempty"`
	// CacheSalt is an optional request parameter to isolate prefix caches for security reasons.
	CacheSalt string `json:"cache_salt,omitempty"`
}

func (r *ChatCompletionsRequest) String() string {
	if r == nil {
		return nilString
	}

	messagesLen := 0
	for _, msg := range r.Messages {
		messagesLen += len(msg.Content.PlainText())
	}
	return fmt.Sprintf("{MessagesLength: %d}", messagesLen)
}

// ResponsesRequest represents the OpenAI /v1/responses request body structure
type ResponsesRequest struct {
	// Input can be either a string or an array of conversation items
	Input interface{} `json:"input,omitempty"`
	// Instructions provides optional system-level guidance
	Instructions interface{} `json:"instructions,omitempty"`
	// Tools field for function calling capabilities
	Tools interface{} `json:"tools,omitempty"`
	// CacheSalt isolates prefix caches for security
	CacheSalt string `json:"cache_salt,omitempty"`
}

func (r *ResponsesRequest) String() string {
	if r == nil {
		return nilString
	}
	return fmt.Sprintf("{InputType: %T, InstructionsType: %T}", r.Input, r.Instructions)
}

// ConversationsRequest represents the OpenAI /v1/conversations request body structure
type ConversationsRequest struct {
	// Items is the array of conversation items (messages, files, etc.)
	Items []ConversationItem `json:"items,omitempty"`
	// Metadata provides additional context for the conversation
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	// CacheSalt isolates prefix caches for security
	CacheSalt string `json:"cache_salt,omitempty"`
}

func (c *ConversationsRequest) String() string {
	if c == nil {
		return nilString
	}
	return fmt.Sprintf("{ItemsCount: %d}", len(c.Items))
}

// EmbeddingsRequest represents the OpenAI /v1/embeddings request body structure.
// Input can be a string or array of strings; see https://platform.openai.com/docs/api-reference/embeddings.
type EmbeddingsRequest struct {
	// Input is the text to embed (string or array of strings).
	Input interface{} `json:"input,omitempty"`
	// CacheSalt is an optional request parameter to isolate prefix caches for security reasons.
	CacheSalt string `json:"cache_salt,omitempty"`
}

func (e *EmbeddingsRequest) String() string {
	if e == nil {
		return nilString
	}
	return fmt.Sprintf("{InputType: %T}", e.Input)
}

// ConversationItem represents a single item in a conversation
type ConversationItem struct {
	// Type specifies the item type (message, file, etc.)
	Type string `json:"type,omitempty"`
	// Role specifies the role (user, assistant, system)
	Role string `json:"role,omitempty"`
	// Content contains the item content
	Content interface{} `json:"content,omitempty"`
}

// Message represents a single message in a chat-completions request.
type Message struct {
	// Role is the message Role, optional values are 'user', 'assistant', ...
	Role string `json:"role,omitempty"`
	// Content defines text of this message
	Content Content `json:"content,omitempty"`
}

type Content struct {
	Raw        string
	Structured []ContentBlock
}

type ContentBlock struct {
	Type       string     `json:"type"`
	Text       string     `json:"text,omitempty"`
	ImageURL   ImageBlock `json:"image_url,omitempty"`
	InputAudio AudioBlock `json:"input_audio,omitempty"`
	VideoURL   VideoBlock `json:"video_url,omitempty"`
}

type ImageBlock struct {
	Url string `json:"url,omitempty"`
}

type AudioBlock struct {
	Data   string `json:"data,omitempty"`
	Format string `json:"format,omitempty"`
}

type VideoBlock struct {
	Url string `json:"url,omitempty"`
}

// UnmarshalJSON allow use both format
func (mc *Content) UnmarshalJSON(data []byte) error {
	// Raw format
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		mc.Raw = str
		return nil
	}

	// Block format
	var blocks []ContentBlock
	if err := json.Unmarshal(data, &blocks); err == nil {
		mc.Structured = blocks
		return nil
	}

	return errors.New("content format not supported")
}

func (mc Content) MarshalJSON() ([]byte, error) {
	if mc.Raw != "" {
		return json.Marshal(mc.Raw)
	}
	if mc.Structured != nil {
		return json.Marshal(mc.Structured)
	}
	return json.Marshal("")
}

func (mc Content) PlainText() string {
	if mc.Raw != "" {
		return mc.Raw
	}
	var sb strings.Builder
	for _, block := range mc.Structured {
		if block.Type == "text" {
			sb.WriteString(block.Text)
			sb.WriteString(" ")
		}
	}
	return sb.String()
}

type Usage struct {
	PromptTokens       int                 `json:"prompt_tokens"`
	CompletionTokens   int                 `json:"completion_tokens"`
	TotalTokens        int                 `json:"total_tokens"`
	PromptTokenDetails *PromptTokenDetails `json:"prompt_token_details,omitempty"`
}

type PromptTokenDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// RequestObjectives represents the scheduling objectives parsed from the InferenceObjectiveSpec, to be
// used in scheduling decisions.
type RequestObjectives struct {
	Priority int
}

// LLMRequest is a structured representation of the fields we parse out of the
// LLMRequest body.
type LLMRequest struct {
	// RequestId is the Envoy generated Id for the request being processed
	RequestId string
	// TargetModel is the final target model after traffic split.
	TargetModel string
	// Data contains the request-body fields that we parse out as user input.
	Body *RequestBody
	// Headers is a map of the request headers.
	Headers map[string]string
	// Request Objective
	Objectives RequestObjectives
	// RequestSizeBytes is the size of the raw request body in bytes when available.
	// Used for token estimation (e.g. inputTokens ≈ RequestSizeBytes/4) without parsing body or calling PlainText().
	RequestSizeBytes int
	// TokenizedPrompt contains the tokenization results if external tokenization is enabled.
	// This is nil if tokenization was not performed or if the tokenizer is not configured.
	TokenizedPrompt *TokenizedPrompt
}

// TokenizedPrompt contains the result of tokenizing the request prompt.
// It is populated by external tokenization plugins (e.g., via a PrepareData plugin)
// and consumed by scheduling plugins that benefit from actual token data
// (e.g., prefix cache scoring, latency prediction).
type TokenizedPrompt struct {
	// TokenIDs are the token IDs for the prompt.
	TokenIDs []uint32
}

type InferenceRequest struct {
	LLM        *LLMRequest
	ParsedBody any `json:"-"`
}

func (r *InferenceRequest) String() string {
	if r == nil || r.LLM == nil {
		return nilString
	}

	return fmt.Sprintf("RequestID: %s, TargetModel: %s, Body: %v, Headers: %v",
		r.LLM.RequestId, r.LLM.TargetModel, r.LLM.Body, r.LLM.Headers)
}
