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

	"google.golang.org/protobuf/proto"
)

const nilStr = "<nil>"

// Modality identifies the type of multimodal content in a prompt.
type Modality string

// ModalityImage is the only currently supported modality.
const ModalityImage Modality = "image"

// RequestPayload represents a strongly-typed unmarshaled request payload or raw bytes.
type RequestPayload interface {
	isRequestPayload()
	IsParsed() bool
}

// PayloadMap represents a JSON request body unmarshaled into a map.
type PayloadMap map[string]any

func (PayloadMap) isRequestPayload() {}
func (PayloadMap) IsParsed() bool    { return true }

// PayloadProto represents a gRPC request body unmarshaled into a proto.Message.
type PayloadProto struct {
	proto.Message
}

func (PayloadProto) isRequestPayload() {}
func (PayloadProto) IsParsed() bool    { return true }

// RawPayload represents an unparsed request body kept as raw bytes.
type RawPayload []byte

func (RawPayload) isRequestPayload() {}
func (RawPayload) IsParsed() bool    { return false }

// InferenceRequestBody contains the request-body fields that we parse out as user input,
// to be used in forming scheduling decisions.
// An InferenceRequestBody must contain exactly one of CompletionsRequest, ChatCompletionsRequest, ResponsesRequest, ConversationsRequest, or EmbeddingsRequest.
type InferenceRequestBody struct {
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
	// Payload contains the unmarshaled request payload or raw bytes.
	// If the payload is unmarshaled, we can perform advanced processing (like prefix cache aware routing).
	// If it remains as raw bytes, such processing may not be supported.
	Payload RequestPayload `json:"-"`
	// TokenizedPrompt contains parser-derived tokenization results when available.
	// It is nil when the request was not already tokenized.
	TokenizedPrompt *TokenizedPrompt `json:"-"`

	// Stream indicates whether the request specifies a streaming response (e.g., via a stream field).
	// This typically implies the model server's response will be streamed.
	Stream bool `json:"-"`
}

// TokenizedPrompt contains the result of tokenizing the request prompt.
// It is consumed by scheduling and request-control plugins that benefit from
// actual token data such as prefix-cache awareness.
type TokenizedPrompt struct {
	// TokenIDs are the token IDs for the prompt, including multimodal placeholder tokens.
	TokenIDs []uint32
	// MultiModalFeatures holds one entry per multimodal item in prompt order.
	// Nil if the prompt contains no multimodal content.
	MultiModalFeatures []MultiModalFeature
}

// MultiModalFeature holds all data needed for prefix-cache scoring of a single
// multimodal item. Items are ordered by token position within the prompt.
// Currently only ModalityImage is supported.
type MultiModalFeature struct {
	// Modality identifies the type of content.
	Modality Modality
	// Hash is the content hash of the item, used for KV-cache reuse across requests.
	Hash string
	// Offset is the index of the first placeholder token for this item in TokenIDs.
	Offset int
	// Length is the number of placeholder tokens this item occupies in TokenIDs.
	Length int
}

// PromptText returns a plain-text representation of the prompt from whichever
// API type is populated, analogous to CacheSalt().
func (r *InferenceRequestBody) PromptText() string {
	switch {
	case r.Completions != nil:
		return r.Completions.Prompt.PlainText()
	case r.ChatCompletions != nil:
		var sb strings.Builder
		for _, msg := range r.ChatCompletions.Messages {
			text := msg.Content.PlainText()
			if text != "" {
				sb.WriteString(text)
				sb.WriteString(" ")
			}
		}
		return sb.String()
	case r.Responses != nil:
		if s, ok := r.Responses.Input.(string); ok {
			return s
		}
		b, _ := json.Marshal(r.Responses.Input)
		return string(b)
	case r.Conversations != nil:
		b, _ := json.Marshal(r.Conversations.Items)
		return string(b)
	default:
		return ""
	}
}

func (r *InferenceRequestBody) CacheSalt() string {
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
	if r.Embeddings != nil {
		return r.Embeddings.CacheSalt
	}
	return ""
}

// Prompt represents the prompt field in a /v1/completions request.
// Per the OpenAI spec it can be a string or an array of strings.
// See https://platform.openai.com/docs/api-reference/completions/create#completions-create-prompt
type Prompt struct {
	Raw     string
	Strings []string
}

func (p *Prompt) UnmarshalJSON(data []byte) error {
	if len(data) > 0 && data[0] == '"' {
		return json.Unmarshal(data, &p.Raw)
	}
	if len(data) > 0 && data[0] == '[' {
		return json.Unmarshal(data, &p.Strings)
	}
	return errors.New("prompt: must be a string or an array of strings")
}

func (p Prompt) MarshalJSON() ([]byte, error) {
	if p.Raw != "" {
		return json.Marshal(p.Raw)
	}
	if p.Strings != nil {
		return json.Marshal(p.Strings)
	}
	return json.Marshal("")
}

func (p Prompt) PlainText() string {
	if p.Raw != "" {
		return p.Raw
	}
	return strings.Join(p.Strings, " ")
}

func (p Prompt) IsEmpty() bool {
	return p.Raw == "" && len(p.Strings) == 0
}

// CompletionsRequest is a structured representation of the fields we parse out of the /v1/completions request
// body. For detailed body fields, please refer to https://platform.openai.com/docs/api-reference/completions.
// This struct includes fields usable for plugins and scheduling decisions - and not the entire
// API spec.
type CompletionsRequest struct {
	// Prompt is the prompt(s) sent in the request body; can be a string or an array of strings.
	Prompt Prompt `json:"prompt"`
	// CacheSalt is an optional request parameter to isolate prefix caches for security reasons.
	CacheSalt string `json:"cache_salt,omitempty"`
}

func (r *CompletionsRequest) String() string {
	if r == nil {
		return nilStr
	}

	return fmt.Sprintf("{PromptLength: %d}", len(r.Prompt.PlainText()))
}

// ChatCompletionsRequest is a structured representation of the fields we parse out of the v1/chat/completions
// request body. For detailed body fields, please refer to https://platform.openai.com/docs/api-reference/chat.
// This struct includes fields usable for plugins and scheduling decisions - and not the entire
// API spec.
type ChatCompletionsRequest struct {
	/* parameters from the official OpenAI chat-completions API */
	Messages []Message `json:"messages,omitempty"`
	Tools    []any     `json:"tools,omitempty"`
	/* parameters from the HuggingFace transformers chat-templates API */
	Documents                 []any          `json:"documents,omitempty"`
	ChatTemplate              string         `json:"chat_template,omitempty"`
	ReturnAssistantTokensMask bool           `json:"return_assistant_tokens_mask,omitempty"`
	ContinueFinalMessage      bool           `json:"continue_final_message,omitempty"`
	AddGenerationPrompt       bool           `json:"add_generation_prompt,omitempty"`
	ChatTemplateKWArgs        map[string]any `json:"chat_template_kwargs,omitempty"`
	// CacheSalt is an optional request parameter to isolate prefix caches for security reasons.
	CacheSalt string `json:"cache_salt,omitempty"`
}

func (r *ChatCompletionsRequest) String() string {
	if r == nil {
		return nilStr
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
	Input any `json:"input,omitempty"`
	// Instructions provides optional system-level guidance
	Instructions any `json:"instructions,omitempty"`
	// Tools field for function calling capabilities
	Tools any `json:"tools,omitempty"`
	// CacheSalt isolates prefix caches for security
	CacheSalt string `json:"cache_salt,omitempty"`
}

func (r *ResponsesRequest) String() string {
	if r == nil {
		return nilStr
	}
	return fmt.Sprintf("{InputType: %T, InstructionsType: %T}", r.Input, r.Instructions)
}

// ConversationsRequest represents the OpenAI /v1/conversations request body structure
type ConversationsRequest struct {
	// Items is the array of conversation items (messages, files, etc.)
	Items []ConversationItem `json:"items,omitempty"`
	// Metadata provides additional context for the conversation
	Metadata map[string]any `json:"metadata,omitempty"`
	// CacheSalt isolates prefix caches for security
	CacheSalt string `json:"cache_salt,omitempty"`
}

func (c *ConversationsRequest) String() string {
	if c == nil {
		return nilStr
	}
	return fmt.Sprintf("{ItemsCount: %d}", len(c.Items))
}

// EmbeddingsRequest represents the OpenAI /v1/embeddings request body structure.
// Input can be a string or array of strings; see https://platform.openai.com/docs/api-reference/embeddings.
type EmbeddingsRequest struct {
	// Input is the text to embed (string or array of strings).
	Input any `json:"input,omitempty"`
	// CacheSalt is an optional request parameter to isolate prefix caches for security reasons.
	CacheSalt string `json:"cache_salt,omitempty"`
}

func (e *EmbeddingsRequest) String() string {
	if e == nil {
		return nilStr
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
	Content any `json:"content,omitempty"`
}

// Message represents a single message in a chat-completions request.
type Message struct {
	// Role is the message Role, optional values are 'user', 'assistant', ...
	Role string `json:"role,omitempty"`
	// Content defines text of this message
	Content Content `json:"content"`
}

type Content struct {
	Raw        string
	Structured []ContentBlock
}

type ContentBlock struct {
	Type       string     `json:"type"`
	Text       string     `json:"text,omitempty"`
	ImageURL   ImageBlock `json:"image_url"`
	InputAudio AudioBlock `json:"input_audio"`
	VideoURL   VideoBlock `json:"video_url"`
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

// Usage counts parsed from the response body.
type Usage struct {
	PromptTokens       int                 `json:"prompt_tokens"`
	CompletionTokens   int                 `json:"completion_tokens"`
	TotalTokens        int                 `json:"total_tokens"`
	PromptTokenDetails *PromptTokenDetails `json:"prompt_token_details,omitempty"`
}

type PromptTokenDetails struct {
	CachedTokens int `json:"cached_tokens"`
}
