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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

const nilString = "<nil>"

// RequestObjectives represents the scheduling objectives parsed from the InferenceObjectiveSpec, to be used in scheduling decisions.
type RequestObjectives struct {
	Priority int
}

// LLMRequest is a structured representation of the fields we parse out of the LLMRequest body.
type LLMRequest struct {
	// RequestId is the Envoy generated Id for the request being processed
	RequestId string
	// TargetModel is the final target model after traffic split.
	TargetModel string
	// Data contains the request-body fields that we parse out as user input.
	Body *LLMRequestBody
	// Headers is a map of the request headers.
	Headers map[string]string
	// Request Objective
	Objectives RequestObjectives
}

func (r *LLMRequest) String() string {
	if r == nil {
		return nilString
	}

	return fmt.Sprintf("RequestID: %s, TargetModel: %s, Body: %s, Headers: %v",
		r.RequestId, r.TargetModel, r.Body, r.Headers)
}

// LLMRequestBody contains the request-body fields that we parse out as user input,
// to be used in forming scheduling decisions.
// An LLMRequestBody must contain exactly one of CompletionsRequest, ChatCompletionsRequest, ResponsesRequest, or ConversationsRequest.
type LLMRequestBody struct {
	// CompletionsRequest is the representation of the OpenAI /v1/completions request body.
	Completions *CompletionsRequest `json:"completions,omitempty"`
	// ChatCompletionsRequest is the representation of the OpenAI /v1/chat/completions request body.
	ChatCompletions *ChatCompletionsRequest `json:"chat_completions,omitempty"`
	// ResponsesRequest is the representation of the OpenAI /v1/responses request body.
	Responses *ResponsesRequest `json:"responses,omitempty"`
	// ConversationsRequest is the representation of the OpenAI /v1/conversations request body.
	Conversations *ConversationsRequest `json:"conversations,omitempty"`
}

// PromptText returns a plain-text representation of the prompt from whichever
// API type is populated, analogous to CacheSalt().
func (r *LLMRequestBody) PromptText() string {
	switch {
	case r.Completions != nil:
		return r.Completions.Prompt
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

func (r *LLMRequestBody) CacheSalt() string {
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

type Endpoint interface {
	GetMetadata() *fwkdl.EndpointMetadata
	GetMetrics() *fwkdl.Metrics
	String() string
	Get(string) (fwkdl.Cloneable, bool)
	Put(string, fwkdl.Cloneable)
	Keys() []string
}

func (ep *endpoint) String() string {
	if ep == nil {
		return nilString
	}

	return fmt.Sprintf("%+v", *ep)
}

func (ep *endpoint) GetMetadata() *fwkdl.EndpointMetadata {
	return ep.EndpointMetadata
}

func (ep *endpoint) GetMetrics() *fwkdl.Metrics {
	return ep.Metrics
}

type endpoint struct {
	*fwkdl.EndpointMetadata
	*fwkdl.Metrics
	fwkdl.AttributeMap
}

func NewEndpoint(meta *fwkdl.EndpointMetadata, metrics *fwkdl.Metrics, attr fwkdl.AttributeMap) Endpoint {
	if attr == nil {
		attr = fwkdl.NewAttributes()
	}

	return &endpoint{
		EndpointMetadata: meta.Clone(),
		Metrics:          metrics.Clone(),
		AttributeMap:     attr.Clone(),
	}
}

func EndpointComparer(a, b Endpoint) bool {
	a_ep := a.(*endpoint)
	b_ep := b.(*endpoint)
	return reflect.DeepEqual(a_ep, b_ep)
}

func ScoredEndpointComparer(a, b ScoredEndpoint) bool {
	return a.Score == b.Score && EndpointComparer(a.Endpoint, b.Endpoint)
}

type ScoredEndpoint struct {
	Endpoint
	Score float64
}

// ProfileRunResult captures the profile run result.
type ProfileRunResult struct {
	TargetEndpoints []Endpoint
}

// SchedulingResult captures the result of the scheduling cycle.
type SchedulingResult struct {
	ProfileResults     map[string]*ProfileRunResult
	PrimaryProfileName string
}

type SchedulerProfile interface {
	Run(ctx context.Context, request *LLMRequest, cycleState *CycleState, candidateEndpoints []Endpoint) (*ProfileRunResult, error)
}
