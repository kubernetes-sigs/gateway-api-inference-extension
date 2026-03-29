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

package openai

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkrc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	OpenAIParserType = "openai-parser"

	conversationsAPI   = "conversations"
	responsesAPI       = "responses"
	chatCompletionsAPI = "chat/completions"
	completionsAPI     = "completions"
	embeddingsAPI      = "embeddings"

	streamingRespPrefix = "data: "
	streamingEndMsg     = "data: [DONE]"

	// OpenAI API object types
	objectTypeResponse            = "response"
	objectTypeConversation        = "conversation"
	objectTypeChatCompletion      = "chat.completion"
	objectTypeChatCompletionChunk = "chat.completion.chunk"
	objectTypeTextCompletion      = "text_completion"

	contentType = "content-type"
	// The base media type for Server-Sent Events. We check for this substring
	// to account for optional parameters like "; charset=utf-8" often appended by proxies.
	eventStreamType = "text/event-stream"
)

// compile-time type validation
var _ fwkrh.Parser = &OpenAIParser{}

// OpenAIParser implements the fwkrh.Parser interface for OpenAI API
// https://developers.openai.com/api/reference/overview
type OpenAIParser struct {
	typedName fwkplugin.TypedName
}

// NewOpenAIParser creates a new OpenAIParser.
func NewOpenAIParser() *OpenAIParser {
	return &OpenAIParser{
		typedName: fwkplugin.TypedName{
			Type: OpenAIParserType,
			Name: OpenAIParserType,
		},
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *OpenAIParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func OpenAIParserPluginFactory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewOpenAIParser().WithName(name), nil
}

func (p *OpenAIParser) WithName(name string) *OpenAIParser {
	p.typedName.Name = name
	return p
}

// ParseRequest parses the request body and headers and returns a map representation.
func (p *OpenAIParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*scheduling.LLMRequestBody, error) {
	bodyMap := make(map[string]any)
	if err := json.Unmarshal(body, &bodyMap); err != nil {
		return nil, errors.New("error unmarshaling request bodyMap")
	}
	extractedBody, err := extractRequestBody(body, headers)
	if err != nil {
		return nil, err
	}
	extractedBody.ParsedBody = bodyMap
	if stream, ok := bodyMap["stream"].(bool); ok && stream {
		extractedBody.Stream = true
	}
	return extractedBody, nil
}

// ParseResponse extracts usage metadata from the provider's response.
// It automatically detects and handles both standard JSON responses and SSE streams.
func (p *OpenAIParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, _ bool) (*fwkrh.ParsedResponse, error) {
	if len(body) == 0 {
		// An empty body can occur during streaming; for instance, Envoy proxies
		// may emit a trailing empty body with the EndOfStream flag set to true.
		return nil, nil
	}

	isStream := false
	for k, v := range headers {
		if strings.ToLower(k) == contentType && strings.Contains(strings.ToLower(v), eventStreamType) {
			isStream = true
			break
		}
	}
	if isStream {
		return p.parseStreamResponse(body)
	}

	usage, err := extractUsage(body)
	if err != nil {
		return nil, err
	}
	return &fwkrh.ParsedResponse{Usage: usage}, nil
}

func (p *OpenAIParser) parseStreamResponse(chunk []byte) (*fwkrh.ParsedResponse, error) {
	usage := extractUsageStreaming(string(chunk))
	return &fwkrh.ParsedResponse{
		Usage: usage,
	}, nil
}

// getRequestPath extracts the request path from headers with fallback priority
func getRequestPath(headers map[string]string) string {
	// Try primary path header
	if path := headers[":path"]; path != "" {
		return path
	}

	// Try fallback headers
	if path := headers["x-original-path"]; path != "" {
		return path
	}

	if path := headers["x-forwarded-path"]; path != "" {
		return path
	}

	// Default to completions API for backward compatibility with existing clients and integration tests
	return "/v1/completions"
}

// determineAPITypeFromPath determines the API type based on the request path.
// Note: path strings have already been cleaned and normalized by the gateway/proxy layer
// (no trailing slashes, query parameters, or additional suffix strings at this point).
func determineAPITypeFromPath(path string) string {
	if strings.Contains(path, "/v1/conversations") {
		return conversationsAPI
	}
	if strings.Contains(path, "/v1/responses") {
		return responsesAPI
	}
	if strings.Contains(path, "/v1/chat/completions") {
		return chatCompletionsAPI
	}
	if strings.Contains(path, "/v1/completions") {
		return completionsAPI
	}
	if strings.Contains(path, "/v1/embeddings") {
		return embeddingsAPI
	}

	// Default to completions API for backward compatibility with existing clients and integration tests
	return completionsAPI
}

// extractRequestBody extracts the LLMRequestBody from the given request body map using path-based detection.
func extractRequestBody(rawBody []byte, headers map[string]string) (*scheduling.LLMRequestBody, error) {
	// Determine API type from request path
	path := getRequestPath(headers)
	apiType := determineAPITypeFromPath(path)

	switch apiType {
	case conversationsAPI:
		var conversations scheduling.ConversationsRequest
		if err := json.Unmarshal(rawBody, &conversations); err == nil && len(conversations.Items) > 0 {
			return &scheduling.LLMRequestBody{Conversations: &conversations}, nil
		}
		return nil, errors.New("invalid conversations request: must have items field")

	case responsesAPI:
		var responses scheduling.ResponsesRequest
		if err := json.Unmarshal(rawBody, &responses); err == nil && responses.Input != nil {
			return &scheduling.LLMRequestBody{Responses: &responses}, nil
		}
		return nil, errors.New("invalid responses request: must have input field")

	case chatCompletionsAPI:
		var chatCompletions scheduling.ChatCompletionsRequest
		if err := json.Unmarshal(rawBody, &chatCompletions); err == nil {
			if err = validateChatCompletionsMessages(chatCompletions.Messages); err == nil {
				return &scheduling.LLMRequestBody{ChatCompletions: &chatCompletions}, nil
			}
		}
		return nil, errors.New("invalid chat completions request: must have valid messages field")

	case completionsAPI:
		var completions scheduling.CompletionsRequest
		if err := json.Unmarshal(rawBody, &completions); err == nil && !completions.Prompt.IsEmpty() {
			return &scheduling.LLMRequestBody{Completions: &completions}, nil
		}
		return nil, errors.New("invalid completions request: must have prompt field")

	case embeddingsAPI:
		var embeddings scheduling.EmbeddingsRequest
		if err := json.Unmarshal(rawBody, &embeddings); err == nil && embeddings.Input != nil {
			return &scheduling.LLMRequestBody{Embeddings: &embeddings}, nil
		}
		return nil, errors.New("invalid embeddings request: must have input field")

	default:
		return nil, errors.New("unsupported API endpoint")
	}
}

func validateChatCompletionsMessages(messages []scheduling.Message) error {
	if len(messages) == 0 {
		return errors.New("chat-completions request must have at least one message")
	}
	return nil
}

func extractUsage(responseBytes []byte) (*fwkrc.Usage, error) {
	var responseErr error
	var responseBody map[string]any
	responseErr = json.Unmarshal(responseBytes, &responseBody)
	if responseErr != nil {
		return nil, responseErr
	}

	if responseBody["usage"] != nil {
		usg := responseBody["usage"].(map[string]any)
		objectType, _ := responseBody["object"].(string)
		usage := extractUsageByAPIType(usg, objectType)
		if usg["prompt_token_details"] != nil {
			detailsMap := usg["prompt_token_details"].(map[string]any)
			if cachedTokens, ok := detailsMap["cached_tokens"]; ok {
				usage.PromptTokenDetails = &fwkrc.PromptTokenDetails{
					CachedTokens: int(cachedTokens.(float64)),
				}
			}
		}
		return &usage, nil
	}
	return nil, nil
}

// extractUsageByAPIType extracts usage statistics using the appropriate field names
// based on the OpenAI API type identified by the "object" field.
func extractUsageByAPIType(usg map[string]any, objectType string) fwkrc.Usage {
	usage := fwkrc.Usage{}

	switch {
	case strings.HasPrefix(objectType, objectTypeResponse) || strings.HasPrefix(objectType, objectTypeConversation):
		// Responses/Conversations APIs use input_tokens/output_tokens
		if usg["input_tokens"] != nil {
			usage.PromptTokens = int(usg["input_tokens"].(float64))
		}
		if usg["output_tokens"] != nil {
			usage.CompletionTokens = int(usg["output_tokens"].(float64))
		}
	case objectType == objectTypeChatCompletion || objectType == objectTypeChatCompletionChunk || objectType == objectTypeTextCompletion:
		// Traditional APIs use prompt_tokens/completion_tokens
		if usg["prompt_tokens"] != nil {
			usage.PromptTokens = int(usg["prompt_tokens"].(float64))
		}
		if usg["completion_tokens"] != nil {
			usage.CompletionTokens = int(usg["completion_tokens"].(float64))
		}
	default:
		// Fallback: try both field naming conventions
		if usg["input_tokens"] != nil {
			usage.PromptTokens = int(usg["input_tokens"].(float64))
		} else if usg["prompt_tokens"] != nil {
			usage.PromptTokens = int(usg["prompt_tokens"].(float64))
		}

		if usg["output_tokens"] != nil {
			usage.CompletionTokens = int(usg["output_tokens"].(float64))
		} else if usg["completion_tokens"] != nil {
			usage.CompletionTokens = int(usg["completion_tokens"].(float64))
		}
	}

	// total_tokens field name is consistent across all API types
	if usg["total_tokens"] != nil {
		usage.TotalTokens = int(usg["total_tokens"].(float64))
	}

	return usage
}

// Example message if "stream_options": {"include_usage": "true"} is included in the request:
// data: {"id":"...","object":"text_completion","created":1739400043,"model":"small-segment-lora-0","choices":[],
// "usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}
//
// data: [DONE]
//
// Noticed that vLLM returns two entries in one response.
// We need to strip the `data:` prefix and next Data: [DONE] from the message to fetch response data.
//
// If include_usage is not included in the request, `data: [DONE]` is returned separately, which
// indicates end of streaming.
//
// For ResponsesAPI streaming, usage is nested in the response object:
//
//	event: response.completed
//	data: {"response":{"usage":{"input_tokens":31,..},...},"type":"response.completed"}
//
// It extracts usage from events with type="response.completed".
func extractUsageStreaming(responseText string) *fwkrc.Usage {

	var streamResponse struct {
		Usage    *fwkrc.Usage `json:"usage"`
		Response struct {
			Usage  map[string]any `json:"usage"`
			Object string         `json:"object"`
		} `json:"response"`
		Type string `json:"type"`
	}

	lines := strings.SplitSeq(responseText, "\n")
	for line := range lines {
		if !strings.HasPrefix(line, streamingRespPrefix) {
			continue
		}
		content := strings.TrimPrefix(line, streamingRespPrefix)
		if content == "[DONE]" {
			continue
		}

		byteSlice := []byte(content)
		if err := json.Unmarshal(byteSlice, &streamResponse); err != nil {
			continue
		}
		// Standard ChatCompletion / vLLM usage format
		if streamResponse.Usage != nil {
			return streamResponse.Usage
		}
		// Responses API streaming format
		if streamResponse.Response.Usage != nil && streamResponse.Type == "response.completed" {
			// Convert map[string]any to JSON and parse
			jsonBytes, _ := json.Marshal(map[string]any{
				"usage":  streamResponse.Response.Usage,
				"object": streamResponse.Response.Object,
			})
			if usage, err := extractUsage(jsonBytes); err == nil && usage != nil {
				return usage
			}
		}
	}
	return nil
}
