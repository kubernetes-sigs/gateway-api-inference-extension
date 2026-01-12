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
	"encoding/json"
	"strings"

	types "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
)

const (
	conversationsAPI   = "conversations"
	responsesAPI       = "responses"
	chatCompletionsAPI = "chat/completions"
	completionsAPI     = "completions"
)

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

	// Default to completions API for backward compatibility with existing clients and integration tests
	return completionsAPI
}

// ExtractRequestBody extracts the LLMRequestBody from the given request body map using path-based detection.
func ExtractRequestBody(rawBody map[string]any, headers map[string]string) (*types.LLMRequestBody, error) {
	jsonBytes, err := json.Marshal(rawBody)
	if err != nil {
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "invalid request body"}
	}

	// Determine API type from request path
	path := getRequestPath(headers)
	apiType := determineAPITypeFromPath(path)

	switch apiType {
	case conversationsAPI:
		var conversations types.ConversationsRequest
		if err = json.Unmarshal(jsonBytes, &conversations); err == nil && len(conversations.Items) > 0 {
			return &types.LLMRequestBody{Conversations: &conversations}, nil
		}
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "invalid conversations request: must have items field"}

	case responsesAPI:
		var responses types.ResponsesRequest
		if err = json.Unmarshal(jsonBytes, &responses); err == nil && responses.Input != nil {
			return &types.LLMRequestBody{Responses: &responses}, nil
		}
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "invalid responses request: must have input field"}

	case chatCompletionsAPI:
		var chatCompletions types.ChatCompletionsRequest
		if err = json.Unmarshal(jsonBytes, &chatCompletions); err == nil {
			if err = validateChatCompletionsMessages(chatCompletions.Messages); err == nil {
				return &types.LLMRequestBody{ChatCompletions: &chatCompletions}, nil
			}
		}
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "invalid chat completions request: must have valid messages field"}

	case completionsAPI:
		var completions types.CompletionsRequest
		if err = json.Unmarshal(jsonBytes, &completions); err == nil && completions.Prompt != "" {
			return &types.LLMRequestBody{Completions: &completions}, nil
		}
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "invalid completions request: must have prompt field"}

	default:
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "unsupported API endpoint"}
	}
}

func validateChatCompletionsMessages(messages []types.Message) error {
	if len(messages) == 0 {
		return errutil.Error{Code: errutil.BadRequest, Msg: "chat-completions request must have at least one message"}
	}

	return nil
}
