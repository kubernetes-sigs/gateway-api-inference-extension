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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
)

// ExtractRequestBody extracts the LLMRequestBody from the given request body map.
func ExtractRequestBody(rawBody map[string]any) (*types.LLMRequestBody, error) {
	// Convert map back to JSON bytes
	jsonBytes, err := json.Marshal(rawBody)
	if err != nil {
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "invalid request body"}
	}

	// Try completions request first
	var completions types.CompletionsRequest
	if err = json.Unmarshal(jsonBytes, &completions); err == nil && completions.Prompt != "" {
		return &types.LLMRequestBody{Completions: &completions}, nil
	}

	// Try chat completions
	var chatCompletions types.ChatCompletionsRequest
	if err = json.Unmarshal(jsonBytes, &chatCompletions); err != nil {
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "invalid request format"}
	}

	if err = validateChatCompletionsMessages(chatCompletions.Messages); err != nil {
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "invalid chat-completions request: " + err.Error()}
	}

	return &types.LLMRequestBody{ChatCompletions: &chatCompletions}, nil
}

func validateChatCompletionsMessages(messages []types.Message) error {
	if len(messages) == 0 {
		return errutil.Error{Code: errutil.BadRequest, Msg: "chat-completions request must have at least one message"}
	}

	return nil
}

// GetPromptText extracts the prompt text from either Completions or ChatCompletions format.
// For completions requests, it returns the prompt string directly.
// For chat completions requests, it concatenates all message contents.
// Returns empty string if the request body is invalid or missing.
func GetPromptText(request *types.LLMRequest) string {
	if request == nil || request.Body == nil {
		return ""
	}

	// Try Completions format first (assumed to be valid if not nil)
	if request.Body.Completions != nil {
		return request.Body.Completions.Prompt
	}

	// Must be ChatCompletions request at this point
	if request.Body.ChatCompletions != nil && len(request.Body.ChatCompletions.Messages) > 0 {
		// Concatenate all message contents
		var result string
		for i, msg := range request.Body.ChatCompletions.Messages {
			text := msg.Content.PlainText()
			if text != "" {
				if i > 0 && result != "" {
					result += " "
				}
				result += text
			}
		}
		return result
	}

	return ""
}
