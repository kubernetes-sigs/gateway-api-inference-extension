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

// ExtractRequestData extracts the LLMRequestData from the given request body map.
func ExtractRequestData(body map[string]any) (*types.LLMRequestData, error) {
	// Convert map back to JSON bytes
	jsonBytes, err := json.Marshal(body)
	if err != nil {
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "invalid request body"}
	}

	// Try completions request first
	var completions types.CompletionsRequest
	if err = json.Unmarshal(jsonBytes, &completions); err == nil && completions.Prompt != "" {
		return &types.LLMRequestData{Completions: &completions}, nil
	}

	// Try chat completions
	var chatCompletions types.ChatCompletionsRequest
	if err = json.Unmarshal(jsonBytes, &chatCompletions); err != nil {
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "invalid request format"}
	}

	if err = validateChatCompletionsMessages(chatCompletions.Messages); err != nil {
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "invalid chat-completions request: " + err.Error()}
	}

	return &types.LLMRequestData{ChatCompletions: &chatCompletions}, nil
}

func validateChatCompletionsMessages(messages []types.Message) error {
	if len(messages) == 0 {
		return errutil.Error{Code: errutil.BadRequest, Msg: "chat-completions request must have at least one message"}
	}

	return nil
}
