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
	"fmt"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
)

// ExtractRequestData extracts the LLMRequestData from the given request body map.
func ExtractRequestData(body map[string]any) (*types.LLMRequestData, error) {
	if body == nil {
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "request body is nil"}
	}

	// Since the types.LLMRequestData is a disjoint union between CompletionsRequest and ChatCompletionsRequest,
	// we need to determine which one it is based on the presence of fields.

	// Try to extract a CompletionsRequest first.
	completionsRequest, found, err := maybeExtractCompletionsRequest(body)
	if err != nil {
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "error extracting completions request: " + err.Error()}
	}

	if found {
		return &types.LLMRequestData{
			Completions: completionsRequest,
		}, nil
	}

	// At this point, must extract a ChatCompletionsRequest or fail.
	chatCompletionsRequest, err := extractChatCompletionsRequest(body)
	if err != nil {
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "failed to extract chat CompletionsRequest"}
	}

	return &types.LLMRequestData{
		ChatCompletions: chatCompletionsRequest,
	}, nil
}

func maybeExtractCompletionsRequest(body map[string]any) (*types.CompletionsRequest, bool, error) {
	prompt, found, err := extractPromptField(body)
	if !found {
		return nil, false, nil
	}
	if err != nil {
		return nil, true, err
	}

	return &types.CompletionsRequest{
		Prompt: prompt,
	}, true, nil
}

func extractPromptField(body map[string]any) (string, bool, error) {
	prompt, ok := body["prompt"]
	if !ok {
		return "", false, errutil.Error{Code: errutil.BadRequest, Msg: "prompt not found in request"}
	}
	promptStr, ok := prompt.(string)
	if !ok {
		return "", true, errutil.Error{Code: errutil.BadRequest, Msg: "prompt is not a string"}
	}
	return promptStr, true, nil
}

func extractChatCompletionsRequest(body map[string]any) (*types.ChatCompletionsRequest, error) {
	messages, err := extractMessagesField(body)
	if err != nil {
		return nil, err
	}

	tools, err := maybeExtractToolsField(body)
	if err != nil {
		return nil, err
	}

	documents, err := maybeExtractDocumentsField(body)
	if err != nil {
		return nil, err
	}

	chatTemplate, err := maybeExtractChatTemplateField(body)
	if err != nil {
		return nil, err
	}

	returnAssistantTokensMask, err := maybeExtractReturnAssistantTokensMaskField(body)
	if err != nil {
		return nil, err
	}

	continueFinalMessage, err := maybeExtractContinueFinalMessageField(body)
	if err != nil {
		return nil, err
	}

	addGenerationPrompt, err := maybeExtractAddGenerationPromptField(body)
	if err != nil {
		return nil, err
	}

	chatTemplateKWArgs, err := maybeExtractChatTemplateKWArgsField(body)
	if err != nil {
		return nil, err
	}

	return &types.ChatCompletionsRequest{
		Messages:                  messages,
		Tools:                     tools,
		Documents:                 documents,
		ChatTemplate:              chatTemplate,
		ReturnAssistantTokensMask: returnAssistantTokensMask,
		ContinueFinalMessage:      continueFinalMessage,
		AddGenerationPrompt:       addGenerationPrompt,
		ChatTemplateKWArgs:        chatTemplateKWArgs,
	}, nil
}

func extractMessagesField(body map[string]any) ([]types.Message, error) {
	messagesRaw, ok := body["messages"]
	if !ok {
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "messages not found in request"}
	}
	messagesSlice, ok := messagesRaw.([]any)
	if !ok {
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "messages is not an array"}
	}

	if len(messagesSlice) == 0 {
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "messages array is empty"}
	}

	messages := make([]types.Message, 0, len(messagesSlice))
	for i, msgRaw := range messagesSlice {
		msgMap, ok := msgRaw.(map[string]any)
		if !ok {
			return nil, errutil.Error{Code: errutil.BadRequest, Msg: fmt.Sprintf("message at index %d is not an object", i)}
		}
		roleRaw, ok := msgMap["role"]
		if !ok {
			return nil, errutil.Error{Code: errutil.BadRequest, Msg: fmt.Sprintf("role not found in message at index %d", i)}
		}
		roleStr, ok := roleRaw.(string)
		if !ok {
			return nil, errutil.Error{Code: errutil.BadRequest, Msg: fmt.Sprintf("role in message at index %d is not a string", i)}
		}
		contentRaw, ok := msgMap["content"]
		if !ok {
			return nil, errutil.Error{Code: errutil.BadRequest, Msg: fmt.Sprintf("content not found in message at index %d", i)}
		}
		contentStr, ok := contentRaw.(string)
		if !ok {
			return nil, errutil.Error{Code: errutil.BadRequest, Msg: fmt.Sprintf("content in message at index %d is not a string", i)}
		}
		messages = append(messages, types.Message{
			Role:    roleStr,
			Content: contentStr,
		})
	}

	return messages, nil
}

func maybeExtractToolsField(body map[string]any) ([]any, error) {
	toolsRaw, ok := body["tools"]
	if !ok {
		// tools is optional
		return nil, nil
	}
	toolsSlice, ok := toolsRaw.([]any)
	if !ok {
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "tools is not an array"}
	}
	return toolsSlice, nil
}

func maybeExtractDocumentsField(body map[string]any) ([]any, error) {
	documentsRaw, ok := body["documents"]
	if !ok {
		// documents is optional
		return nil, nil
	}
	documentsSlice, ok := documentsRaw.([]any)
	if !ok {
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "documents is not an array"}
	}
	return documentsSlice, nil
}

func maybeExtractChatTemplateField(body map[string]any) (string, error) {
	chatTemplateRaw, ok := body["chat_template"]
	if !ok {
		// chat_template is optional
		return "", nil
	}
	chatTemplateStr, ok := chatTemplateRaw.(string)
	if !ok {
		return "", errutil.Error{Code: errutil.BadRequest, Msg: "chat_template is not a string"}
	}
	return chatTemplateStr, nil
}

func maybeExtractReturnAssistantTokensMaskField(body map[string]any) (bool, error) {
	returnAssistantTokensMaskRaw, ok := body["return_assistant_tokens_mask"]
	if !ok {
		// return_assistant_tokens_mask is optional
		return false, nil
	}
	returnAssistantTokensMaskBool, ok := returnAssistantTokensMaskRaw.(bool)
	if !ok {
		return false, errutil.Error{Code: errutil.BadRequest, Msg: "return_assistant_tokens_mask is not a boolean"}
	}
	return returnAssistantTokensMaskBool, nil
}

func maybeExtractContinueFinalMessageField(body map[string]any) (bool, error) {
	continueFinalMessageRaw, ok := body["continue_final_message"]
	if !ok {
		// continue_final_message is optional
		return false, nil
	}
	continueFinalMessageBool, ok := continueFinalMessageRaw.(bool)
	if !ok {
		return false, errutil.Error{Code: errutil.BadRequest, Msg: "continue_final_message is not a boolean"}
	}
	return continueFinalMessageBool, nil
}

func maybeExtractAddGenerationPromptField(body map[string]any) (bool, error) {
	addGenerationPromptRaw, ok := body["add_generation_prompt"]
	if !ok {
		// add_generation_prompt is optional
		return false, nil
	}
	addGenerationPromptBool, ok := addGenerationPromptRaw.(bool)
	if !ok {
		return false, errutil.Error{Code: errutil.BadRequest, Msg: "add_generation_prompt is not a boolean"}
	}
	return addGenerationPromptBool, nil
}

func maybeExtractChatTemplateKWArgsField(body map[string]any) (map[string]any, error) {
	chatTemplateKWArgsRaw, ok := body["chat_template_kwargs"]
	if !ok {
		// chat_template_kwargs is optional
		return nil, nil
	}
	chatTemplateKWArgsMap, ok := chatTemplateKWArgsRaw.(map[string]any)
	if !ok {
		return nil, errutil.Error{Code: errutil.BadRequest, Msg: "chat_template_kwargs is not an object"}
	}
	return chatTemplateKWArgsMap, nil
}
