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

	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
)

func ExtractPromptFromRequestBody(body map[string]interface{}) (string, error) {
	if _, ok := body["messages"]; ok {
		return extractPromptFromMessagesField(body)
	}
	return extractPromptField(body)
}

func extractPromptField(body map[string]interface{}) (string, error) {
	prompt, ok := body["prompt"]
	if !ok {
		return "", errutil.Error{Code: errutil.BadRequest, Msg: "prompt not found in request"}
	}
	promptStr, ok := prompt.(string)
	if !ok {
		return "", errutil.Error{Code: errutil.BadRequest, Msg: "prompt is not a string"}
	}
	return promptStr, nil
}

func extractPromptFromMessagesField(body map[string]interface{}) (string, error) {
	messages, ok := body["messages"]
	if !ok {
		return "", errutil.Error{Code: errutil.BadRequest, Msg: "messages not found in request"}
	}
	messageList, ok := messages.([]interface{})
	if !ok {
		return "", errutil.Error{Code: errutil.BadRequest, Msg: "messages is not a list"}
	}
	if len(messageList) == 0 {
		return "", errutil.Error{Code: errutil.BadRequest, Msg: "messages is empty"}
	}

	prompt := ""
	for _, msg := range messageList {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}
		content, ok := msgMap["content"]
		if !ok {
			continue
		}
		contentStr, ok := content.(string)
		if !ok {
			continue
		}
		role, ok := msgMap["role"]
		if !ok {
			continue
		}
		roleStr, ok := role.(string)
		if !ok {
			continue
		}
		prompt += constructChatMessage(roleStr, contentStr)
	}
	return prompt, nil
}

func constructChatMessage(role string, content string) string {
	return fmt.Sprintf("<|im_start|>%s\n%s<|im_end|>\n", role, content)
}
