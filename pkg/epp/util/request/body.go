package request

import (
	"fmt"

	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
)

func ExtractPromptFromRequestBody(body map[string]interface{}) (string, error) {
	_, ok := body["messages"]
	if ok {
		return extractPromptForChatCompletions(body)
	}
	return extractPromptForCompletions(body)
}

func extractPromptForCompletions(body map[string]interface{}) (string, error) {
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

func extractPromptForChatCompletions(body map[string]interface{}) (string, error) {
	messages, ok := body["messages"]
	if !ok {
		return "", errutil.Error{Code: errutil.BadRequest, Msg: "prompt not found in request"}
	}
	messageList, ok := messages.([]interface{})
	if !ok {
		return "", errutil.Error{Code: errutil.BadRequest, Msg: "messages is not a list"}
	}

	prompt := ""
	if len(messageList) > 0 {
		for _, msg := range messageList {
			msgMap, ok := msg.(map[string]interface{})
			if !ok {
				continue
			}
			content := msgMap["content"]
			if content == nil {
				continue
			}
			contentStr, ok := content.(string)
			if !ok {
				continue
			}
			role := msgMap["role"]
			if role == nil {
				continue
			}
			roleStr, ok := role.(string)
			if !ok {
				continue
			}
			prompt += constructChatMessage(roleStr, contentStr)
		}
	}
	return prompt, nil
}

func constructChatMessage(role string, content string) string {
	return fmt.Sprintf("<|im_start|>%s\n%s<|im_end|>\n", role, content)
}
