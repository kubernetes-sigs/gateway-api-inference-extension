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

package payloadprocess

import (
	"encoding/json"
	"errors"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/payloadprocess"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
	resputil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/response"
)

const (
	streamingRespPrefix = "data: "       //nolint:unused
	streamingEndMsg     = "data: [DONE]" //nolint:unused

	// OpenAI API object types
	objectTypeResponse            = "response"              //nolint:unused
	objectTypeConversation        = "conversation"          //nolint:unused
	objectTypeChatCompletion      = "chat.completion"       //nolint:unused
	objectTypeChatCompletionChunk = "chat.completion.chunk" //nolint:unused
	objectTypeTextCompletion      = "text_completion"       //nolint:unused
)

const (
	OpenAIParserName = "openai-parser"
)

// compile-time type validation
var _ payloadprocess.Parser = &OpenAIParser{}

// OpenAIParser implements the backend.OpenAIParser interface for OpenAI API.
type OpenAIParser struct {
	typedName fwkplugin.TypedName
}

// NewOpenAIParser creates a new OpenAIParser.
func NewOpenAIParser() *OpenAIParser {
	return &OpenAIParser{
		typedName: fwkplugin.TypedName{
			Type: payloadprocess.ParserType,
			Name: OpenAIParserName,
		},
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *OpenAIParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// ParseRequest parses the request body and headers and returns a map representation.
func (p *OpenAIParser) ParseRequest(headers map[string]string, body []byte) (*scheduling.LLMRequestBody, error) {
	bodyMap := make(map[string]any)
	if err := json.Unmarshal(body, &bodyMap); err != nil {
		return nil, errors.New("error unmarshalling the bodyMap")
	}
	extractedBody, err := requtil.ExtractRequestBody(body, headers)
	if err != nil {
		return nil, err
	}
	extractedBody.ParsedBody = bodyMap
	return extractedBody, nil
}

// // ParseResponse parses the response body and returns a ParsedResponse
func (p *OpenAIParser) ParseResponse(body []byte) (*payloadprocess.ParsedResponse, error) {
	usage, err := resputil.ExtractUsage(body)
	if err != nil || usage == nil {
		return nil, err
	}
	return &payloadprocess.ParsedResponse{Usage: usage}, nil
}

// ParseStreamResponse parses a chunk of the streaming response and returns a ParsedResponse
func (p *OpenAIParser) ParseStreamResponse(chunk []byte) (*payloadprocess.ParsedResponse, error) {
	responseBody := resputil.ExtractUsageStreaming(string(chunk))
	if responseBody.Usage == nil {
		return nil, errors.New("unable to parse usage from stream response")
	}
	return &payloadprocess.ParsedResponse{
		Usage: responseBody.Usage,
	}, nil
}
