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
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	// PluggableParserFeatureGate is the feature gate of the whole pluggable parser layer.
	PluggableParserFeatureGate = "pluggableParser"
	// ParserType is the type of parser plugin.
	ParserType = "parser-type"
)

// Parser defines the interface for parsing payload(requests and responses).
type Parser interface {
	fwkplugin.Plugin
	// ParseRequest parses the request body and headers and returns a map representation.
	ParseRequest(headers map[string]string, body []byte) (*scheduling.LLMRequestBody, error)

	// ParseResponse parses the response body and returns ParsedResponse.
	ParseResponse(body []byte) (*ParsedResponse, error)

	// ParseStreamResponse parses a chunk of the streaming response and returns ParsedResponse.
	ParseStreamResponse(chunk []byte) (*ParsedResponse, error)
}

type ParsedResponse struct {
	// Usage is only populate when the raw response has usage.
	Usage *requestcontrol.Usage
}
