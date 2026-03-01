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

package router

import (
	"context"
	"errors"
	"strings"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
)

const (
	RouterParserType = "router-parser"
)

var _ fwkrh.Parser = (*RouterParser)(nil)

// RouterParser wraps a delegate and returns AudioTranscriptionsRequest for allowed header-only paths; otherwise delegates.
type RouterParser struct {
	typedName fwkplugin.TypedName
	delegate  fwkrh.Parser
}

// NewRouterParser returns a parser that handles allowed header-only paths via AudioTranscriptionsRequest and delegates the rest.
func NewRouterParser(delegate fwkrh.Parser) *RouterParser {
	return &RouterParser{
		typedName: fwkplugin.TypedName{Type: RouterParserType, Name: RouterParserType},
		delegate:  delegate,
	}
}

// TypedName returns the type and name of this plugin.
func (p *RouterParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// ParseRequest returns AudioTranscriptionsRequest for allowed header-only paths; otherwise delegates.
func (p *RouterParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*scheduling.LLMRequestBody, error) {
	ct := getHeader(headers, "content-type")
	path := getHeader(headers, ":path")
	if strings.Contains(strings.ToLower(ct), "multipart/form-data") && metadata.PathAllowedForMultipartModelExtraction(path) {
		model := getHeader(headers, metadata.ModelNameKey)
		if model == "" {
			return nil, errors.New("multipart request missing x-gateway-model-name header")
		}
		return &scheduling.LLMRequestBody{AudioTranscriptions: &scheduling.AudioTranscriptionsRequest{ModelName: model}}, nil
	}
	return p.delegate.ParseRequest(ctx, body, headers)
}

// ParseResponse delegates to the underlying parser.
func (p *RouterParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, endOfStream bool) (*fwkrh.ParsedResponse, error) {
	return p.delegate.ParseResponse(ctx, body, headers, endOfStream)
}

func getHeader(headers map[string]string, key string) string {
	for k, v := range headers {
		if strings.EqualFold(k, key) {
			return v
		}
	}
	return ""
}
