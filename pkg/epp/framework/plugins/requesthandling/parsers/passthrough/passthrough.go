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

package passthrough

import (
	"context"
	"encoding/json"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
)

const (
	PassthroughParserType = "passthrough-parser"
)

// compile-time type validation
var _ fwkrh.Parser = &PassthroughParser{}

// PassthroughParser implements the fwkrh.Parser interface and does nothing.
type PassthroughParser struct {
	typedName fwkplugin.TypedName
}

// NewPassthroughParser creates a new PassthroughParser.
func NewPassthroughParser() *PassthroughParser {
	return &PassthroughParser{
		typedName: fwkplugin.TypedName{
			Type: PassthroughParserType,
			Name: PassthroughParserType,
		},
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *PassthroughParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *PassthroughParser) SupportedAppProtocols() []v1.AppProtocol {
	return []v1.AppProtocol{}
}

func PassthroughParserPluginFactory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewPassthroughParser().WithName(name), nil
}

func (p *PassthroughParser) WithName(name string) *PassthroughParser {
	p.typedName.Name = name
	return p
}

// ParseRequest converts the request to RawPayload.
func (p *PassthroughParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*fwkrh.InferenceRequestBody, error) {
	return &fwkrh.InferenceRequestBody{
		Payload: fwkrh.RawPayload(body),
	}, nil
}

// ParseResponse does nothing and returns nil.
func (p *PassthroughParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, isEnd bool) (*fwkrh.ParsedResponse, error) {
	return nil, nil
}
