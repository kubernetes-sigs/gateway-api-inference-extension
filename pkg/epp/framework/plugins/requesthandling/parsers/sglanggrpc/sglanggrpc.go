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

package sglanggrpc

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"

	"google.golang.org/protobuf/proto"

	"sigs.k8s.io/controller-runtime/pkg/log"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	pb "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/requesthandling/parsers/sglanggrpc/api/gen"
)

const (
	// SglangGRPCParserType is the type name for the SGLang gRPC parser.
	SglangGRPCParserType = "sglanggrpc-parser"

	gRPCPayloadHeaderLen = 5

	methodPathKey      = ":path"
	sglangGeneratePath = "/sglang.runtime.v1.SglangService/TextGenerate"
)

// compile-time type validation
var _ fwkrh.Parser = &SglangGRPCParser{}

// SglangGRPCParser implements the fwkrh.Parser interface for SGLang gRPC.
type SglangGRPCParser struct {
	typedName fwkplugin.TypedName
}

// NewSglangGRPCParser creates a new SglangGRPCParser.
func NewSglangGRPCParser() *SglangGRPCParser {
	return &SglangGRPCParser{
		typedName: fwkplugin.TypedName{
			Type: SglangGRPCParserType,
			Name: SglangGRPCParserType,
		},
	}
}

// SglangGRPCParserPluginFactory is the factory function for creating SglangGRPCParser instances.
func SglangGRPCParserPluginFactory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewSglangGRPCParser().WithName(name), nil
}

// WithName sets the name of the plugin instance.
func (p *SglangGRPCParser) WithName(name string) *SglangGRPCParser {
	p.typedName.Name = name
	return p
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *SglangGRPCParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *SglangGRPCParser) SupportedAppProtocols() []v1.AppProtocol {
	return []v1.AppProtocol{v1.AppProtocolH2C}
}

// ParseRequest parses the gRPC request body and headers and returns a InferenceRequestBody.
func (p *SglangGRPCParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*fwkrh.InferenceRequestBody, error) {
	logger := log.FromContext(ctx)

	path := headers[methodPathKey]
	switch path {
	case sglangGeneratePath:
		extractedBody, err := convertToRequestBody(body)
		if err != nil {
			return nil, fmt.Errorf("parsing gRPC payload for TextGenerate: %w", err)
		}
		logger.V(logutil.TRACE).Info("parsed TextGenerateRequest")
		return extractedBody, nil
	default:
		return nil, fmt.Errorf("unsupported gRPC path: %s", path)
	}
}

// ParseResponse parses the response body and returns a ParsedResponse.
func (p *SglangGRPCParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, endofStream bool) (*fwkrh.ParsedResponse, error) {
	logger := log.FromContext(ctx)
	resp := &pb.TextGenerateResponse{}

	if err := toTextGenerateResponse(body, resp); err != nil {
		return nil, fmt.Errorf("failed to parse gRPC response payload as TextGenerateResponse: %w", err)
	}

	result := &fwkrh.ParsedResponse{}

	// Extract usage from meta_info if present.
	// Note: In proto, meta_info is map<string, string>, so token counts are strings.
	if resp.MetaInfo != nil {
		var usage fwkrh.Usage
		if promptTokensStr, ok := resp.MetaInfo["prompt_tokens"]; ok {
			var promptTokens int
			if _, err := fmt.Sscanf(promptTokensStr, "%d", &promptTokens); err == nil {
				usage.PromptTokens = promptTokens
			}
		}
		if completionTokensStr, ok := resp.MetaInfo["completion_tokens"]; ok {
			var completionTokens int
			if _, err := fmt.Sscanf(completionTokensStr, "%d", &completionTokens); err == nil {
				usage.CompletionTokens = completionTokens
			}
		}
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
		result.Usage = &usage
		logger.V(logutil.DEBUG).Info("extracted usage from meta_info", "prompt", usage.PromptTokens, "completion", usage.CompletionTokens)
	}

	return result, nil
}

func toTextGenerateResponse(payload []byte, resp *pb.TextGenerateResponse) error {
	parsedPayload, compressed, err := parseGrpcPayload(payload)
	if err != nil {
		return errors.New("not able to parse payload")
	}
	if compressed {
		return errors.New("compressed sglanggrpc payload is not supported")
	}

	return proto.Unmarshal(parsedPayload, resp)
}

func convertToRequestBody(payload []byte) (*fwkrh.InferenceRequestBody, error) {
	pbReq := &pb.TextGenerateRequest{}
	if err := toTextGenerateRequest(payload, pbReq); err != nil {
		return nil, err
	}

	body := &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{
			Prompt: fwkrh.Prompt{Raw: pbReq.GetText()},
		},
		Payload: fwkrh.PayloadProto{Message: pbReq},
	}
	body.Stream = pbReq.GetStream()
	return body, nil
}

func toTextGenerateRequest(payload []byte, req *pb.TextGenerateRequest) error {
	parsedPayload, compressed, err := parseGrpcPayload(payload)
	if err != nil {
		return errors.New("not able to parse payload")
	}
	if compressed {
		return errors.New("compressed sglanggrpc payload is not supported")
	}

	return proto.Unmarshal(parsedPayload, req)
}

// parseGrpcPayload extracts the message payload and its compression status from a gRPC frame.
func parseGrpcPayload(data []byte) ([]byte, bool, error) {
	if len(data) < gRPCPayloadHeaderLen {
		return nil, false, fmt.Errorf("invalid gRPC frame: expected at least %d bytes for header, got %d", gRPCPayloadHeaderLen, len(data))
	}

	isCompressed := data[0] == 1
	msgLen := binary.BigEndian.Uint32(data[1:5])

	if uint32(len(data)) < gRPCPayloadHeaderLen+msgLen {
		return nil, false, fmt.Errorf("incomplete gRPC payload: header indicates %d bytes, but only %d bytes are available", msgLen, uint32(len(data))-gRPCPayloadHeaderLen)
	}
	return data[gRPCPayloadHeaderLen : gRPCPayloadHeaderLen+msgLen], isCompressed, nil
}
