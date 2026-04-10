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

package vllmgrpc

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
	pb "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/gen"
)

const (
	VllmGRPCParserType = "vllmgrpc-parser"

	gRPCPayloadHeaderLen = 5

	methodPathKey    = ":path"
	vllmGeneratePath = "/vllm.grpc.engine.VllmEngine/Generate"
	vllmEmbedPath    = "/vllm.grpc.engine.VllmEngine/Embed"
)

// compile-time type validation
var _ fwkrh.Parser = &VllmGRPCParser{}

// VllmGRPCParser implements the fwkrh.Parser interface for vLLM gRPC.
type VllmGRPCParser struct {
	typedName fwkplugin.TypedName
}

// NewVllmGRPCParser creates a new VllmGRPCParser.
func NewVllmGRPCParser() *VllmGRPCParser {
	return &VllmGRPCParser{
		typedName: fwkplugin.TypedName{
			Type: VllmGRPCParserType,
			Name: VllmGRPCParserType,
		},
	}
}

func VllmGRPCParserPluginFactory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewVllmGRPCParser().WithName(name), nil
}

func (p *VllmGRPCParser) WithName(name string) *VllmGRPCParser {
	p.typedName.Name = name
	return p
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *VllmGRPCParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *VllmGRPCParser) SupportedAppProtocols() []v1.AppProtocol {
	return []v1.AppProtocol{v1.AppProtocolH2C}
}

// ParseRequest parses the gRPC request body and headers and returns an LLMRequestBody.
func (p *VllmGRPCParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*fwkrh.InferenceRequestBody, error) {
	logger := log.FromContext(ctx)

	path := headers[methodPathKey]
	switch path {
	case vllmEmbedPath:
		extractedBody, err := convertEmbedToInferenceRequestBody(body)
		if err != nil {
			return nil, fmt.Errorf("parsing gRPC payload for Embed: %w", err)
		}
		logger.V(logutil.TRACE).Info("parsed EmbedRequest")
		return extractedBody, nil
	case vllmGeneratePath:
		extractedBody, err := convertToInferenceRequestBody(body)
		if err != nil {
			return nil, fmt.Errorf("parsing gRPC payload for Generate: %w", err)
		}
		logger.V(logutil.TRACE).Info("parsed GenerateRequest")
		return extractedBody, nil
	default:
		return nil, fmt.Errorf("unsupported gRPC path: %s", headers[":path"])
	}
}

// ParseResponse parses the response body and returns a ParsedResponse
func (p *VllmGRPCParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, endofStream bool) (*fwkrh.ParsedResponse, error) {
	logger := log.FromContext(ctx)
	resp := &pb.GenerateResponse{}

	// Try to parse as GenerateResponse first. If it fails or the response field is nil,
	// try to parse as EmbedResponse.
	if err := toGenerateResponse(body, resp); err != nil || resp.Response == nil {
		embedResp := &pb.EmbedResponse{}
		if err := toEmbedResponse(body, embedResp); err == nil && (len(embedResp.Embedding) > 0 || embedResp.PromptTokens > 0) {
			logger.V(logutil.DEBUG).Info("parsed EmbedResponse", "promptTokens", embedResp.PromptTokens)
			result := &fwkrh.ParsedResponse{}
			if embedResp.PromptTokens > 0 {
				result.Usage = &fwkrh.Usage{
					PromptTokens:     int(embedResp.PromptTokens),
					CompletionTokens: 0,
					TotalTokens:      int(embedResp.PromptTokens),
				}
			}
			return result, nil
		}
		return nil, fmt.Errorf("failed to parse gRPC response payload as GenerateResponse or EmbedResponse: %w", err)
	}

	result := &fwkrh.ParsedResponse{}

	switch v := resp.Response.(type) {
	case *pb.GenerateResponse_Chunk:
		logger.V(logutil.DEBUG).Info("parsed GenerateResponse_Chunk", "tokenLength", len(v.Chunk.TokenIds))
		// Only populate Usage if the chunk actually contains token data.
		// Streaming chunks often leave this empty until the final chunk.
		promptToken, completionToken, cahcedToken := int(v.Chunk.PromptTokens), int(v.Chunk.CompletionTokens), int(v.Chunk.CachedTokens)
		if promptToken > 0 || completionToken > 0 {
			result.Usage = requestControlUsage(promptToken, completionToken, cahcedToken)
		}

	case *pb.GenerateResponse_Complete:
		logger.V(logutil.DEBUG).Info("parsed GenerateResponse_Complete", "finishReason", v.Complete.FinishReason)
		// Populate Usage for complete, non-streaming responses.
		promptToken, completionToken, cahcedToken := int(v.Complete.PromptTokens), int(v.Complete.CompletionTokens), int(v.Complete.CachedTokens)
		if promptToken > 0 || completionToken > 0 {
			result.Usage = requestControlUsage(promptToken, completionToken, cahcedToken)
		}

	default:
		return nil, errors.New("unrecognized response type in GenerateResponse")
	}

	return result, nil
}

func requestControlUsage(promptToken, completionToken, cachedToken int) *fwkrh.Usage {
	return &fwkrh.Usage{
		PromptTokens:     promptToken,
		CompletionTokens: completionToken,
		TotalTokens:      promptToken + completionToken,
		PromptTokenDetails: &fwkrh.PromptTokenDetails{
			CachedTokens: cachedToken,
		},
	}
}

func toGenerateResponse(payload []byte, resp *pb.GenerateResponse) error {
	parsedPayload, compressed, err := parseGrpcPayload(payload)
	if err != nil {
		return errors.New("not able to parse payload")
	}
	if compressed {
		// TODO(#2635): handle compressed payload.
		return errors.New("compressed vllmgrpc payload is not supported")
	}

	return proto.Unmarshal(parsedPayload, resp)
}

func convertToInferenceRequestBody(payload []byte) (*fwkrh.InferenceRequestBody, error) {
	pbReq := &pb.GenerateRequest{}
	if err := toGenerateRequest(payload, pbReq); err != nil {
		return nil, err
	}
	var body *fwkrh.InferenceRequestBody
	switch pbReq.Input.(type) {
	case *pb.GenerateRequest_Text:
		body = &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Raw: pbReq.GetText()},
			},
			Payload: fwkrh.PayloadProto{Message: pbReq},
		}
	case *pb.GenerateRequest_Tokenized:
		body = &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Raw: pbReq.GetTokenized().OriginalText},
			},
			Payload: fwkrh.PayloadProto{Message: pbReq},
		}
	default:
		return nil, errors.New("not supported request inputType")
	}
	body.Stream = pbReq.GetStream()
	return body, nil
}

// parseGrpcPayload extracts the message payload and its compression status from a gRPC frame.
// A standard gRPC frame consists of a 1-byte compression flag, a 4-byte message length,
// and the actual message payload.
func parseGrpcPayload(data []byte) ([]byte, bool, error) {
	if len(data) < gRPCPayloadHeaderLen {
		return nil, false, fmt.Errorf("invalid gRPC frame: expected at least %d bytes for header, got %d", gRPCPayloadHeaderLen, len(data))
	}

	// gRPC frame header: [Compression Flag (1 byte)] [Message Length (4 bytes)]
	// Compression Flag 0 = uncompressed, 1 = compressed
	isCompressed := data[0] == 1
	msgLen := binary.BigEndian.Uint32(data[1:5])

	if uint32(len(data)) < gRPCPayloadHeaderLen+msgLen {
		return nil, false, fmt.Errorf("incomplete gRPC payload: header indicates %d bytes, but only %d bytes are available", msgLen, uint32(len(data))-gRPCPayloadHeaderLen)
	}
	return data[gRPCPayloadHeaderLen : gRPCPayloadHeaderLen+msgLen], isCompressed, nil
}

func toGenerateRequest(payload []byte, req *pb.GenerateRequest) error {
	parsedPayload, compressed, err := parseGrpcPayload(payload)
	if err != nil {
		return errors.New("not able to parse payload")
	}
	if compressed {
		// TODO(#2635): handle compressed payload.
		return errors.New("compressed vllmgrpc payload is not supported")
	}

	return proto.Unmarshal(parsedPayload, req)
}

func convertEmbedToInferenceRequestBody(payload []byte) (*fwkrh.InferenceRequestBody, error) {
	pbReq := &pb.EmbedRequest{}
	if err := toEmbedRequest(payload, pbReq); err != nil {
		return nil, err
	}
	var body *fwkrh.InferenceRequestBody
	if pbReq.Tokenized != nil {
		body = &fwkrh.InferenceRequestBody{
			Embeddings: &fwkrh.EmbeddingsRequest{
				Input: pbReq.GetTokenized().OriginalText,
			},
			Payload: fwkrh.PayloadProto{Message: pbReq},
		}
	} else {
		return nil, errors.New("missing tokenized input in EmbedRequest")
	}
	return body, nil
}

func toEmbedRequest(payload []byte, req *pb.EmbedRequest) error {
	parsedPayload, compressed, err := parseGrpcPayload(payload)
	if err != nil {
		return errors.New("not able to parse payload")
	}
	if compressed {
		return errors.New("compressed vllmgrpc payload is not supported")
	}

	return proto.Unmarshal(parsedPayload, req)
}

func toEmbedResponse(payload []byte, resp *pb.EmbedResponse) error {
	parsedPayload, compressed, err := parseGrpcPayload(payload)
	if err != nil {
		return errors.New("not able to parse payload")
	}
	if compressed {
		return errors.New("compressed vllmgrpc payload is not supported")
	}

	return proto.Unmarshal(parsedPayload, resp)
}
