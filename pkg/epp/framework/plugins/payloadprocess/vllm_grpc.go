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
	"fmt"
	"strings"

	"github.com/google/uuid"
	"google.golang.org/protobuf/proto"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/payloadprocess"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	vllm "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/payloadprocess/protos/vllm/grpc"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

const (
	VLLMGrpcParserName = "vllm-grpc-parser"
	vllmMaxTokens     = 1024
)

// compile-time type validation
var _ payloadprocess.Parser = &VLLMGrpcParser{}

// VLLMGrpcParser implements the gateway-api-inference-extension parser for vLLM gRPC
type VLLMGrpcParser struct {
	typedName fwkplugin.TypedName
}

// NewVLLMGrpcParser creates a new VLLMGrpcParser.
func NewVLLMGrpcParser() *VLLMGrpcParser {
	return &VLLMGrpcParser{
		typedName: fwkplugin.TypedName{
			Type: payloadprocess.ParserType,
			Name: VLLMGrpcParserName,
		},
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *VLLMGrpcParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// samplingParams is an internal struct to help unmarshal OpenAI sampling parameters
// and stream flag from the request body.
type samplingParams struct {
	MaxTokens        *int     `json:"max_tokens,omitempty"`
	Temperature      *float32 `json:"temperature,omitempty"`
	TopP             *float32 `json:"top_p,omitempty"`
	FrequencyPenalty *float32 `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float32 `json:"presence_penalty,omitempty"`
	N                *int     `json:"n,omitempty"`
	Seed             *int32   `json:"seed,omitempty"`
	Stream           bool     `json:"stream,omitempty"`
	Stop             any      `json:"stop,omitempty"`
}


// ParseRequest transforms an incoming OpenAI HTTP/JSON request into the vLLM gRPC GenerateRequest structure.
// We intercept JSON, populate scheduling context, and prepare the protobuf payload.
func (p *VLLMGrpcParser) ParseRequest(headers map[string]string, body []byte) (*scheduling.LLMRequestBody, error) {
	// Extract standard fields usable for scheduling decisions (CompletionsRequest, etc.)
	extractedBody, err := requtil.ExtractRequestBody(body, headers)
	if err != nil {
		return nil, err
	}

	vllmReq, err := p.TranscodeJsonToGrpc(headers, body, extractedBody)
	if err != nil {
		return nil, err
	}

	// Store protobuf back into LLMRequestBody so it can be forwarded
	protoBody, err := proto.Marshal(vllmReq)
	if err != nil {
		return nil, fmt.Errorf("error marshaling proto: %v", err)
	}
	extractedBody.ParsedBody = protoBody

	return extractedBody, nil
}



// ParseResponse parses a standard unary vLLM gRPC response.
func (p *VLLMGrpcParser) ParseResponse(body []byte) (*payloadprocess.ParsedResponse, error) {
	resp := &vllm.GenerateResponse{}
	if err := proto.Unmarshal(body, resp); err != nil {
		return nil, fmt.Errorf("error unmarshalling gRPC GenerateResponse: %v", err)
	}

	complete := resp.GetComplete()
	if complete == nil {
		return nil, errors.New("unary response did not contain Complete block")
	}

	usage := &requestcontrol.Usage{
		PromptTokens:     int(complete.PromptTokens),
		CompletionTokens: int(complete.CompletionTokens),
		TotalTokens:      int(complete.PromptTokens + complete.CompletionTokens),
	}

	return &payloadprocess.ParsedResponse{
		Usage: usage,
	}, nil
}

// ParseStreamResponse intercepts chunks of a gRPC stream.
func (p *VLLMGrpcParser) ParseStreamResponse(chunk []byte) (*payloadprocess.ParsedResponse, error) {
	resp := &vllm.GenerateResponse{}
	if err := proto.Unmarshal(chunk, resp); err != nil {
		return nil, fmt.Errorf("error unmarshalling chunk: %v", err)
	}

	// In streaming scenarios, intermediate increments come in `chunk`, and usage stats accumulate.
	// Or sometimes they only come in the terminal `complete` block.
	if complete := resp.GetComplete(); complete != nil {
		usage := &requestcontrol.Usage{
			PromptTokens:     int(complete.PromptTokens),
			CompletionTokens: int(complete.CompletionTokens),
			TotalTokens:      int(complete.PromptTokens + complete.CompletionTokens),
		}
		return &payloadprocess.ParsedResponse{Usage: usage}, nil
	}

	if ch := resp.GetChunk(); ch != nil {
		// Just returning parsed chunk increments if they don't hold the total usage stats.
		// If vLLM populates incremental totals per chunk, we can parse it here.
		return &payloadprocess.ParsedResponse{
			Usage: &requestcontrol.Usage{
				PromptTokens:     int(ch.PromptTokens),
				CompletionTokens: int(ch.CompletionTokens),
				TotalTokens:      int(ch.PromptTokens + ch.CompletionTokens),
			},
		}, nil
	}

	return nil, errors.New("unable to parse usage from stream chunk")
}

// TranscodeJsonToGrpc transforms OpenAI fields to vLLM gRPC protobuf.
// It is exposed for testing purposes or internal usage.
// TranscodeJsonToGrpc transforms OpenAI fields to vLLM gRPC protobuf.
// It is exposed for testing purposes or internal usage.
func (p *VLLMGrpcParser) TranscodeJsonToGrpc(headers map[string]string, body []byte, extractedBody *scheduling.LLMRequestBody) (*vllm.GenerateRequest, error) {
	vllmReq := &vllm.GenerateRequest{}

	vllmReq.RequestId = ExtractRequestID(headers)

	prompt, err := ExtractCombinedPrompt(extractedBody)
	if err != nil {
		return nil, err
	}
	vllmReq.Input = &vllm.GenerateRequest_Text{
		Text: prompt,
	}

	samplingParams, stream, err := ParseSamplingParams(body)
	if err != nil {
		return nil, err
	}
	vllmReq.SamplingParams = samplingParams

	if stream {
		return nil, errors.New("streaming is not yet implemented for vLLM gRPC")
	}
	vllmReq.Stream = stream

	return vllmReq, nil
}

func ExtractRequestID(headers map[string]string) string {
	if reqId, ok := headers[requtil.RequestIdHeaderKey]; ok {
		return reqId
	}
	return uuid.NewString()
}

func ExtractCombinedPrompt(extractedBody *scheduling.LLMRequestBody) (string, error) {
	if extractedBody.ChatCompletions == nil {
		return "", errors.New("Not implemented. vLLM gRPC parser expects a chat completions request")
	}

	var combinedPrompt strings.Builder
	for _, msg := range extractedBody.ChatCompletions.Messages {
		combinedPrompt.WriteString(msg.Content.PlainText() + "\n")
	}
	return combinedPrompt.String(), nil
}

func ParseSamplingParams(body []byte) (*vllm.SamplingParams, bool, error) {
	var params samplingParams
	if err := json.Unmarshal(body, &params); err != nil {
		return nil, false, fmt.Errorf("error unmarshalling sampling params: %v", err)
	}

	sp := &vllm.SamplingParams{}

	// Default values
	sp.MaxTokens = proto.Uint32(vllmMaxTokens)
	sp.TopP = 1.0
	sp.N = 1

	if params.MaxTokens != nil {
		sp.MaxTokens = proto.Uint32(uint32(*params.MaxTokens))
	}
	if params.Temperature != nil {
		sp.Temperature = proto.Float32(*params.Temperature)
	}
	if params.TopP != nil {
		sp.TopP = *params.TopP
	}
	if params.FrequencyPenalty != nil {
		sp.FrequencyPenalty = *params.FrequencyPenalty
	}
	if params.PresencePenalty != nil {
		sp.PresencePenalty = *params.PresencePenalty
	}
	if params.N != nil {
		sp.N = uint32(*params.N)
	}
	if params.Seed != nil {
		sp.Seed = proto.Int32(*params.Seed)
	}

	if params.Stop != nil {
		switch v := params.Stop.(type) {
		case string:
			sp.Stop = []string{v}
		case []any:
			for _, item := range v {
				if s, ok := item.(string); ok {
					sp.Stop = append(sp.Stop, s)
				}
			}
		}
	}

	return sp, params.Stream, nil
}
