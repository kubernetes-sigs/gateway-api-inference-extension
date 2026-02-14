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

package codec

import (
	"encoding/binary"
	"errors"
	"log"

	"google.golang.org/protobuf/proto"
	pb "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/api/gen"
	types "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// ConvertToLLMRequestBody converts gRPC payload to LLMRequestBody used in scheduling.
func ConvertToLLMRequestBody(payload []byte) (*types.LLMRequestBody, error) {
	pbReq, err := toGenerateRequest(payload)
	if err != nil {
		return nil, err
	}
	if pbReq == nil {
		return nil, err
	}
	switch pbReq.Input.(type) {
	case *pb.GenerateRequest_Text:
		return &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: pbReq.GetText(),
			},
		}, nil
	case *pb.GenerateRequest_Tokenized:
		return &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: pbReq.GetTokenized().OriginalText,
			},
		}, nil
	}
	return nil, errors.New("not supported request inputType")
}

// parseGrpcPayload extracts the payload and the compression status.
//
// Returns: (payload, isCompressed, success)
func parseGrpcPayload(data []byte) ([]byte, bool, bool) {
	if len(data) < 5 {
		return nil, false, false
	}

	// gRPC frame header: [Compression Flag (1 byte)] [Message Length (4 bytes)]
	// Compression Flag 0 = uncompressed, 1 = compressed
	isCompressed := data[0] == 1
	msgLen := binary.BigEndian.Uint32(data[1:5])

	if uint32(len(data)) < 5+msgLen {
		return nil, false, false
	}
	return data[5 : 5+msgLen], isCompressed, true
}

func toGenerateRequest(payload []byte) (*pb.GenerateRequest, error) {
	parsedPayload, compressed, ok := parseGrpcPayload(payload)
	if !ok {
		return nil, errors.New("not able to parse payload")
	}
	if compressed {
		// TODO: handle compressed payload.
		return nil, errors.New("not able to parse compressed payload")
	}
	req := &pb.GenerateRequest{}
	err := proto.Unmarshal(parsedPayload, req)
	if err != nil {
		return nil, err
	}
	switch v := req.Input.(type) {
	case *pb.GenerateRequest_Text:
		originalText := req.GetText()
		log.Printf("[Parser] Success: GenerateRequest originalText: %s\n", originalText)
	case *pb.GenerateRequest_Tokenized:
		originalText := v.Tokenized.OriginalText
		log.Printf("[Parser] Success: GenerateRequest originalText: %s\n", originalText)
	}
	return req, nil
}
