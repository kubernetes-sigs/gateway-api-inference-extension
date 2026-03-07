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
	"context"
	"encoding/binary"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"
	pb "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/api/gen"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

func TestNewVllmGrpcParser(t *testing.T) {
	parser := NewVllmGRPCParser()

	expectedName := fwkplugin.TypedName{
		Type: VllmGRPCParserType,
		Name: VllmGRPCParserType,
	}

	if diff := cmp.Diff(expectedName, parser.TypedName()); diff != "" {
		t.Errorf("TypedName() mismatch (-want +got):\n%s", diff)
	}
}

func TestVllmGRPCParser_ParseRequest(t *testing.T) {
	parser := NewVllmGRPCParser()

	tests := []struct {
		name       string
		headers    map[string]string
		body       []byte
		wantErr    bool
		wantPrompt string
	}{
		{
			name:    "Valid GenerateRequest_Text",
			headers: map[string]string{"content-type": "application/grpc"},
			body: createFramedPayload(t, &pb.GenerateRequest{
				Input: &pb.GenerateRequest_Text{
					Text: "Hello, world!",
				},
			}, false),
			wantErr:    false,
			wantPrompt: "Hello, world!",
		},
		{
			name:    "Valid GenerateRequest_Tokenized",
			headers: map[string]string{"content-type": "application/grpc"},
			body: createFramedPayload(t, &pb.GenerateRequest{
				Input: &pb.GenerateRequest_Tokenized{
					Tokenized: &pb.TokenizedInput{
						OriginalText: "Tokenized Text",
					},
				},
			}, false),
			wantErr:    false,
			wantPrompt: "Tokenized Text",
		},
		{
			name:    "Invalid framing (too short)",
			headers: map[string]string{"content-type": "application/grpc"},
			body:    []byte{0, 0, 0, 0},
			wantErr: true,
		},
		{
			name:    "Compressed payload (unsupported)",
			headers: map[string]string{"content-type": "application/grpc"},
			body: createFramedPayload(t, &pb.GenerateRequest{
				Input: &pb.GenerateRequest_Text{
					Text: "Hello, world!",
				},
			}, true),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parser.ParseRequest(context.Background(), tt.body, tt.headers)

			if (err != nil) != tt.wantErr {
				t.Errorf("ParseRequest() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if got == nil || got.Completions == nil {
					t.Errorf("Expected LLMRequestBody with Completions, got nil")
					return
				}
				if got.Completions.Prompt != tt.wantPrompt {
					t.Errorf("Expected prompt %q, got %q", tt.wantPrompt, got.Completions.Prompt)
				}
			}
		})
	}
}

func TestVllmGRPCParser_ParseResponse(t *testing.T) {
	parser := NewVllmGRPCParser()

	got, err := parser.ParseResponse(context.Background(), []byte{}, map[string]string{}, true)
	if got != nil || err != nil {
		t.Errorf("ParseResponse got non-nil: %v, got non-nil err %v", got, err)
	}
}

// Helper function to create gRPC framed payload for testing
func createFramedPayload(t *testing.T, req *pb.GenerateRequest, compressed bool) []byte {
	t.Helper()
	data, err := proto.Marshal(req)
	if err != nil {
		t.Fatalf("Failed to marshal request: %v", err)
	}
	frame := make([]byte, 5+len(data))
	if compressed {
		frame[0] = 1
	} else {
		frame[0] = 0
	}
	binary.BigEndian.PutUint32(frame[1:5], uint32(len(data)))
	copy(frame[5:], data)
	return frame
}
