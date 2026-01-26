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
	"testing"

	"google.golang.org/protobuf/proto"
	pb "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/api/gen"
	types "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestGRPC_ConvertToLLMRequestBody(t *testing.T) {
	tests := []struct {
		name    string
		payload []byte
		want    *types.LLMRequestBody
		wantErr bool
	}{
		{
			name: "Valid GenerateRequest_Text",
			payload: createFramedPayload(t, &pb.GenerateRequest{
				Input: &pb.GenerateRequest_Text{
					Text: "Hello, world!",
				},
			}, false),
			want: &types.LLMRequestBody{
				Completions: &types.CompletionsRequest{
					Prompt: "Hello, world!",
				},
			},
			wantErr: false,
		},
		{
			name: "Valid GenerateRequest_Tokenized",
			payload: createFramedPayload(t, &pb.GenerateRequest{
				Input: &pb.GenerateRequest_Tokenized{
					Tokenized: &pb.TokenizedInput{
						OriginalText: "Tokenized Text",
					},
				},
			}, false),
			want: &types.LLMRequestBody{
				Completions: &types.CompletionsRequest{
					Prompt: "Tokenized Text",
				},
			},
			wantErr: false,
		},
		{
			name:    "Invalid framing (too short)",
			payload: []byte{0, 0, 0, 0},
			want:    nil,
			wantErr: true,
		},
		{
			name: "Compressed payload (unsupported)",
			payload: createFramedPayload(t, &pb.GenerateRequest{
				Input: &pb.GenerateRequest_Text{
					Text: "Hello, world!",
				},
			}, true),
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ConvertToLLMRequestBody(tt.payload)
			if (err != nil) != tt.wantErr {
				t.Errorf("ConvertToLLMRequestBody() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if got == nil || got.Completions == nil {
					t.Errorf("Expected LLMRequestBody with Completions, got nil")
					return
				}
				if got.Completions.Prompt != tt.want.Completions.Prompt {
					t.Errorf("Expected prompt %q, got %q", tt.want.Completions.Prompt, got.Completions.Prompt)
				}
			}
		})
	}
}

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
