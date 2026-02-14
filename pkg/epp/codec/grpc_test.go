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

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"
	pb "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/api/gen"
	fwkrq "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	types "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestGRPC_ConvertToLLMRequestBody(t *testing.T) {
	tests := []struct {
		name       string
		payload    []byte
		want       *types.LLMRequestBody
		wantStream bool
		wantErr    bool
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
			wantStream: false,
			wantErr:    false,
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
			wantStream: false,
			wantErr:    false,
		},
		{
			name: "Valid Streaming GenerateRequest_Tokenized",
			payload: createFramedPayload(t, &pb.GenerateRequest{
				Input: &pb.GenerateRequest_Tokenized{
					Tokenized: &pb.TokenizedInput{
						OriginalText: "Tokenized Text",
					},
				},
				Stream: true,
			}, false),
			want: &types.LLMRequestBody{
				Completions: &types.CompletionsRequest{
					Prompt: "Tokenized Text",
				},
			},
			wantStream: true,
			wantErr:    false,
		},
		{
			name:       "Invalid framing (too short)",
			payload:    []byte{0, 0, 0, 0},
			want:       nil,
			wantStream: false,
			wantErr:    true,
		},
		{
			name: "Compressed payload (unsupported)",
			payload: createFramedPayload(t, &pb.GenerateRequest{
				Input: &pb.GenerateRequest_Text{
					Text: "Hello, world!",
				},
			}, true),
			want:       nil,
			wantStream: false,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, gotStream, err := ConvertToLLMRequestBody(tt.payload)
			if (err != nil) != tt.wantErr {
				t.Errorf("ConvertToLLMRequestBody() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantStream != gotStream {
				t.Errorf("ConvertToLLMRequestBody() gotStream = %v, wantStream %v", gotStream, tt.wantStream)
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

func TestParseUsage(t *testing.T) {
	tests := []struct {
		name    string
		payload []byte
		want    *fwkrq.Usage
		wantErr bool
	}{
		{
			name: "Valid Complete Response",
			payload: createFramedPayload(t, &pb.GenerateResponse{
				Response: &pb.GenerateResponse_Complete{
					Complete: &pb.GenerateComplete{
						PromptTokens:     10,
						CompletionTokens: 20,
						CachedTokens:     5,
					},
				},
			}, false),
			want: &fwkrq.Usage{
				PromptTokens:     10,
				CompletionTokens: 20,
				TotalTokens:      30,
				PromptTokenDetails: &fwkrq.PromptTokenDetails{
					CachedTokens: 5,
				},
			},
			wantErr: false,
		},
		{
			name:    "Different Response Type (Returns Nil)",
			payload: createFramedPayload(t, &pb.GenerateResponse{}, false),
			want:    nil,
			wantErr: false,
		},
		{
			name:    "Invalid Framing (Too Short)",
			payload: []byte{0, 0, 0, 0},
			want:    nil,
			wantErr: true,
		},
		{
			name: "Compressed Payload (Unsupported)",
			payload: createFramedPayload(t, &pb.GenerateResponse{
				Response: &pb.GenerateResponse_Complete{
					Complete: &pb.GenerateComplete{PromptTokens: 1},
				},
			}, true),
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseUsage(tt.payload)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseUsage() error = %v, wantErr %v", err, tt.wantErr)
			}
			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("ParseUsage() returned unexpected isage, diff(-want, +got): %v", diff)
			}
		})
	}
}

func createFramedPayload(t *testing.T, msg proto.Message, compressed bool) []byte {
	t.Helper()
	data, err := proto.Marshal(msg)
	if err != nil {
		t.Fatalf("Failed to marshal message: %v", err)
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
