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
	"encoding/binary"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"
	fwkpayload "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/payloadprocess"
	fwkrc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	vllm "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/payloadprocess/protos/vllm/grpc"
)

func TestVLLMGrpcParser_ParseRequest_Streaming(t *testing.T) {
	parser := NewVLLMGrpcParser()

	tests := []struct {
		name       string
		body       []byte
		wantStream bool
	}{
		{
			name:       "Stream enabled",
			body:       []byte(`{"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": true}`),
			wantStream: true,
		},
		{
			name:       "Stream disabled",
			body:       []byte(`{"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": false}`),
			wantStream: false,
		},
		{
			name:       "Stream omitted (default false)",
			body:       []byte(`{"model": "test", "messages": [{"role": "user", "content": "hi"}]}`),
			wantStream: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			headers := map[string]string{":path": "/v1/chat/completions", "content-type": "application/json"}
			gotBody, err := parser.ParseRequest(headers, tt.body)
			if err != nil {
				t.Fatalf("ParseRequest() error = %v", err)
			}

			// ParseRequest returns the body with a 5-byte gRPC frame
			encoded := gotBody.ParsedBody.([]byte)
			if len(encoded) < 5 {
				t.Fatalf("Encoded body too short: %d", len(encoded))
			}

			// Verify framing (compression=0, length matches)
			if encoded[0] != 0 {
				t.Errorf("Expected compression flag 0, got %d", encoded[0])
			}
			length := binary.BigEndian.Uint32(encoded[1:5])
			if int(length) != len(encoded)-5 {
				t.Errorf("Frame length mismatch: header %d, actual %d", length, len(encoded)-5)
			}

			// Unmarshal options
			vllmReq := &vllm.GenerateRequest{}
			if err := proto.Unmarshal(encoded[5:], vllmReq); err != nil {
				t.Fatalf("Failed to unmarshal vllm request: %v", err)
			}

			if vllmReq.Stream != tt.wantStream {
				t.Errorf("generateRequest.Stream = %v, want %v", vllmReq.Stream, tt.wantStream)
			}
		})
	}
}

func TestVLLMGrpcParser_ParseStreamResponse(t *testing.T) {
	parser := NewVLLMGrpcParser()

	tests := []struct {
		name    string
		resp    *vllm.GenerateResponse
		want    *fwkpayload.ParsedResponse
		wantErr bool
	}{
		{
			name: "StreamChunk with usage",
			resp: &vllm.GenerateResponse{
				Response: &vllm.GenerateResponse_Chunk{
					Chunk: &vllm.GenerateStreamChunk{
						TokenIds:         []uint32{1, 2},
						PromptTokens:     10,
						CompletionTokens: 5,
					},
				},
			},
			want: &fwkpayload.ParsedResponse{
				Usage: &fwkrc.Usage{
					PromptTokens:     10,
					CompletionTokens: 5,
					TotalTokens:      15,
				},
			},
		},
		{
			name: "Complete (final) with usage",
			resp: &vllm.GenerateResponse{
				Response: &vllm.GenerateResponse_Complete{
					Complete: &vllm.GenerateComplete{
						OutputIds:        []uint32{1, 2, 3},
						PromptTokens:     10,
						CompletionTokens: 20,
					},
				},
			},
			want: &fwkpayload.ParsedResponse{
				Usage: &fwkrc.Usage{
					PromptTokens:     10,
					CompletionTokens: 20,
					TotalTokens:      30,
				},
			},
		},
		{
			name: "Chunk without usage (zero values)",
			resp: &vllm.GenerateResponse{
				Response: &vllm.GenerateResponse_Chunk{
					Chunk: &vllm.GenerateStreamChunk{
						TokenIds: []uint32{1},
						// other fields 0
					},
				},
			},
			want: &fwkpayload.ParsedResponse{
				Usage: &fwkrc.Usage{
					PromptTokens:     0,
					CompletionTokens: 0,
					TotalTokens:      0,
				},
			},
		},
		{
			name: "Empty Response (neither chunk nor complete)",
			resp: &vllm.GenerateResponse{},
			// Current implementation returns error for empty message
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := proto.Marshal(tt.resp)
			if err != nil {
				t.Fatalf("Failed to marshal input response: %v", err)
			}

			got, err := parser.ParseStreamResponse(data)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseStreamResponse() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}

			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ParseStreamResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
