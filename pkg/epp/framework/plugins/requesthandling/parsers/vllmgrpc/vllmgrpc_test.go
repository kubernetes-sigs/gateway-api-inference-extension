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
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"

	// Note: Adjust these imports if your local aliases differ
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	pb "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/gen"
)

// helper function to simulate the gRPC payload framing
// [1 byte compression flag] [4 bytes message length] [message bytes...]
func createGrpcPayload(t *testing.T, msg proto.Message) []byte {
	t.Helper()
	b, err := proto.Marshal(msg)
	if err != nil {
		t.Fatalf("failed to marshal proto: %v", err)
	}

	payload := make([]byte, 5+len(b))
	payload[0] = 0 // 0 = uncompressed
	binary.BigEndian.PutUint32(payload[1:5], uint32(len(b)))
	copy(payload[5:], b)

	return payload
}

func TestVllmGRPCParser_PluginLifecycle(t *testing.T) {
	parser := NewVllmGRPCParser()

	wantName := fwkplugin.TypedName{
		Type: VllmGRPCParserType,
		Name: VllmGRPCParserType,
	}
	if diff := cmp.Diff(wantName, parser.TypedName()); diff != "" {
		t.Errorf("TypedName() mismatch (-want +got):\n%s", diff)
	}

	parser.WithName("custom-name")
	wantName.Name = "custom-name"
	if diff := cmp.Diff(wantName, parser.TypedName()); diff != "" {
		t.Errorf("TypedName() mismatch (-want +got):\n%s", diff)
	}

	plugin, err := VllmGRPCParserPluginFactory("factory-name", nil, nil)
	if err != nil {
		t.Fatalf("unexpected error from factory: %v", err)
	}

	p, ok := plugin.(*VllmGRPCParser)
	if !ok {
		t.Fatalf("expected *VllmGRPCParser, got %T", plugin)
	}

	wantName.Name = "factory-name"
	if diff := cmp.Diff(wantName, p.TypedName()); diff != "" {
		t.Errorf("TypedName() mismatch (-want +got):\n%s", diff)
	}
}

func TestVllmGRPCParser_ParseRequest(t *testing.T) {
	tests := []struct {
		name          string
		reqMsg        proto.Message
		headers       map[string]string
		malformedData []byte
		wantErr       bool
		want          *fwkrh.InferenceRequestBody
	}{
		{
			name: "Valid Text Request",
			reqMsg: &pb.GenerateRequest{
				Input: &pb.GenerateRequest_Text{
					Text: "Hello world",
				},
			},
			headers: map[string]string{":path": "/vllm.grpc.engine.VllmEngine/Generate"},
			want: &fwkrh.InferenceRequestBody{
				Completions: &fwkrh.CompletionsRequest{
					Prompt: fwkrh.Prompt{Raw: "Hello world"},
				},
				Payload: fwkrh.PayloadProto{
					Message: &pb.GenerateRequest{
						Input: &pb.GenerateRequest_Text{
							Text: "Hello world",
						},
					},
				},
			},
		},

		{
			name: "Valid Tokenized Request",
			reqMsg: &pb.GenerateRequest{
				Input: &pb.GenerateRequest_Tokenized{
					Tokenized: &pb.TokenizedInput{
						OriginalText: "Tokenized hello",
					},
				},
			},
			headers: map[string]string{":path": "/vllm.grpc.engine.VllmEngine/Generate"},
			want: &fwkrh.InferenceRequestBody{
				Completions: &fwkrh.CompletionsRequest{
					Prompt: fwkrh.Prompt{Raw: "Tokenized hello"},
				},
				Payload: fwkrh.PayloadProto{
					Message: &pb.GenerateRequest{
						Input: &pb.GenerateRequest_Tokenized{
							Tokenized: &pb.TokenizedInput{
								OriginalText: "Tokenized hello",
							},
						},
					}},
			},
		},

		{
			name:          "Malformed gRPC payload (too short)",
			malformedData: []byte{0, 0, 0},
			wantErr:       true,
		},
		{
			name:          "Compressed payload (unsupported)",
			malformedData: []byte{1, 0, 0, 0, 0}, // Flag 1 = compressed
			wantErr:       true,
		},
		{
			name:    "Nil Input Request",
			reqMsg:  &pb.GenerateRequest{},
			wantErr: true,
		},
		{
			name: "Valid Text Request with Stream",
			reqMsg: &pb.GenerateRequest{
				Input: &pb.GenerateRequest_Text{
					Text: "Hello world",
				},
				Stream: true,
			},
			headers: map[string]string{":path": "/vllm.grpc.engine.VllmEngine/Generate"},
			want: &fwkrh.InferenceRequestBody{
				Stream: true,
				Completions: &fwkrh.CompletionsRequest{
					Prompt: fwkrh.Prompt{Raw: "Hello world"},
				},
				Payload: fwkrh.PayloadProto{
					Message: &pb.GenerateRequest{
						Input: &pb.GenerateRequest_Text{
							Text: "Hello world",
						},
						Stream: true,
					}},
			},
		},
		{
			name: "Valid Embed Request",
			reqMsg: &pb.EmbedRequest{
				Tokenized: &pb.TokenizedInput{
					OriginalText: "Embed this",
				},
			},
			headers: map[string]string{":path": "/vllm.grpc.engine.VllmEngine/Embed"},
			want: &fwkrh.InferenceRequestBody{
				Embeddings: &fwkrh.EmbeddingsRequest{
					Input: "Embed this",
				},
				Payload: fwkrh.PayloadProto{
					Message: &pb.EmbedRequest{
						Tokenized: &pb.TokenizedInput{
							OriginalText: "Embed this",
						},
					},
				},
			},
		},
	}

	parser := NewVllmGRPCParser()
	ctx := context.Background()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var payload []byte
			if tt.malformedData != nil {
				payload = tt.malformedData
			} else {
				payload = createGrpcPayload(t, tt.reqMsg)
			}

			got, err := parser.ParseRequest(ctx, payload, tt.headers)

			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseRequest() error = %v, wantErr %v", err, tt.wantErr)
			}

			if tt.wantErr {
				return
			}

			if diff := cmp.Diff(tt.want, got, protocmp.Transform()); diff != "" {
				t.Errorf("ParseRequest() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestVllmGRPCParser_ParseResponse(t *testing.T) {
	tests := []struct {
		name    string
		respMsg proto.Message
		wantErr bool
		want    *fwkrh.ParsedResponse
	}{
		{
			name: "Valid Chunk Response",
			respMsg: &pb.GenerateResponse{
				Response: &pb.GenerateResponse_Chunk{
					Chunk: &pb.GenerateStreamChunk{
						TokenIds:         []uint32{1, 2, 3},
						PromptTokens:     10,
						CompletionTokens: 5,
						CachedTokens:     2,
					},
				},
			},
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     10,
					CompletionTokens: 5,
					TotalTokens:      15,
					PromptTokenDetails: &fwkrh.PromptTokenDetails{
						CachedTokens: 2,
					},
				},
			},
		},

		{
			name: "Valid Complete Response",
			respMsg: &pb.GenerateResponse{
				Response: &pb.GenerateResponse_Complete{
					Complete: &pb.GenerateComplete{
						FinishReason:     "stop",
						PromptTokens:     20,
						CompletionTokens: 15,
						CachedTokens:     5,
					},
				},
			},
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     20,
					CompletionTokens: 15,
					TotalTokens:      35,
					PromptTokenDetails: &fwkrh.PromptTokenDetails{
						CachedTokens: 5,
					},
				},
			},
		},
		{
			name: "Empty Chunk (No Tokens, Streaming intermediate)",
			respMsg: &pb.GenerateResponse{
				Response: &pb.GenerateResponse_Chunk{
					Chunk: &pb.GenerateStreamChunk{
						TokenIds: []uint32{4},
						// PromptTokens and CompletionTokens are 0
					},
				},
			},
			want: &fwkrh.ParsedResponse{
				Usage: nil,
			},
		},
		{
			name:    "Nil Response",
			respMsg: &pb.GenerateResponse{},
			wantErr: true,
		},
		{
			name: "Valid Embed Response",
			respMsg: &pb.EmbedResponse{
				PromptTokens: 10,
				Embedding:    []float32{1.0, 2.0},
				EmbeddingDim: 2,
			},
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     10,
					CompletionTokens: 0,
					TotalTokens:      10,
				},
			},
		},
		{
			name: "Valid Embed Response with Context",
			respMsg: &pb.EmbedResponse{
				PromptTokens: 10,
				Embedding:    []float32{1.0, 2.0},
				EmbeddingDim: 2,
			},
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     10,
					CompletionTokens: 0,
					TotalTokens:      10,
				},
			},
		},
		{
			name: "Valid Generate Response with Context",
			respMsg: &pb.GenerateResponse{
				Response: &pb.GenerateResponse_Complete{
					Complete: &pb.GenerateComplete{
						PromptTokens:     20,
						CompletionTokens: 15,
					},
				},
			},
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     20,
					CompletionTokens: 15,
					TotalTokens:      35,
					PromptTokenDetails: &fwkrh.PromptTokenDetails{
						CachedTokens: 0,
					},
				},
			},
		},
	}

	parser := NewVllmGRPCParser()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			payload := createGrpcPayload(t, tt.respMsg)
			got, err := parser.ParseResponse(context.Background(), payload, nil, false)

			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseResponse() error = %v, wantErr %v", err, tt.wantErr)
			}

			if tt.wantErr {
				return
			}

			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ParseResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestVllmGRPCParser_SupportedAppProtocols(t *testing.T) {
	parser := NewVllmGRPCParser()

	supported := parser.SupportedAppProtocols()
	want := []v1.AppProtocol{v1.AppProtocolH2C}

	if diff := cmp.Diff(want, supported); diff != "" {
		t.Errorf("SupportedAppProtocols() mismatch (-want +got):\n%s", diff)
	}
}
