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
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	pb "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/requesthandling/parsers/sglanggrpc/api/gen"
)

// helper function to simulate the gRPC payload framing
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

func TestSglangGRPCParser_PluginLifecycle(t *testing.T) {
	parser := NewSglangGRPCParser()

	wantName := fwkplugin.TypedName{
		Type: SglangGRPCParserType,
		Name: SglangGRPCParserType,
	}
	if diff := cmp.Diff(wantName, parser.TypedName()); diff != "" {
		t.Errorf("TypedName() mismatch (-want +got):\n%s", diff)
	}

	parser.WithName("custom-name")
	wantName.Name = "custom-name"
	if diff := cmp.Diff(wantName, parser.TypedName()); diff != "" {
		t.Errorf("TypedName() mismatch (-want +got):\n%s", diff)
	}

	plugin, err := SglangGRPCParserPluginFactory("factory-name", nil, nil)
	if err != nil {
		t.Fatalf("unexpected error from factory: %v", err)
	}

	p, ok := plugin.(*SglangGRPCParser)
	if !ok {
		t.Fatalf("expected *SglangGRPCParser, got %T", plugin)
	}

	wantName.Name = "factory-name"
	if diff := cmp.Diff(wantName, p.TypedName()); diff != "" {
		t.Errorf("TypedName() mismatch (-want +got):\n%s", diff)
	}
}

func TestSglangGRPCParser_ParseRequest(t *testing.T) {
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
			reqMsg: &pb.TextGenerateRequest{
				Text: "Hello world",
			},
			headers: map[string]string{":path": sglangGeneratePath},
			want: &fwkrh.InferenceRequestBody{
				Completions: &fwkrh.CompletionsRequest{
					Prompt: fwkrh.Prompt{Raw: "Hello world"},
				},
				Payload: fwkrh.PayloadProto{Message: &pb.TextGenerateRequest{
					Text: "Hello world",
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
			name: "Valid Text Request with Stream",
			reqMsg: &pb.TextGenerateRequest{
				Text:   "Hello world",
				Stream: proto.Bool(true),
			},
			headers: map[string]string{":path": sglangGeneratePath},
			want: &fwkrh.InferenceRequestBody{
				Completions: &fwkrh.CompletionsRequest{
					Prompt: fwkrh.Prompt{Raw: "Hello world"},
				},
				Payload: fwkrh.PayloadProto{Message: &pb.TextGenerateRequest{
					Text:   "Hello world",
					Stream: proto.Bool(true),
				}},
				Stream: true,
			},
		},
	}

	parser := NewSglangGRPCParser()
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

func TestSglangGRPCParser_ParseResponse(t *testing.T) {
	tests := []struct {
		name    string
		respMsg proto.Message
		wantErr bool
		want    *fwkrh.ParsedResponse
	}{
		{
			name: "Valid Response with meta_info",
			respMsg: &pb.TextGenerateResponse{
				MetaInfo: map[string]string{
					"prompt_tokens":     "10",
					"completion_tokens": "20",
				},
			},
			want: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{
					PromptTokens:     10,
					CompletionTokens: 20,
					TotalTokens:      30,
				},
			},
		},
		{
			name: "Response without usage",
			respMsg: &pb.TextGenerateResponse{
				Text: "generated text",
			},
			want: &fwkrh.ParsedResponse{
				Usage: nil,
			},
		},
		{
			name:    "Invalid Response (Unmarshal failure)",
			respMsg: nil,
			wantErr: true,
		},
	}

	parser := NewSglangGRPCParser()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var payload []byte
			if tt.respMsg != nil {
				payload = createGrpcPayload(t, tt.respMsg)
			} else {
				payload = []byte{0, 0, 0, 0, 1, 0xFF} // Malformed
			}

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
