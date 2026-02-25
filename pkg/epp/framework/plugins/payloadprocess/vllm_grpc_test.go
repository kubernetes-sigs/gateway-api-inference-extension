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
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	vllm "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/payloadprocess/protos/vllm/grpc"
)

const (
	chatCompletionsPath = "/v1/chat/completions"
)

func TestParseRequest(t *testing.T) {
	tests := []struct {
		name    string
		headers map[string]string
		body    map[string]any
		want    *vllm.GenerateRequest
		wantErr bool
	}{
		{
			name:    "Basic Chat Completion",
			headers: map[string]string{":path": chatCompletionsPath},
			body: map[string]any{
				"model": "llama-2", // Model field is required by extract libs but not used by vLLM directly usually, but we check extracted body
				"messages": []map[string]string{
					{"role": "user", "content": "Hello"},
				},
				"max_tokens": float64(100),
			},
			want: &vllm.GenerateRequest{
				Input: &vllm.GenerateRequest_Text{
					Text: "Hello\n",
				},
				SamplingParams: &vllm.SamplingParams{
					MaxTokens: ptrUint32(100),
					TopP:      1.0,
					N:         1,
				},
				Stream: false,
			},
		},
		{
			name:    "With Sampling Params",
			headers: map[string]string{":path": chatCompletionsPath},
			body: map[string]any{
				"model": "llama-2",
				"messages": []map[string]string{
					{"role": "user", "content": "Hello"},
				},
				"max_tokens":        float64(50),
				"temperature":       float64(0.7),
				"top_p":             float64(0.9),
				"frequency_penalty": float64(0.5),
				"presence_penalty":  float64(0.5),
				"n":                 float64(2),
				"seed":              float64(42),
				"stop":              "STOP",
			},
			want: &vllm.GenerateRequest{
				Input: &vllm.GenerateRequest_Text{
					Text: "Hello\n",
				},
				SamplingParams: &vllm.SamplingParams{
					MaxTokens:        ptrUint32(50),
					Temperature:      ptrFloat32(0.7),
					TopP:             0.9,
					FrequencyPenalty: 0.5,
					PresencePenalty:  0.5,
					N:                2,
					Seed:             ptrInt32(42),
					Stop:             []string{"STOP"},
				},
				Stream: false,
			},
		},
		{
			name:    "Stop Sequence List",
			headers: map[string]string{":path": chatCompletionsPath},
			body: map[string]any{
				"model": "gpt-4",
				"messages": []map[string]string{
					{"role": "user", "content": "Hello World!"},
				},
				"stop": []any{"STOP"},
			},
			want: &vllm.GenerateRequest{
				Input: &vllm.GenerateRequest_Text{
					Text: "Hello World!\n",
				},
				SamplingParams: &vllm.SamplingParams{
					MaxTokens: ptrUint32(1024), // Default
					TopP:      1.0,             // Default
					N:         1,               // Default
					Stop:      []string{"STOP"},
				},
				Stream: false,
			},
		},
		{
			name:    "Streaming Not Implemented",
			headers: map[string]string{":path": chatCompletionsPath},
			body: map[string]any{
				"model": "gpt-4",
				"messages": []map[string]string{
					{"role": "user", "content": "Hi"},
				},
				"stream": true,
			},
			wantErr: true,
		},
	}

	parser := NewVLLMGrpcParser()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bodyBytes, _ := json.Marshal(tt.body)
			got, err := parser.ParseRequest(tt.headers, bodyBytes)

			if tt.wantErr {
				if err == nil {
					t.Errorf("ParseRequest() error = nil, wantErr %v", tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseRequest() error = %v, wantErr %v", err, tt.wantErr)
			}

			// We need to unmarshal the ParsedBody proto to check it
			gotProto := &vllm.GenerateRequest{}
			if err := startProtoUnmarshal(got.ParsedBody.([]byte), gotProto); err != nil {
				t.Fatalf("Failed to unmarshal parsed body: %v", err)
			}

			// Ignore RequestID for comparison as it is random if not provided
			if diff := cmp.Diff(tt.want, gotProto, protocmp.Transform(), protocmp.IgnoreFields(&vllm.GenerateRequest{}, "request_id")); diff != "" {
				t.Errorf("ParseRequest() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestParseResponse(t *testing.T) {
	parser := NewVLLMGrpcParser()

	tests := []struct {
		name    string
		body    *vllm.GenerateResponse
		want    *requestcontrol.Usage
		wantErr bool
	}{
		{
			name: "Successful Response",
			body: &vllm.GenerateResponse{
				Response: &vllm.GenerateResponse_Complete{
					Complete: &vllm.GenerateComplete{
						PromptTokens:     10,
						CompletionTokens: 20,
						FinishReason:     "stop",
					},
				},
			},
			want: &requestcontrol.Usage{
				PromptTokens:     10,
				CompletionTokens: 20,
				TotalTokens:      30,
			},
		},
		{
			name:    "Missing Complete Block",
			body:    &vllm.GenerateResponse{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bodyBytes, _ := proto.Marshal(tt.body)
			got, err := parser.ParseResponse(bodyBytes)

			if tt.wantErr {
				if err == nil {
					t.Errorf("ParseResponse() error = nil, wantErr %v", tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseResponse() error = %v, wantErr %v", err, tt.wantErr)
			}

			if diff := cmp.Diff(tt.want, got.Usage); diff != "" {
				t.Errorf("ParseResponse() usage mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// Helpers
func ptrUint32(v uint32) *uint32    { return &v }
func ptrFloat32(v float32) *float32 { return &v }
func ptrInt32(v int32) *int32       { return &v }

func startProtoUnmarshal(b []byte, m *vllm.GenerateRequest) error {
	// The ParsedBody is []byte (marshalled proto)
	// But in the code: `extractedBody.ParsedBody = protoBody` where protoBody is []byte.
	// `extractedBody.ParsedBody` is interface{}.
	// We cast it in the test.
	// But wait, `extractedBody.ParsedBody` is `any`. In `vllm_grpc.go`, we define it as `protoBody`.
	// Check `startProtoUnmarshal` usage.
	return (proto.UnmarshalOptions{}).Unmarshal(b, m)
}
