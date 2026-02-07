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

package handlers

import (
	"context"
	"encoding/binary"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/util/logging"
	pb "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/api/gen"
	fwkrq "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

func TestHandleGRPCRequestBody(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	tests := []struct {
		name    string
		reqPath string
		body    []byte
		want    *schedulingtypes.LLMRequestBody
		wantErr bool
	}{
		{
			name:    "non-matching path",
			reqPath: "/some/other/path",
			body:    []byte("some data"),
			want:    nil,
			wantErr: false,
		},
		{
			name:    "matching path, valid body",
			reqPath: VllmGeneratePath,
			body:    createFramedRequest("hello world"),
			want: &schedulingtypes.LLMRequestBody{
				Completions: &schedulingtypes.CompletionsRequest{
					Prompt: "hello world",
				},
			},
			wantErr: false,
		},
		{
			name:    "matching path, invalid body (short)",
			reqPath: VllmGeneratePath,
			body:    []byte{0, 0, 0, 0}, // < 5 bytes
			want:    nil,
			wantErr: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := &StreamingServer{}
			reqCtx := &RequestContext{
				ReqPath: test.reqPath,
			}

			err := server.handleGRPCRequestBody(ctx, reqCtx, test.body)
			if (err != nil) != test.wantErr {
				t.Errorf("handleGRPCRequestBody() error = %v, wantErr %v", err, test.wantErr)
				return
			}
			if diff := cmp.Diff(test.want, reqCtx.SchedulingRequestBody); diff != "" {
				t.Errorf("handleGRPCRequestBody() mismatch (-want +got): %v", diff)
			}
		})
	}
}

func TestHandleGRPCResponseTrailers(t *testing.T) {
	server := &StreamingServer{}
	reqCtx := &RequestContext{}
	body := []byte("response body")

	server.handleGRPCResponseTrailers(reqCtx, body)

	if reqCtx.respBodyResp == nil {
		t.Error("handleGRPCResponseTrailers() expected respBodyResp to be set")
	}
	if reqCtx.respTrailerResp == nil {
		t.Error("handleGRPCResponseTrailers() expected respTrailerResp to be set")
	}
}

func TestHandleGRPCResponseBodyModelStreaming(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	tests := []struct {
		name   string
		resp   *pb.GenerateResponse
		reqCtx *RequestContext
		want   fwkrq.Usage
	}{
		{
			name: "complete response",
			resp: &pb.GenerateResponse{
				Response: &pb.GenerateResponse_Complete{
					Complete: &pb.GenerateComplete{
						CompletionTokens: 4,
						PromptTokens:     3,
						CachedTokens:     2,
						FinishReason:     "stop",
					},
				},
			},
			reqCtx: &RequestContext{
				modelServerStreaming: true,
				RespContentType:      request.GRPCContentType,
			},
			want: fwkrq.Usage{
				PromptTokens:     3,
				TotalTokens:      7,
				CompletionTokens: 4,
				PromptTokenDetails: &fwkrq.PromptTokenDetails{
					CachedTokens: 2,
				},
			},
		},
		{
			name: "chunk response",
			resp: &pb.GenerateResponse{
				Response: &pb.GenerateResponse_Chunk{},
			},
			reqCtx: &RequestContext{
				modelServerStreaming: true,
				RespContentType:      request.GRPCContentType,
			},
			want: fwkrq.Usage{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := &StreamingServer{}
			server.director = &mockDirector{}
			reqCtx := test.reqCtx
			if reqCtx == nil {
				reqCtx = &RequestContext{}
			}
			payload := toGRPCFrame(test.resp)
			server.HandleResponseBodyModelStreaming(ctx, reqCtx, payload)

			if diff := cmp.Diff(test.want, reqCtx.Usage); diff != "" {
				t.Errorf("HandleResponseBody returned unexpected response, diff(-want, +got): %v", diff)
			}
		})
	}
}

func createFramedRequest(text string) []byte {
	req := &pb.GenerateRequest{
		Input: &pb.GenerateRequest_Text{
			Text: text,
		},
	}
	return toGRPCFrame(req)
}

func toGRPCFrame(m proto.Message) []byte {
	data, _ := proto.Marshal(m)
	buf := make([]byte, 5+len(data))
	buf[0] = 0 // Uncompressed
	binary.BigEndian.PutUint32(buf[1:], uint32(len(data)))
	copy(buf[5:], data)
	return buf
}
