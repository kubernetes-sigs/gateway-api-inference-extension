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
	"encoding/json"
	"strconv"
	"testing"

	basepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/testing/protocmp"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins/basemodelextractor"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins/bodyfieldtoheader"
	envoytest "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/test"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

func TestHandleRequestBodyStreaming(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	b, _ := json.Marshal(map[string]any{"model": "foo"})
	cases := []struct {
		desc      string
		streaming bool
		body      []byte
		want      []*extProcPb.ProcessingResponse
	}{
		{
			desc: "no-streaming",
			body: b,
			want: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*basepb.HeaderValueOption{
										{
											Header: &basepb.HeaderValue{
												Key:      bodyfieldtoheader.ModelHeader,
												RawValue: []byte("foo"),
											},
										},
										{
											Header: &basepb.HeaderValue{
												Key:      basemodelextractor.BaseModelHeader,
												RawValue: []byte(""),
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			desc:      "streaming",
			streaming: true,
			body:      b,
			want: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestHeaders{
						RequestHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*basepb.HeaderValueOption{
										{
											Header: &basepb.HeaderValue{
												Key:      contentLengthHeader,
												RawValue: []byte(strconv.Itoa(len(b))),
											},
										},
										{
											Header: &basepb.HeaderValue{
												Key:      bodyfieldtoheader.ModelHeader,
												RawValue: []byte("foo"),
											},
										},
										{
											Header: &basepb.HeaderValue{
												Key:      basemodelextractor.BaseModelHeader,
												RawValue: []byte(""),
											},
										},
									},
								},
							},
						},
					},
				},
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        b,
											EndOfStream: true,
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	baseModelToHeaderPlugin := &basemodelextractor.BaseModelToHeaderPlugin{AdaptersStore: basemodelextractor.NewAdaptersStore()}
	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			modelToHeaderPlugin, _ := bodyfieldtoheader.NewBodyFieldToHeaderPlugin(modelField, bodyfieldtoheader.ModelHeader)
			srv := NewServer(tc.streaming, []framework.RequestProcessor{modelToHeaderPlugin, baseModelToHeaderPlugin}, []framework.ResponseProcessor{})
			reqCtx := &RequestContext{
				CycleState: framework.NewCycleState(),
				Request:    framework.NewInferenceRequest(),
			}
			got, err := srv.HandleRequestBody(ctx, reqCtx, tc.body)
			if err != nil {
				t.Fatalf("HandleRequestBody(): %v", err)
			}

			// sort headers in responses for deterministic tests
			envoytest.SortSetHeadersInResponses(tc.want)
			envoytest.SortSetHeadersInResponses(got)
			if diff := cmp.Diff(tc.want, got, protocmp.Transform()); diff != "" {
				t.Errorf("HandleRequestBody returned unexpected response, diff(-want, +got): %v", diff)
			}
		})
	}
}

func TestHandleResponseBody_Streaming(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	wantFullBody := []byte(`{"choices":[{"text":"Hello!"}]}`)

	ref := NewServer(true, []framework.RequestProcessor{}, []framework.ResponseProcessor{})
	want, err := ref.HandleResponseBody(ctx, newTestRequestContext(), wantFullBody)
	if err != nil {
		t.Fatalf("reference HandleResponseBody: %v", err)
	}

	type chunk struct {
		body        []byte
		endOfStream bool
	}
	tests := []struct {
		name   string
		chunks []chunk
	}{
		{
			name: "single chunk with EoS",
			chunks: []chunk{
				{body: wantFullBody, endOfStream: true},
			},
		},
		{
			name: "split JSON across two chunks, EoS on last",
			chunks: []chunk{
				{body: []byte(`{"choices":[{"te`), endOfStream: false},
				{body: []byte(`xt":"Hello!"}]}`), endOfStream: true},
			},
		},
		{
			name: "fragmented: three chunks, EoS on last",
			chunks: []chunk{
				{body: []byte(`{"choices":`), endOfStream: false},
				{body: []byte(`[{"text":"Hello!"}]`), endOfStream: false},
				{body: []byte(`}`), endOfStream: true},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			streamCtx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
			srv := NewServer(true, []framework.RequestProcessor{}, []framework.ResponseProcessor{})
			testListener, errChan := utils.SetupTestStreamingServer(t, streamCtx, srv)
			process, conn := utils.GetStreamingServerClient(streamCtx, t)
			defer conn.Close()
			defer func() {
				cancel()
				<-errChan
				testListener.Close()
			}()

			respHeaders := utils.BuildEnvoyGRPCHeaders(map[string]string{
				"x-test":       "body",
				":method":      "POST",
				"content-type": "text/event-stream",
			}, true)
			request := &extProcPb.ProcessingRequest{
				Request: &extProcPb.ProcessingRequest_ResponseHeaders{
					ResponseHeaders: respHeaders,
				},
			}
			if err := process.Send(request); err != nil {
				t.Fatalf("send response headers: %v", err)
			}

			for _, c := range tc.chunks {
				request = &extProcPb.ProcessingRequest{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{
							Body:        c.body,
							EndOfStream: c.endOfStream,
						},
					},
				}
				if err := process.Send(request); err != nil {
					t.Fatalf("send response body chunk: %v", err)
				}
			}

			got := make([]*extProcPb.ProcessingResponse, 0, len(want))
			for range want {
				msg, err := process.Recv()
				if err != nil {
					t.Fatalf("recv response phase: %v", err)
				}
				got = append(got, msg)
			}

			envoytest.SortSetHeadersInResponses(want)
			envoytest.SortSetHeadersInResponses(got)
			if diff := cmp.Diff(want, got, protocmp.Transform()); diff != "" {
				t.Errorf("unexpected ProcessingResponse after buffered streaming response body, diff(-want, +got): %s", diff)
			}
		})
	}
}
