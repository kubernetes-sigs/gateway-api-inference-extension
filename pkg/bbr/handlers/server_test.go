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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins/basemodelextractor"
	envoytest "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/test"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
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
												Key:      ModelHeader,
												RawValue: []byte("foo"),
											},
										},
										{
											Header: &basepb.HeaderValue{
												Key:      BaseModelHeader,
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
												Key:      ModelHeader,
												RawValue: []byte("foo"),
											},
										},
										{
											Header: &basepb.HeaderValue{
												Key:      BaseModelHeader,
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
	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			modelToHeaderPlugin, _ := plugins.NewBodyFieldToHeaderPlugin(ModelField, ModelHeader)
			baseModelToHeaderPlugin := basemodelextractor.NewBaseModelToHeaderPlugin()
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
