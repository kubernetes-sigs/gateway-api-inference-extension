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
	"testing"

	basepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/testing/protocmp"
	"sigs.k8s.io/controller-runtime/pkg/log"

	bbrplugins "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins"
	utils "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/utils"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func TestProcessRequestBody(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	//set environment variables expected by the code under test
	//testing a minimal configuration
	//request plugin chain always contains the default bbr plugin that extracts a model name and sets it on the X-Gateway-Model-Name header
	t.Setenv("REQUEST_PLUGINS_CHAIN", "simple_model_extractor")

	cases := []struct {
		desc      string
		streaming bool
		bodys     []*extProcPb.HttpBody
		want      []*extProcPb.ProcessingResponse
	}{
		{
			desc: "no-streaming",
			bodys: []*extProcPb.HttpBody{
				{
					Body: mapToBytes(t, map[string]any{
						"model": "foo",
					}),
				},
			},
			want: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								// Necessary so that the new headers are used in the routing decision.
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*basepb.HeaderValueOption{
										{
											Header: &basepb.HeaderValue{
												Key:      bbrplugins.ModelHeader,
												RawValue: []byte("foo"),
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
			bodys: []*extProcPb.HttpBody{
				{
					Body: mapToBytes(t, map[string]any{
						"model": "foo",
					}),
				},
				{
					EndOfStream: true,
				},
			},
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
												Key:      bbrplugins.ModelHeader,
												RawValue: []byte("foo"),
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
											Body: mapToBytes(t, map[string]any{
												"model": "foo",
											}),
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

	//Initialize PluginRegistry and request/response PluginsChain instances based on the minimal configuration setting vi env vars
	registry, requestChain, responseChain, err := utils.InitPlugins()
	if err != nil {
		t.Fatalf("processRequestBody(): %v", err)
	}

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			srv := NewServer(tc.streaming, registry, requestChain, responseChain)
			streamedBody := &streamedBody{}
			for i, body := range tc.bodys {
				got, err := srv.processRequestBody(context.Background(), body, streamedBody, log.FromContext(ctx))
				if err != nil {
					t.Fatalf("processRequestBody(): %v", err)
				}

				if i == len(tc.bodys)-1 {
					if diff := cmp.Diff(tc.want, got, protocmp.Transform()); diff != "" {
						t.Errorf("processRequestBody returned unexpected response, diff(-want, +got): %v", diff)
					}
				}
			}
		})
	}
}
