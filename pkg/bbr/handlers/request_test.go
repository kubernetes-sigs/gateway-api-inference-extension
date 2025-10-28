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
	"strings"
	"testing"

	basepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/testing/protocmp"
	metricsutils "k8s.io/component-base/metrics/testutil"
	crmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func TestHandleRequestBody(t *testing.T) {
	metrics.Register()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	tests := []struct {
		name      string
		body      map[string]any
		streaming bool
		want      []*extProcPb.ProcessingResponse
		wantErr   bool
	}{
		{
			name: "model not found",
			body: map[string]any{
				"prompt": "Tell me a joke",
			},
			want: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{},
					},
				},
			},
		},
		{
			name: "model not found with streaming",
			body: map[string]any{
				"prompt": "Tell me a joke",
			},
			streaming: true,
			want: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestHeaders{
						RequestHeaders: &extProcPb.HeadersResponse{},
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
												"prompt": "Tell me a joke",
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
		{
			name: "model is not string",
			body: map[string]any{
				"model":  1,
				"prompt": "Tell me a joke",
			},
			wantErr: true,
		},
		{
			name: "success",
			body: map[string]any{
				"model":  "foo",
				"prompt": "Tell me a joke",
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
												Key:      "X-Gateway-Model-Name",
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
			name: "success-with-streaming",
			body: map[string]any{
				"model":  "foo",
				"prompt": "Tell me a joke",
			},
			streaming: true,
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
												Key:      "X-Gateway-Model-Name",
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
												"model":  "foo",
												"prompt": "Tell me a joke",
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

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := &Server{streaming: test.streaming}
			bodyBytes, _ := json.Marshal(test.body)
			resp, err := server.HandleRequestBody(ctx, bodyBytes)
			if err != nil {
				if !test.wantErr {
					t.Fatalf("HandleRequestBody returned unexpected error: %v, want %v", err, test.wantErr)
				}
				return
			}

			if diff := cmp.Diff(test.want, resp, protocmp.Transform()); diff != "" {
				t.Errorf("HandleRequestBody returned unexpected response, diff(-want, +got): %v", diff)
			}
		})
	}

	wantMetrics := `
	# HELP bbr_model_not_in_body_total [ALPHA] Count of times the model was not present in the request body.
	# TYPE bbr_model_not_in_body_total counter
	bbr_model_not_in_body_total{} 1
	# HELP bbr_model_not_parsed_total [ALPHA] Count of times the model was in the request body but we could not parse it.
	# TYPE bbr_model_not_parsed_total counter
	bbr_model_not_parsed_total{} 1
	# HELP bbr_success_total [ALPHA] Count of successes pulling model name from body and injecting it in the request headers.
	# TYPE bbr_success_total counter
	bbr_success_total{} 1
	`

	if err := metricsutils.GatherAndCompare(crmetrics.Registry, strings.NewReader(wantMetrics), "inference_objective_request_total"); err != nil {
		t.Error(err)
	}
}

func mapToBytes(t *testing.T, m map[string]any) []byte {
	// Convert map to JSON byte array
	bytes, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("Marshal(): %v", err)
	}
	return bytes
}
