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

// Package epp contains integration tests for the Endpoint Picker extension.
package epp

import (
	"context"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/testing/protocmp"

	reqcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/request"
	pb "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/gen"
	"sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

const (
	testConfigWithVllmGRPCParser = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: queue-scorer
  - type: kv-cache-utilization-scorer
  - type: prefix-cache-scorer
  - type: lora-affinity-scorer
  - type: vllmgrpc-parser
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: queue-scorer
      - pluginRef: kv-cache-utilization-scorer
      - pluginRef: prefix-cache-scorer
      - pluginRef: lora-affinity-scorer
parser:
  pluginRef: vllmgrpc-parser
featureGates:
  - disableDataLayer
`
)

func TestFullDuplexStreamed_GRPC_KubeInferenceObjectiveRequest(t *testing.T) {
	tests := []struct {
		name          string
		requests      []*extProcPb.ProcessingRequest
		pods          []podState
		wantResponses []*extProcPb.ProcessingResponse
		wantMetrics   map[string]string
		// requiresCRDs indicates that this test case relies on specific Gateway API CRD features (like
		// InferenceModelRewrite) which are not available in Standalone runMode without CRD.
		requiresCRDs bool
	}{
		// --- Standard Routing Logic ---
		{
			name:     "select lower queue and kv cache",
			requests: integration.ReqGRPCLLM(logger, "test1", inferenceObjectiveWithPriority4),
			pods: []podState{
				P(0, 3, 0.2),
				P(1, 0, 0.1), // Winner (Low Queue, Low KV)
				P(2, 10, 0.2),
			},
			wantResponses: ExpectGRPCRouteTo("192.168.1.2:8000", "test1"),
			wantMetrics: map[string]string{
				"inference_objective_request_total": cleanMetric(metricReqTotal("", "", 4)),
				"inference_pool_ready_pods":         cleanMetric(metricReadyPods(3)),
			},
		},
		{
			name:     "select lower queue with streaming request",
			requests: integration.ReqGRPCLLMWithStream(logger, "test-stream", inferenceObjectiveWithPriority4),
			pods: []podState{
				P(0, 3, 0.2),
				P(1, 0, 0.1), // Winner
				P(2, 10, 0.2),
			},
			wantResponses: ExpectGRPCRouteToWithStream("192.168.1.2:8000", "test-stream"),
			wantMetrics: map[string]string{
				"inference_objective_request_total": cleanMetric(metricReqTotal("", "", 4)),
				"inference_pool_ready_pods":         cleanMetric(metricReadyPods(3)),
			},
		},
		{
			name:     "do not shed requests by default",
			requests: integration.ReqGRPCLLM(logger, "test2", ""),
			pods: []podState{
				P(0, 6, 0.2, "foo", "bar"), // Winner (Lowest saturated)
				P(1, 0, 0.85, "foo"),
				P(2, 10, 0.9, "foo"),
			},
			wantResponses: ExpectGRPCRouteTo("192.168.1.1:8000", "test2"),
			wantMetrics: map[string]string{
				"inference_objective_request_total": cleanMetric(metricReqTotal("", "", 0)),
			},
		},

		// --- Error Handling & Edge Cases ----
		{
			name: "invalid gRPC payload",
			requests: integration.ReqRaw(
				map[string]string{"hi": "mom"},
				"plain text not grpc payload style",
			),
			pods: []podState{
				P(0, 0, 0.2, "foo", "bar"),
			},
			wantResponses: ExpectReject(
				envoyTypePb.StatusCode_BadRequest,
				"inference error: BadRequest - parsing gRPC payload: not able to parse payload",
			),
		},
		{
			name: "split body across chunks",
			requests: func() []*extProcPb.ProcessingRequest {
				req := &pb.GenerateRequest{
					Input: &pb.GenerateRequest_Text{
						Text: "test3",
					},
				}
				gRPCPayload, _ := integration.CreateGrpcPayload(req)
				return integration.ReqRaw(
					map[string]string{
						"hi":                         "mom",
						reqcommon.RequestIdHeaderKey: "test-request-id",
					},
					string(gRPCPayload[0:len(gRPCPayload)/2]),
					string(gRPCPayload[len(gRPCPayload)/2:]),
				)
			}(),
			pods: []podState{
				P(0, 4, 0.2, "foo", "bar"),
				P(1, 4, 0.85, "foo"),
			},
			wantResponses: ExpectGRPCRouteTo("192.168.1.1:8000", "test3"),
			wantMetrics: map[string]string{
				"inference_objective_request_total": cleanMetric(metricReqTotal("", "", 0)),
			},
		},
		{
			name:     "no backend pods available",
			requests: integration.ReqHeaderOnly(map[string]string{"content-type": "application/json"}),
			pods:     nil,
			wantResponses: ExpectReject(envoyTypePb.StatusCode_InternalServerError,
				"inference error: Internal - no pods available in datastore"),
		},

		// // --- Subsetting & Metadata ---
		{
			name: "subsetting: select best from subset",
			// Only pods in the subset list are eligible.
			requests: integration.GenerateStreamedGRPCRequestSet(logger, "test2", "",
				[]string{"192.168.1.1:8000", "192.168.1.2:8000", "192.168.1.3:8000"}),
			pods: []podState{
				P(0, 0, 0.2, "foo"),
				P(1, 0, 0.1, "foo", modelSQLLoraTarget), // Winner (Low Queue + Matches Subset)
				P(2, 10, 0.2, "foo"),
			},
			wantResponses: ExpectGRPCRouteTo("192.168.1.2:8000", "test2"),
		},
		{
			name:     "subsetting: partial match",
			requests: integration.GenerateStreamedGRPCRequestSet(logger, "test2", "", []string{"192.168.1.3:8000"}),
			pods: []podState{
				P(0, 0, 0.2, "foo"),
				P(1, 0, 0.1, "foo", modelSQLLoraTarget),
				P(2, 10, 0.2, "foo"), // Winner (Matches Subset, despite load)
			},
			wantResponses: ExpectGRPCRouteTo("192.168.1.3:8000", "test2"),
		},
		{
			name:     "subsetting: no pods match",
			requests: integration.GenerateStreamedGRPCRequestSet(logger, "test2", "", []string{"192.168.1.99:8000"}),
			pods: []podState{
				P(0, 0, 0.2, "foo"),
				P(1, 0, 0.1, "foo", modelSQLLoraTarget),
			},
			wantResponses: ExpectReject(envoyTypePb.StatusCode_ServiceUnavailable,
				"inference error: ServiceUnavailable - failed to find candidate endpoints for serving the request"),
		},

		// --- Response Processing (Non-streaming) ---
		{
			name: "response buffering: multi-chunk",
			requests: func() []*extProcPb.ProcessingRequest {
				resp := &pb.GenerateResponse{
					Response: &pb.GenerateResponse_Chunk{
						Chunk: &pb.GenerateStreamChunk{
							TokenIds: []uint32{1, 2, 3, 4, 5, 6, 7},
						},
					},
				}
				gRPCPayload, _ := integration.CreateGrpcPayload(resp)
				return ReqResponseGRPCWithTailer(
					map[string]string{"content-type": "application/grpc"},
					gRPCPayload[:len(gRPCPayload)/2],
					gRPCPayload[len(gRPCPayload)/2:],
				)
			}(),
			pods: []podState{P(0, 4, 0.2)},
			wantResponses: func() []*extProcPb.ProcessingResponse {
				resp := &pb.GenerateResponse{
					Response: &pb.GenerateResponse_Chunk{
						Chunk: &pb.GenerateStreamChunk{
							TokenIds: []uint32{1, 2, 3, 4, 5, 6, 7},
						},
					},
				}
				gRPCPayload, _ := integration.CreateGrpcPayload(resp)
				return ExpectBufferResp(string(gRPCPayload), "application/grpc")
			}(),
		},
		{
			name: "response buffering: invalid gRPC",
			requests: ReqResponseGRPCWithTailer(
				map[string]string{"content-type": "application/grpc"},
				[]byte("no healthy upstream"),
			),
			pods:          []podState{P(0, 4, 0.2)},
			wantResponses: ExpectBufferResp("no healthy upstream", "application/grpc"),
		},
		{
			name: "response with token usage",
			requests: func() []*extProcPb.ProcessingRequest {
				resp := &pb.GenerateResponse{
					Response: &pb.GenerateResponse_Complete{
						Complete: &pb.GenerateComplete{
							PromptTokens:     7,
							CompletionTokens: 10,
						},
					},
				}
				gRPCPayload, _ := integration.CreateGrpcPayload(resp)
				return ReqResponseGRPCWithTailer(
					map[string]string{"content-type": "application/grpc"},
					gRPCPayload[:len(gRPCPayload)/2],
					gRPCPayload[len(gRPCPayload)/2:],
				)
			}(),
			pods: []podState{P(0, 4, 0.2)},
			wantResponses: func() []*extProcPb.ProcessingResponse {
				resp := &pb.GenerateResponse{
					Response: &pb.GenerateResponse_Complete{
						Complete: &pb.GenerateComplete{
							PromptTokens:     7,
							CompletionTokens: 10,
						},
					},
				}
				gRPCPayload, _ := integration.CreateGrpcPayload(resp)
				return ExpectBufferResp(string(gRPCPayload), "application/grpc")
			}(),
			// Labels are empty because we skipped the Request phase.
			wantMetrics: map[string]string{
				"inference_objective_input_tokens": cleanMetric(`
					# HELP inference_objective_input_tokens [ALPHA] Inference objective input token count distribution for requests in each model.
					# TYPE inference_objective_input_tokens histogram
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="1"} 0
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="8"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="16"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="32"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="64"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="128"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="256"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="512"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="1024"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="2048"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="4096"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="8192"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="16384"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="32778"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="65536"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="131072"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="262144"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="524288"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="1.048576e+06"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="+Inf"} 1
					inference_objective_input_tokens_sum{model_name="",target_model_name=""} 7
					inference_objective_input_tokens_count{model_name="",target_model_name=""} 1
					`),
			},
		},
		{
			name: "response streaming with token usage",
			requests: func() []*extProcPb.ProcessingRequest {
				reqs := integration.ReqGRPCLLMWithStream(logger, "test-stream", inferenceObjectiveWithPriority4)

				resp1 := &pb.GenerateResponse{
					Response: &pb.GenerateResponse_Chunk{
						Chunk: &pb.GenerateStreamChunk{
							TokenIds: []uint32{1, 2, 3},
						},
					},
				}
				resp2 := &pb.GenerateResponse{
					Response: &pb.GenerateResponse_Complete{
						Complete: &pb.GenerateComplete{
							PromptTokens:     7,
							CompletionTokens: 10,
						},
					},
				}

				gRPCPayload1, _ := integration.CreateGrpcPayload(resp1)
				gRPCPayload2, _ := integration.CreateGrpcPayload(resp2)

				respHeaders := &extProcPb.ProcessingRequest{
					Request: &extProcPb.ProcessingRequest_ResponseHeaders{
						ResponseHeaders: &extProcPb.HttpHeaders{
							Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
								{Key: "content-type", Value: "application/grpc"},
							}},
						},
					},
				}
				respBody1 := &extProcPb.ProcessingRequest{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{
							Body: gRPCPayload1,
						},
					},
				}
				respBody2 := &extProcPb.ProcessingRequest{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{
							Body: gRPCPayload2,
						},
					},
				}
				respTrailers := &extProcPb.ProcessingRequest{
					Request: &extProcPb.ProcessingRequest_ResponseTrailers{
						ResponseTrailers: &extProcPb.HttpTrailers{},
					},
				}

				return append(reqs, respHeaders, respBody1, respBody2, respTrailers)
			}(),
			pods: []podState{P(0, 4, 0.2)},
			wantResponses: func() []*extProcPb.ProcessingResponse {
				reqs := ExpectGRPCRouteToWithStream("192.168.1.1:8000", "test-stream")

				resp1 := &pb.GenerateResponse{
					Response: &pb.GenerateResponse_Chunk{
						Chunk: &pb.GenerateStreamChunk{
							TokenIds: []uint32{1, 2, 3},
						},
					},
				}
				resp2 := &pb.GenerateResponse{
					Response: &pb.GenerateResponse_Complete{
						Complete: &pb.GenerateComplete{
							PromptTokens:     7,
							CompletionTokens: 10,
						},
					},
				}
				gRPCPayload1, _ := integration.CreateGrpcPayload(resp1)
				gRPCPayload2, _ := integration.CreateGrpcPayload(resp2)

				// Expect Headers response
				respRespHeaders := integration.NewResponseHeaders(
					&configPb.HeaderValueOption{Header: &configPb.HeaderValue{
						Key:      "x-went-into-resp-headers",
						RawValue: []byte("true"),
					}},
					&configPb.HeaderValueOption{Header: &configPb.HeaderValue{
						Key:      "content-type",
						RawValue: []byte("application/grpc"),
					}},
				)

				// Expect Streaming body frame responses
				respChunk1 := integration.NewResponseStreamChunk(string(gRPCPayload1), false)
				respChunk2 := integration.NewResponseStreamChunk(string(gRPCPayload2), false)

				return append(reqs, respRespHeaders, respChunk1, respChunk2)
			}(),
			wantMetrics: map[string]string{
				"inference_objective_input_tokens": cleanMetric(`
					# HELP inference_objective_input_tokens [ALPHA] Inference objective input token count distribution for requests in each model.
					# TYPE inference_objective_input_tokens histogram
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="1"} 0
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="8"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="16"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="32"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="64"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="128"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="256"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="512"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="1024"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="2048"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="4096"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="8192"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="16384"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="32778"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="65536"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="131072"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="262144"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="524288"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="1.048576e+06"} 1
					inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="+Inf"} 1
					inference_objective_input_tokens_sum{model_name="",target_model_name=""} 7
					inference_objective_input_tokens_count{model_name="",target_model_name=""} 1
					`),
				"inference_objective_output_tokens": cleanMetric(`
					# HELP inference_objective_output_tokens [ALPHA] Inference objective output token count distribution for requests in each model.
					# TYPE inference_objective_output_tokens histogram
					inference_objective_output_tokens_bucket{model_name="",target_model_name="",le="1"} 0
					inference_objective_output_tokens_bucket{model_name="",target_model_name="",le="8"} 0
					inference_objective_output_tokens_bucket{model_name="",target_model_name="",le="16"} 1
					inference_objective_output_tokens_bucket{model_name="",target_model_name="",le="32"} 1
					inference_objective_output_tokens_bucket{model_name="",target_model_name="",le="64"} 1
					inference_objective_output_tokens_bucket{model_name="",target_model_name="",le="128"} 1
					inference_objective_output_tokens_bucket{model_name="",target_model_name="",le="256"} 1
					inference_objective_output_tokens_bucket{model_name="",target_model_name="",le="512"} 1
					inference_objective_output_tokens_bucket{model_name="",target_model_name="",le="1024"} 1
					inference_objective_output_tokens_bucket{model_name="",target_model_name="",le="2048"} 1
					inference_objective_output_tokens_bucket{model_name="",target_model_name="",le="4096"} 1
					inference_objective_output_tokens_bucket{model_name="",target_model_name="",le="8192"} 1
					inference_objective_output_tokens_bucket{model_name="",target_model_name="",le="+Inf"} 1
					inference_objective_output_tokens_sum{model_name="",target_model_name=""} 10
					inference_objective_output_tokens_count{model_name="",target_model_name=""} 1
					`),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			h := NewTestHarness(t, ctx, WithStandardMode(), WithConfigText(testConfigWithVllmGRPCParser)).WithBaseResources()

			h.WithPods(tc.pods).WaitForSync(len(tc.pods), modelMyModel)
			if len(tc.pods) > 0 {
				h.WaitForReadyPodsMetric(len(tc.pods))
			}

			responses, err := integration.StreamedRequest(t, h.Client, tc.requests, len(tc.wantResponses))
			require.NoError(t, err)

			if diff := cmp.Diff(tc.wantResponses, responses,
				protocmp.Transform(),
				protocmp.SortRepeated(func(a, b *configPb.HeaderValueOption) bool {
					return a.GetHeader().GetKey() < b.GetHeader().GetKey()
				}),
			); diff != "" {
				t.Errorf("Response mismatch (-want +got): %v", diff)
			}

			if len(tc.wantMetrics) > 0 {
				h.ExpectMetrics(tc.wantMetrics)
			}
		})
	}
}
