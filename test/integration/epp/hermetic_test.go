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

// Package epp contains integration tests for the ext proc while faking the backend pods.
package epp

import (
	"os"
	"strconv"
	"strings"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/testing/protocmp"
	"google.golang.org/protobuf/types/known/structpb"
	"k8s.io/component-base/metrics/legacyregistry"
	metricsutils "k8s.io/component-base/metrics/testutil"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/server"
	utiltesting "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

var models = []*v1alpha2.InferenceModel{
	utiltesting.MakeInferenceModel("sample").
		Namespace(pool.Namespace).
		ModelName("sql-lora").
		Criticality(v1alpha2.Critical).
		PoolName(pool.Name).
		TargetModel("sql-lora-1fdg2").
		ObjRef(),
	utiltesting.MakeInferenceModel("sheddable").
		Namespace(pool.Namespace).
		ModelName("sql-lora-sheddable").
		Criticality(v1alpha2.Sheddable).
		PoolName(pool.Name).
		TargetModel("sql-lora-1fdg3").
		ObjRef(),
	utiltesting.MakeInferenceModel("generic").
		Namespace(pool.Namespace).
		ModelName("my-model").
		Criticality(v1alpha2.Critical).
		PoolName(pool.Name).
		TargetModel("my-model-12345").
		ObjRef(),
	utiltesting.MakeInferenceModel("direct-model").
		Namespace(pool.Namespace).
		ModelName("direct-model").
		Criticality(v1alpha2.Critical).
		PoolName(pool.Name).
		ObjRef(),
}

func TestMain(m *testing.M) {
	cleanup := BeforeSuite()
	code := m.Run()
	cleanup()
	os.Exit(code)
}

func TestKubeInferenceModelRequest(t *testing.T) {
	tests := []struct {
		name              string
		req               *extProcPb.ProcessingRequest
		pods              map[backendmetrics.Pod]*backendmetrics.Metrics
		wantHeaders       []*configPb.HeaderValueOption
		wantMetadata      *structpb.Struct
		wantBody          []byte
		wantMetrics       string
		wantErr           bool
		immediateResponse *extProcPb.ImmediateResponse
	}{
		{
			name: "select lower queue and kv cache, no active lora",
			req:  utiltesting.GenerateRequest(logger, "test1", "my-model"),
			// pod-1 will be picked because it has relatively low queue size and low KV cache.
			pods: map[backendmetrics.Pod]*backendmetrics.Metrics{
				fakePod(0): {
					WaitingQueueSize:    3,
					KVCacheUsagePercent: 0.2,
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.1,
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.2,
				},
			},
			wantHeaders: []*configPb.HeaderValueOption{
				{
					Header: &configPb.HeaderValue{
						Key:      runserver.DefaultDestinationEndpointHintKey,
						RawValue: []byte("192.168.1.2:8000"),
					},
				},
				{
					Header: &configPb.HeaderValue{
						Key:      "Content-Length",
						RawValue: []byte("76"),
					},
				},
			},
			wantMetadata: makeMetadata("192.168.1.2:8000"),
			wantBody:     []byte("{\"max_tokens\":100,\"model\":\"my-model-12345\",\"prompt\":\"test1\",\"temperature\":0}"),
			wantMetrics: `
			# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
			# TYPE inference_model_request_total counter
			inference_model_request_total{model_name="my-model",target_model_name="my-model-12345"} 1
			`,
			wantErr: false,
		},
		{
			name: "select active lora, low queue",
			req:  utiltesting.GenerateRequest(logger, "test2", "sql-lora"),
			// pod-1 will be picked because it has relatively low queue size, with the requested
			// model being active, and has low KV cache.
			pods: map[backendmetrics.Pod]*backendmetrics.Metrics{
				fakePod(0): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo": 1,
						"bar": 1,
					},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.1,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg2": 1,
					},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo": 1,
						"bar": 1,
					},
				},
			},
			wantHeaders: []*configPb.HeaderValueOption{
				{
					Header: &configPb.HeaderValue{
						Key:      runserver.DefaultDestinationEndpointHintKey,
						RawValue: []byte("192.168.1.2:8000"),
					},
				},
				{
					Header: &configPb.HeaderValue{
						Key:      "Content-Length",
						RawValue: []byte("76"),
					},
				},
			},
			wantMetadata: makeMetadata("192.168.1.2:8000"),
			wantBody:     []byte("{\"max_tokens\":100,\"model\":\"sql-lora-1fdg2\",\"prompt\":\"test2\",\"temperature\":0}"),
			wantMetrics: `
			# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
			# TYPE inference_model_request_total counter
			inference_model_request_total{model_name="sql-lora",target_model_name="sql-lora-1fdg2"} 1
			`,
			wantErr: false,
		},
		{
			name: "select no lora despite active model, avoid excessive queue size",
			req:  utiltesting.GenerateRequest(logger, "test3", "sql-lora"),
			// pod-2 will be picked despite it NOT having the requested model being active
			// as it's above the affinity for queue size. Also is critical, so we should
			// still honor request despite all queues > 5
			pods: map[backendmetrics.Pod]*backendmetrics.Metrics{
				fakePod(0): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo": 1,
						"bar": 1,
					},
				},
				fakePod(1): {
					WaitingQueueSize:    200,
					KVCacheUsagePercent: 0.1,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg2": 1,
					},
				},
				fakePod(2): {
					WaitingQueueSize:    6,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo": 1,
					},
				},
			},
			wantHeaders: []*configPb.HeaderValueOption{
				{
					Header: &configPb.HeaderValue{
						Key:      runserver.DefaultDestinationEndpointHintKey,
						RawValue: []byte("192.168.1.3:8000"),
					},
				},
				{
					Header: &configPb.HeaderValue{
						Key:      "Content-Length",
						RawValue: []byte("76"),
					},
				},
			},
			wantMetadata: makeMetadata("192.168.1.3:8000"),
			wantBody:     []byte("{\"max_tokens\":100,\"model\":\"sql-lora-1fdg2\",\"prompt\":\"test3\",\"temperature\":0}"),
			wantMetrics: `
			# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
			# TYPE inference_model_request_total counter
			inference_model_request_total{model_name="sql-lora",target_model_name="sql-lora-1fdg2"} 1
			`,
			wantErr: false,
		},
		{
			name: "noncritical and all models past threshold, shed request",
			req:  utiltesting.GenerateRequest(logger, "test4", "sql-lora-sheddable"),
			// no pods will be picked as all models are either above kv threshold,
			// queue threshold, or both.
			pods: map[backendmetrics.Pod]*backendmetrics.Metrics{
				fakePod(0): {
					WaitingQueueSize:    6,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo":            1,
						"bar":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.85,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.9,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
			},
			wantHeaders:  []*configPb.HeaderValueOption{},
			wantMetadata: &structpb.Struct{},
			wantBody:     []byte(""),
			wantErr:      false,
			immediateResponse: &extProcPb.ImmediateResponse{
				Status: &envoyTypePb.HttpStatus{
					Code: envoyTypePb.StatusCode_TooManyRequests,
				},
			},
			wantMetrics: "",
		},
		{
			name: "noncritical, but one server has capacity, do not shed",
			req:  utiltesting.GenerateRequest(logger, "test5", "sql-lora-sheddable"),
			// pod 0 will be picked as all other models are above threshold
			pods: map[backendmetrics.Pod]*backendmetrics.Metrics{
				fakePod(0): {
					WaitingQueueSize:    4,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo":            1,
						"bar":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.85,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.9,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
			},
			wantHeaders: []*configPb.HeaderValueOption{
				{
					Header: &configPb.HeaderValue{
						Key:      runserver.DefaultDestinationEndpointHintKey,
						RawValue: []byte("192.168.1.1:8000"),
					},
				},
				{
					Header: &configPb.HeaderValue{
						Key:      "Content-Length",
						RawValue: []byte("76"),
					},
				},
			},
			wantMetadata: makeMetadata("192.168.1.1:8000"),
			wantBody:     []byte("{\"max_tokens\":100,\"model\":\"sql-lora-1fdg3\",\"prompt\":\"test5\",\"temperature\":0}"),
			wantMetrics: `
			# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
			# TYPE inference_model_request_total counter
			inference_model_request_total{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3"} 1
			`,
			wantErr: false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, cleanup := startEPPServer(t, &eppOptions{podMetrics: test.pods, models: models})
			t.Cleanup(cleanup)
			want := &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_RequestBody{
					RequestBody: &extProcPb.BodyResponse{
						Response: &extProcPb.CommonResponse{
							HeaderMutation: &extProcPb.HeaderMutation{
								SetHeaders: test.wantHeaders,
							},
							BodyMutation: &extProcPb.BodyMutation{
								Mutation: &extProcPb.BodyMutation_Body{
									Body: test.wantBody,
								},
							},
						},
					},
				},
				DynamicMetadata: test.wantMetadata,
			}
			res, err := sendRequest(t, client, test.req)

			if err != nil && !test.wantErr {
				t.Errorf("Unexpected error, got: %v, want error: %v", err, test.wantErr)
			}
			if test.immediateResponse != nil {
				want = &extProcPb.ProcessingResponse{
					Response: &extProcPb.ProcessingResponse_ImmediateResponse{
						ImmediateResponse: test.immediateResponse,
					},
				}
			}
			if diff := cmp.Diff(want, res, protocmp.Transform()); diff != "" {
				t.Errorf("Unexpected response, (-want +got): %v", diff)
			}

			if test.wantMetrics != "" {
				if err := metricsutils.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(test.wantMetrics), "inference_model_request_total"); err != nil {
					t.Error(err)
				}
			}

			legacyregistry.Reset()
		})
	}
}

func TestFullDuplexStreamed_KubeInferenceModelRequest(t *testing.T) {
	tests := []struct {
		name              string
		requests          []*extProcPb.ProcessingRequest
		pods              map[backendmetrics.Pod]*backendmetrics.Metrics
		wantResponses     []*extProcPb.ProcessingResponse
		wantMetrics       string
		wantErr           bool
		immediateResponse *extProcPb.ImmediateResponse
	}{
		// Request flow tests
		{
			name:     "select lower queue and kv cache, no active lora",
			requests: utiltesting.GenerateStreamedRequestSet(logger, "test1", "my-model"),
			// pod-1 will be picked because it has relatively low queue size and low KV cache.
			pods: map[backendmetrics.Pod]*backendmetrics.Metrics{
				fakePod(0): {
					WaitingQueueSize:    3,
					KVCacheUsagePercent: 0.2,
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.1,
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.2,
				},
			},
			wantMetrics: `
			# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
			# TYPE inference_model_request_total counter
			inference_model_request_total{model_name="my-model",target_model_name="my-model-12345"} 1
			`,
			wantErr: false,
			wantResponses: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestHeaders{
						RequestHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*configPb.HeaderValueOption{
										{
											Header: &configPb.HeaderValue{
												Key:      "x-gateway-destination-endpoint",
												RawValue: []byte("192.168.1.2:8000"),
											},
										},
										{
											Header: &configPb.HeaderValue{
												Key:      "Content-Length",
												RawValue: []byte(strconv.Itoa(76)),
											},
										},
									}},
							},
						},
					},
					DynamicMetadata: makeMetadata("192.168.1.2:8000"),
				},
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte("{\"max_tokens\":100,\"model\":\"my-model-12345\",\"prompt\":\"test1\",\"temperature\":0}"),
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
			name:     "select active lora, low queue",
			requests: utiltesting.GenerateStreamedRequestSet(logger, "test2", "sql-lora"),
			// pod-1 will be picked because it has relatively low queue size, with the requested
			// model being active, and has low KV cache.
			pods: map[backendmetrics.Pod]*backendmetrics.Metrics{
				fakePod(0): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo": 1,
						"bar": 1,
					},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.1,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg2": 1,
					},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo": 1,
						"bar": 1,
					},
				},
			},
			wantMetrics: `
			# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
			# TYPE inference_model_request_total counter
			inference_model_request_total{model_name="sql-lora",target_model_name="sql-lora-1fdg2"} 1
			`,
			wantErr: false,
			wantResponses: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestHeaders{
						RequestHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*configPb.HeaderValueOption{
										{
											Header: &configPb.HeaderValue{
												Key:      "x-gateway-destination-endpoint",
												RawValue: []byte("192.168.1.2:8000"),
											},
										},
										{
											Header: &configPb.HeaderValue{
												Key:      "Content-Length",
												RawValue: []byte(strconv.Itoa(76)),
											},
										},
									}},
							},
						},
					},
					DynamicMetadata: makeMetadata("192.168.1.2:8000"),
				},
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte("{\"max_tokens\":100,\"model\":\"sql-lora-1fdg2\",\"prompt\":\"test2\",\"temperature\":0}"),
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
			name:     "select no lora despite active model, avoid excessive queue size",
			requests: utiltesting.GenerateStreamedRequestSet(logger, "test3", "sql-lora"),
			// pod-2 will be picked despite it NOT having the requested model being active
			// as it's above the affinity for queue size. Also is critical, so we should
			// still honor request despite all queues > 5
			pods: map[backendmetrics.Pod]*backendmetrics.Metrics{
				fakePod(0): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo": 1,
						"bar": 1,
					},
				},
				fakePod(1): {
					WaitingQueueSize:    200,
					KVCacheUsagePercent: 0.1,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg2": 1,
					},
				},
				fakePod(2): {
					WaitingQueueSize:    6,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo": 1,
					},
				},
			},
			wantMetrics: `
			# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
			# TYPE inference_model_request_total counter
			inference_model_request_total{model_name="sql-lora",target_model_name="sql-lora-1fdg2"} 1
			`,
			wantErr: false,
			wantResponses: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestHeaders{
						RequestHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*configPb.HeaderValueOption{
										{
											Header: &configPb.HeaderValue{
												Key:      "x-gateway-destination-endpoint",
												RawValue: []byte("192.168.1.3:8000"),
											},
										},
										{
											Header: &configPb.HeaderValue{
												Key:      "Content-Length",
												RawValue: []byte(strconv.Itoa(76)),
											},
										},
									}},
							},
						},
					},
					DynamicMetadata: makeMetadata("192.168.1.3:8000"),
				},
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte("{\"max_tokens\":100,\"model\":\"sql-lora-1fdg2\",\"prompt\":\"test3\",\"temperature\":0}"),
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
			name:     "noncritical and all models past threshold, shed request",
			requests: utiltesting.GenerateStreamedRequestSet(logger, "test4", "sql-lora-sheddable"),
			// no pods will be picked as all models are either above kv threshold,
			// queue threshold, or both.
			pods: map[backendmetrics.Pod]*backendmetrics.Metrics{
				fakePod(0): {
					WaitingQueueSize:    6,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo":            1,
						"bar":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.85,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.9,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
			},
			wantErr:     false,
			wantMetrics: "",
			wantResponses: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_ImmediateResponse{
						ImmediateResponse: &extProcPb.ImmediateResponse{
							Status: &envoyTypePb.HttpStatus{
								Code: envoyTypePb.StatusCode_TooManyRequests,
							},
						},
					},
				},
			},
		},
		{
			name:     "noncritical, but one server has capacity, do not shed",
			requests: utiltesting.GenerateStreamedRequestSet(logger, "test5", "sql-lora-sheddable"),
			// pod 0 will be picked as all other models are above threshold
			pods: map[backendmetrics.Pod]*backendmetrics.Metrics{
				fakePod(0): {
					WaitingQueueSize:    4,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo":            1,
						"bar":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.85,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.9,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
			},
			wantMetrics: `
			# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
			# TYPE inference_model_request_total counter
			inference_model_request_total{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3"} 1
			`,
			wantErr: false,
			wantResponses: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestHeaders{
						RequestHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*configPb.HeaderValueOption{
										{
											Header: &configPb.HeaderValue{
												Key:      "x-gateway-destination-endpoint",
												RawValue: []byte("192.168.1.1:8000"),
											},
										},
										{
											Header: &configPb.HeaderValue{
												Key:      "Content-Length",
												RawValue: []byte(strconv.Itoa(76)),
											},
										},
									}},
							},
						},
					},
					DynamicMetadata: makeMetadata("192.168.1.1:8000"),
				},
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte("{\"max_tokens\":100,\"model\":\"sql-lora-1fdg3\",\"prompt\":\"test5\",\"temperature\":0}"),
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
			name: "body sent over multiple requests, noncritical, but one server has capacity, do not shed",
			requests: []*extProcPb.ProcessingRequest{
				{
					Request: &extProcPb.ProcessingRequest_RequestHeaders{
						RequestHeaders: &extProcPb.HttpHeaders{
							Headers: &configPb.HeaderMap{
								Headers: []*configPb.HeaderValue{
									{
										Key:   "hi",
										Value: "mom",
									},
								},
							},
						},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_RequestBody{
						RequestBody: &extProcPb.HttpBody{Body: []byte("{\"max_tokens\":100,\"model\":\"sql-lo"), EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_RequestBody{
						RequestBody: &extProcPb.HttpBody{Body: []byte("ra-sheddable\",\"prompt\":\"test6\",\"temperature\":0}"), EndOfStream: true},
					},
				},
			},

			//
			// pod 0 will be picked as all other models are above threshold
			pods: map[backendmetrics.Pod]*backendmetrics.Metrics{
				fakePod(0): {
					WaitingQueueSize:    4,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo":            1,
						"bar":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.85,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.9,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
			},
			wantMetrics: `
			# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
			# TYPE inference_model_request_total counter
			inference_model_request_total{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3"} 1
			`,
			wantErr: false,
			wantResponses: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestHeaders{
						RequestHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*configPb.HeaderValueOption{
										{
											Header: &configPb.HeaderValue{
												Key:      "x-gateway-destination-endpoint",
												RawValue: []byte("192.168.1.1:8000"),
											},
										},
										{
											Header: &configPb.HeaderValue{
												Key:      "Content-Length",
												RawValue: []byte(strconv.Itoa(76)),
											},
										},
									}},
							},
						},
					},
					DynamicMetadata: makeMetadata("192.168.1.1:8000"),
				},
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte("{\"max_tokens\":100,\"model\":\"sql-lora-1fdg3\",\"prompt\":\"test6\",\"temperature\":0}"),
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
			name: "inferencemodel's modelName is not translated, passthrough",
			requests: []*extProcPb.ProcessingRequest{
				{
					Request: &extProcPb.ProcessingRequest_RequestHeaders{
						RequestHeaders: &extProcPb.HttpHeaders{
							Headers: &configPb.HeaderMap{
								Headers: []*configPb.HeaderValue{
									{
										Key:   "hi",
										Value: "mom",
									},
								},
							},
						},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_RequestBody{
						RequestBody: &extProcPb.HttpBody{Body: []byte("{\"max_tokens\":100,\"model\":\"direct-"), EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_RequestBody{
						RequestBody: &extProcPb.HttpBody{Body: []byte("model\",\"prompt\":\"test6\",\"temperature\":0}"), EndOfStream: true},
					},
				},
			},

			//
			// pod 0 will be picked as all other models are above threshold
			pods: map[backendmetrics.Pod]*backendmetrics.Metrics{
				fakePod(0): {
					WaitingQueueSize:    4,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo":            1,
						"bar":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.85,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.9,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
			},
			wantMetrics: `
			# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
			# TYPE inference_model_request_total counter
			inference_model_request_total{model_name="direct-model",target_model_name="direct-model"} 1
			`,
			wantErr: false,
			wantResponses: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestHeaders{
						RequestHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*configPb.HeaderValueOption{
										{
											Header: &configPb.HeaderValue{
												Key:      "x-gateway-destination-endpoint",
												RawValue: []byte("192.168.1.2:8000"),
											},
										},
										{
											Header: &configPb.HeaderValue{
												Key:      "Content-Length",
												RawValue: []byte(strconv.Itoa(74)),
											},
										},
									}},
							},
						},
					},
					DynamicMetadata: makeMetadata("192.168.1.2:8000"),
				},
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte("{\"max_tokens\":100,\"model\":\"direct-model\",\"prompt\":\"test6\",\"temperature\":0}"),
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
		// Response flow tests
		{
			name: "responsebody sent over multiple requests, content-type is json, buffer",
			requests: []*extProcPb.ProcessingRequest{
				{
					Request: &extProcPb.ProcessingRequest_ResponseHeaders{
						ResponseHeaders: &extProcPb.HttpHeaders{
							Headers: &configPb.HeaderMap{
								Headers: []*configPb.HeaderValue{
									{
										Key:   "content-type",
										Value: "application/json",
									},
								},
							},
						},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{Body: []byte("{\"max_tokens\":100,\"model\":\"sql-lo"), EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{Body: []byte("ra-sheddable\",\"prompt\":\"test6\",\"temperature\":0}"), EndOfStream: true},
					},
				},
			},

			//
			// pod 0 will be picked as all other models are above threshold
			pods: map[backendmetrics.Pod]*backendmetrics.Metrics{
				fakePod(0): {
					WaitingQueueSize:    4,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo":            1,
						"bar":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.85,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.9,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
			},
			wantErr: false,
			wantResponses: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_ResponseHeaders{
						ResponseHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*configPb.HeaderValueOption{
										{
											Header: &configPb.HeaderValue{
												Key:      "x-went-into-resp-headers",
												RawValue: []byte("true"),
											},
										},
									},
								},
							},
						},
					},
				},
				{
					Response: &extProcPb.ProcessingResponse_ResponseBody{
						ResponseBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte("{\"max_tokens\":100,\"model\":\"sql-lora-sheddable\",\"prompt\":\"test6\",\"temperature\":0}"),
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
			name: "responsebody sent over a single request, but empty body with EndOfStream in the second request(this is how envoy operates); content-type is json, buffer",
			requests: []*extProcPb.ProcessingRequest{
				{
					Request: &extProcPb.ProcessingRequest_ResponseHeaders{
						ResponseHeaders: &extProcPb.HttpHeaders{
							Headers: &configPb.HeaderMap{
								Headers: []*configPb.HeaderValue{
									{
										Key:   "content-type",
										Value: "application/json",
									},
								},
							},
						},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{Body: []byte("{\"max_tokens\":100,\"model\":\"sql-lora-sheddable\",\"prompt\":\"test6\",\"temperature\":0}"), EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{Body: []byte(""), EndOfStream: true},
					},
				},
			},

			//
			// pod 0 will be picked as all other models are above threshold
			pods: map[backendmetrics.Pod]*backendmetrics.Metrics{
				fakePod(0): {
					WaitingQueueSize:    4,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo":            1,
						"bar":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.85,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.9,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
			},
			wantErr: false,
			wantResponses: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_ResponseHeaders{
						ResponseHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*configPb.HeaderValueOption{
										{
											Header: &configPb.HeaderValue{
												Key:      "x-went-into-resp-headers",
												RawValue: []byte("true"),
											},
										},
									},
								},
							},
						},
					},
				},
				{
					Response: &extProcPb.ProcessingResponse_ResponseBody{
						ResponseBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte("{\"max_tokens\":100,\"model\":\"sql-lora-sheddable\",\"prompt\":\"test6\",\"temperature\":0}"),
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
			name: "responsebody sent over a single request, but empty body with EndOfStream in the second request(this is how envoy operates); content-type is json, buffer",
			requests: []*extProcPb.ProcessingRequest{
				{
					Request: &extProcPb.ProcessingRequest_ResponseHeaders{
						ResponseHeaders: &extProcPb.HttpHeaders{
							Headers: &configPb.HeaderMap{
								Headers: []*configPb.HeaderValue{
									{
										Key:      "content-type",
										RawValue: []byte("text/event-stream"),
									},
									{
										Key:      "status",
										RawValue: []byte("200"),
									},
								},
							},
						},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{
							Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"tweet-summary-1","choices":[{"index":0,"text":"NEVER","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
							EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{
							Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"tweet-summary-1","choices":[{"index":0,"text":"GONNA","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
							EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{
							Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"tweet-summary-1","choices":[{"index":0,"text":"GIVE","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
							EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{
							Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"tweet-summary-1","choices":[{"index":0,"text":"YOU","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
							EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{
							Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"tweet-summary-1","choices":[{"index":0,"text":"UP","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
							EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{
							Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"tweet-summary-1","choices":[],"usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}`),
							EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{
							Body:        []byte("data: [DONE]"),
							EndOfStream: true},
					},
				},
			},
			wantErr: false,
			wantResponses: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_ResponseHeaders{
						ResponseHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*configPb.HeaderValueOption{
										{
											Header: &configPb.HeaderValue{
												Key:      "x-went-into-resp-headers",
												RawValue: []byte("true"),
											},
										},
									},
								},
							},
						},
					},
				},
				{
					Response: &extProcPb.ProcessingResponse_ResponseBody{
						ResponseBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"tweet-summary-1","choices":[{"index":0,"text":"NEVER","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
											EndOfStream: false,
										},
									},
								},
							},
						},
					},
				},
				{
					Response: &extProcPb.ProcessingResponse_ResponseBody{
						ResponseBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"tweet-summary-1","choices":[{"index":0,"text":"GONNA","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
											EndOfStream: false,
										},
									},
								},
							},
						},
					},
				},
				{
					Response: &extProcPb.ProcessingResponse_ResponseBody{
						ResponseBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"tweet-summary-1","choices":[{"index":0,"text":"GIVE","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
											EndOfStream: false,
										},
									},
								},
							},
						},
					},
				},
				{
					Response: &extProcPb.ProcessingResponse_ResponseBody{
						ResponseBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"tweet-summary-1","choices":[{"index":0,"text":"YOU","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
											EndOfStream: false,
										},
									},
								},
							},
						},
					},
				},
				{
					Response: &extProcPb.ProcessingResponse_ResponseBody{
						ResponseBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"tweet-summary-1","choices":[{"index":0,"text":"UP","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
											EndOfStream: false,
										},
									},
								},
							},
						},
					},
				},
				{
					Response: &extProcPb.ProcessingResponse_ResponseBody{
						ResponseBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"tweet-summary-1","choices":[],"usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}`),
											EndOfStream: false,
										},
									},
								},
							},
						},
					},
				},
				{
					Response: &extProcPb.ProcessingResponse_ResponseBody{
						ResponseBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte("data: [DONE]"),
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
			client, cleanup := startEPPServer(t, &eppOptions{podMetrics: test.pods, models: models, streamed: true})
			t.Cleanup(cleanup)
			responses, err := streamedRequest(t, client, test.requests, len(test.wantResponses))

			if err != nil && !test.wantErr {
				t.Errorf("Unexpected error, got: %v, want error: %v", err, test.wantErr)
			}
			if diff := cmp.Diff(test.wantResponses, responses, protocmp.Transform()); diff != "" {
				t.Errorf("Unexpected response, (-want +got): %v", diff)
			}

			if test.wantMetrics != "" {
				if err := metricsutils.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(test.wantMetrics), "inference_model_request_total"); err != nil {
					t.Error(err)
				}
			}

			legacyregistry.Reset()
		})
	}
}
