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
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/google/go-cmp/cmp"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/testing/protocmp"
	"google.golang.org/protobuf/types/known/structpb"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	k8syaml "k8s.io/apimachinery/pkg/util/yaml"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/component-base/metrics/legacyregistry"
	metricsutils "k8s.io/component-base/metrics/testutil"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	k8sclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/config"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/server"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/server"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	epptestutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
	integrationutils "sigs.k8s.io/gateway-api-inference-extension/test/integration"
	"sigs.k8s.io/yaml"
)

const (
	port        = runserver.DefaultGrpcPort
	metricsPort = 8889
)

var (
	serverRunner *runserver.ExtProcServerRunner
	k8sClient    k8sclient.Client
	testEnv      *envtest.Environment
	scheme       = runtime.NewScheme()
	logger       = logutil.NewTestLogger().V(logutil.VERBOSE)
)

func TestMain(m *testing.M) {
	cleanup := BeforeSuite()
	code := m.Run()
	cleanup()
	os.Exit(code)
}

func TestFullDuplexStreamed_KubeInferenceModelRequest(t *testing.T) {
	tests := []struct {
		name              string
		requests          []*extProcPb.ProcessingRequest
		pods              map[backendmetrics.Pod]*backendmetrics.Metrics
		wantResponses     []*extProcPb.ProcessingResponse
		wantMetrics       map[string]string
		wantErr           bool
		immediateResponse *extProcPb.ImmediateResponse
	}{
		// Request flow tests
		{
			name:     "select lower queue and kv cache, no active lora",
			requests: integrationutils.GenerateStreamedRequestSet(logger, "test1", "my-model"),
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
			wantMetrics: map[string]string{`inference_model_request_total`: `
					# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
					# TYPE inference_model_request_total counter
					inference_model_request_total{model_name="my-model",target_model_name="my-model-12345"} 1
					`,
				`inference_pool_ready_pods`: `
					# HELP inference_pool_ready_pods [ALPHA] The number of ready pods in the inference server pool.
					# TYPE inference_pool_ready_pods gauge
					inference_pool_ready_pods{name="vllm-llama3-8b-instruct-pool"} 3
					`,
			},
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
			requests: integrationutils.GenerateStreamedRequestSet(logger, "test2", "sql-lora"),
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
					WaitingModels: map[string]int{},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.1,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg2": 1,
					},
					WaitingModels: map[string]int{},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo": 1,
						"bar": 1,
					},
					WaitingModels: map[string]int{},
				},
			},
			wantMetrics: map[string]string{`inference_model_request_total`: `
					# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
					# TYPE inference_model_request_total counter
					inference_model_request_total{model_name="sql-lora",target_model_name="sql-lora-1fdg2"} 1
					`},
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
			requests: integrationutils.GenerateStreamedRequestSet(logger, "test3", "sql-lora"),
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
					WaitingModels: map[string]int{},
				},
				fakePod(1): {
					WaitingQueueSize:    200,
					KVCacheUsagePercent: 0.1,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg2": 1,
					},
					WaitingModels: map[string]int{},
				},
				fakePod(2): {
					WaitingQueueSize:    6,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo": 1,
					},
					WaitingModels: map[string]int{},
				},
			},
			wantMetrics: map[string]string{`inference_model_request_total`: `
					# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
					# TYPE inference_model_request_total counter
					inference_model_request_total{model_name="sql-lora",target_model_name="sql-lora-1fdg2"} 1
					`},
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
			requests: integrationutils.GenerateStreamedRequestSet(logger, "test4", "sql-lora-sheddable"),
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
					WaitingModels: map[string]int{},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.85,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
					WaitingModels: map[string]int{},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.9,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
					WaitingModels: map[string]int{},
				},
			},
			wantErr:     false,
			wantMetrics: map[string]string{},
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
			requests: integrationutils.GenerateStreamedRequestSet(logger, "test5", "sql-lora-sheddable"),
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
					WaitingModels: map[string]int{},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.85,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
					WaitingModels: map[string]int{},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.9,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
					WaitingModels: map[string]int{},
				},
			},
			wantMetrics: map[string]string{`inference_model_request_total`: `
					# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
					# TYPE inference_model_request_total counter
					inference_model_request_total{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3"} 1
					`},
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
					WaitingModels: map[string]int{},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.85,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
					WaitingModels: map[string]int{},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.9,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
					WaitingModels: map[string]int{},
				},
			},
			wantMetrics: map[string]string{`inference_model_request_total`: `
					# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
					# TYPE inference_model_request_total counter
					inference_model_request_total{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3"} 1
					`},
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
					WaitingModels: map[string]int{},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.85,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
					WaitingModels: map[string]int{},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.9,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
					WaitingModels: map[string]int{},
				},
			},
			wantMetrics: map[string]string{`inference_model_request_total`: `
					# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
					# TYPE inference_model_request_total counter
					inference_model_request_total{model_name="direct-model",target_model_name="direct-model"} 1
					`},
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
					WaitingModels: map[string]int{},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.85,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
					WaitingModels: map[string]int{},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.9,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
					WaitingModels: map[string]int{},
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
					WaitingModels: map[string]int{},
				},
				fakePod(1): {
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.85,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
					WaitingModels: map[string]int{},
				},
				fakePod(2): {
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.9,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
					WaitingModels: map[string]int{},
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
							Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[{"index":0,"text":"NEVER","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
							EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{
							Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[{"index":0,"text":"GONNA","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
							EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{
							Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[{"index":0,"text":"GIVE","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
							EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{
							Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[{"index":0,"text":"YOU","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
							EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{
							Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[{"index":0,"text":"UP","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
							EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{
							Body: []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[],"usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}
		data: [DONE]`,
							),
							EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_ResponseBody{
						ResponseBody: &extProcPb.HttpBody{
							Body:        []byte(""),
							EndOfStream: true},
					},
				},
			},
			wantErr: false,
			wantMetrics: map[string]string{`inference_model_input_tokens`: `
					# HELP inference_model_input_tokens [ALPHA] Inference model input token count distribution for requests in each model.
					# TYPE inference_model_input_tokens histogram
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="1"} 0
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="8"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="16"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="32"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="64"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="128"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="256"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="512"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="1024"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="2048"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="4096"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="8192"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="16384"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="32778"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="65536"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="131072"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="262144"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="524288"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="1.048576e+06"} 1
		            inference_model_input_tokens_bucket{model_name="",target_model_name="",le="+Inf"} 1
		            inference_model_input_tokens_sum{model_name="",target_model_name=""} 7
		            inference_model_input_tokens_count{model_name="",target_model_name=""} 1
					`},
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
											Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[{"index":0,"text":"NEVER","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
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
											Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[{"index":0,"text":"GONNA","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
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
											Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[{"index":0,"text":"GIVE","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
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
											Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[{"index":0,"text":"YOU","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
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
											Body:        []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[{"index":0,"text":"UP","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`),
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
											Body: []byte(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[],"usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}
		data: [DONE]`,
											),
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
											Body:        []byte(""),
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
		// Bodyless Request test
		{
			name: "simple GET Request",
			requests: []*extProcPb.ProcessingRequest{
				{
					Request: &extProcPb.ProcessingRequest_RequestHeaders{
						RequestHeaders: &extProcPb.HttpHeaders{
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
							EndOfStream: true,
						},
					},
				},
			},
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
									}},
							},
						},
					},
					DynamicMetadata: makeMetadata("192.168.1.1:8000"),
				},
			},
			pods: map[backendmetrics.Pod]*backendmetrics.Metrics{
				fakePod(0): {
					WaitingQueueSize:    4,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo":            1,
						"bar":            1,
						"sql-lora-1fdg3": 1,
					},
					WaitingModels: map[string]int{},
				},
			},
			wantMetrics: map[string]string{`inference_pool_ready_pods`: `
				# HELP inference_pool_ready_pods [ALPHA] The number of ready pods in the inference server pool.
				# TYPE inference_pool_ready_pods gauge
				inference_pool_ready_pods{name="vllm-llama3-8b-instruct-pool"} 1
				`},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, cleanup := setUpHermeticServer(t, test.pods, true)
			t.Cleanup(cleanup)
			responses, err := integrationutils.StreamedRequest(t, client, test.requests, len(test.wantResponses))

			if err != nil && !test.wantErr {
				t.Errorf("Unexpected error, got: %v, want error: %v", err, test.wantErr)
			}
			if diff := cmp.Diff(test.wantResponses, responses, protocmp.Transform()); diff != "" {
				t.Errorf("Unexpected response, (-want +got): %v", diff)
			}

			if len(test.wantMetrics) != 0 {
				for metricName, value := range test.wantMetrics {
					if err := metricsutils.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(value), metricName); err != nil {
						t.Error(err)
					}
				}
			}

			legacyregistry.Reset()
		})
	}
}

func setUpHermeticServer(t *testing.T, podAndMetrics map[backendmetrics.Pod]*backendmetrics.Metrics, streamed bool) (client extProcPb.ExternalProcessor_ProcessClient, cleanup func()) {
	// Reconfigure the TestPodMetricsClient.
	res := map[types.NamespacedName]*backendmetrics.Metrics{}
	for pod, metrics := range podAndMetrics {
		res[pod.NamespacedName] = metrics
	}
	serverRunner.TestPodMetricsClient.SetRes(res)
	serverRunner.UseStreaming = streamed

	serverCtx, stopServer := context.WithCancel(context.Background())

	// TODO: this should be consistent with the inference pool
	podLabels := map[string]string{
		"app": "vllm-llama3-8b-instruct-pool",
	}

	for pod := range podAndMetrics {
		pod := epptestutil.MakePod(pod.NamespacedName.Name).
			Namespace(pod.NamespacedName.Namespace).
			ReadyCondition().
			Labels(podLabels).
			IP(pod.Address).
			Complete().
			ObjRef()

		copy := pod.DeepCopy()
		if err := k8sClient.Create(context.Background(), copy); err != nil {
			logutil.Fatal(logger, err, "Failed to create pod", "pod", pod)
		}

		// since no pod controllers deployed in fake environment, we manually update pod status
		copy.Status = pod.Status
		if err := k8sClient.Status().Update(context.Background(), copy); err != nil {
			logutil.Fatal(logger, err, "Failed to update pod status", "pod", pod)
		}
	}
	go func() {
		if err := serverRunner.AsRunnable(logger.WithName("ext-proc")).Start(serverCtx); err != nil {
			logutil.Fatal(logger, err, "Failed to start ext-proc server")
		}
	}()

	time.Sleep(serverRunner.RefreshPrometheusMetricsInterval) // wait for metrics to get available before running tests that rely on these metrics

	// check if all pods are synced to datastore
	assert.EventuallyWithT(t, func(t *assert.CollectT) {
		assert.Len(t, serverRunner.Datastore.PodGetAll(), len(podAndMetrics), "Datastore not synced")
	}, 10*time.Second, time.Second)

	address := fmt.Sprintf("localhost:%v", port)
	// Create a grpc connection
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		logutil.Fatal(logger, err, "Failed to connect", "address", address)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	client, err = extProcPb.NewExternalProcessorClient(conn).Process(ctx)
	if err != nil {
		logutil.Fatal(logger, err, "Failed to create client")
	}
	return client, func() {
		cancel()
		conn.Close()
		stopServer()

		// clear created pods
		for pod := range podAndMetrics {
			pod := epptestutil.MakePod(pod.NamespacedName.Name).
				Namespace(pod.NamespacedName.Namespace).Complete().ObjRef()

			if err := k8sClient.Delete(context.Background(), pod); err != nil {
				logutil.Fatal(logger, err, "Failed to delete pod", "pod", fakePod)
			}
		}
	}
}

func fakePod(index int) backendmetrics.Pod {
	return backendmetrics.Pod{
		NamespacedName: types.NamespacedName{Name: fmt.Sprintf("pod-%v", index), Namespace: "default"},
		Address:        fmt.Sprintf("192.168.1.%d", index+1),
	}
}

// Sets up a test environment and returns the runner struct
func BeforeSuite() func() {
	// Set up mock k8s API Client
	testEnv = &envtest.Environment{
		CRDDirectoryPaths:     []string{filepath.Join("..", "..", "..", "config", "crd", "bases")},
		ErrorIfCRDPathMissing: true,
	}
	cfg, err := testEnv.Start()
	if err != nil {
		logutil.Fatal(logger, err, "Failed to start test environment", "config", cfg)
	}

	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(v1alpha2.Install(scheme))

	k8sClient, err = k8sclient.New(cfg, k8sclient.Options{Scheme: scheme})
	if err != nil {
		logutil.Fatal(logger, err, "Failed to start k8s Client")
	} else if k8sClient == nil {
		logutil.Fatal(logger, nil, "No error, but returned kubernetes client is nil", "config", cfg)
	}

	// Init runtime.
	ctrl.SetLogger(logger)

	mgr, err := server.NewManagerWithOptions(cfg, managerTestOptions("default", "vllm-llama3-8b-instruct-pool"))
	if err != nil {
		logutil.Fatal(logger, err, "Failed to create controller manager")
	}

	if err := registerMetricsHandler(mgr, metricsPort); err != nil {
		logutil.Fatal(logger, err, "Failed to register metrics handler")
	}

	serverRunner = runserver.NewDefaultExtProcServerRunner()
	serverRunner.TestPodMetricsClient = &backendmetrics.FakePodMetricsClient{}
	pmf := backendmetrics.NewPodMetricsFactory(serverRunner.TestPodMetricsClient, 10*time.Millisecond)
	// Adjust from defaults
	serverRunner.PoolNamespacedName = types.NamespacedName{Name: "vllm-llama3-8b-instruct-pool", Namespace: "default"}
	serverRunner.Datastore = datastore.NewDatastore(context.Background(), pmf)
	serverRunner.SecureServing = false

	if err := serverRunner.SetupWithManager(context.Background(), mgr); err != nil {
		logutil.Fatal(logger, err, "Failed to setup server runner")
	}

	// Start the controller manager in a go routine, not blocking
	go func() {
		if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
			logutil.Fatal(logger, err, "Failed to start manager")
		}
	}()

	logger.Info("Setting up hermetic ExtProc server")

	// Unmarshal CRDs from file into structs
	manifestsPath := filepath.Join("..", "..", "testdata", "inferencepool-with-model-hermetic.yaml")
	docs, err := readDocuments(manifestsPath)
	if err != nil {
		logutil.Fatal(logger, err, "Can't read object manifests", "path", manifestsPath)
	}

	for _, doc := range docs {
		obj := &unstructured.Unstructured{}
		if err = yaml.Unmarshal(doc, obj); err != nil {
			logutil.Fatal(logger, err, "Can't unmarshal object", "document", doc)
		}
		logger.Info("Creating object", "kind", obj.GetKind(), "object", obj)
		if err := k8sClient.Create(context.Background(), obj); err != nil {
			logutil.Fatal(logger, err, "Unable to create object", "object", obj.GetName())
		}
	}

	assert.Eventually(nil, func() bool {
		modelExist := serverRunner.Datastore.ModelGet("my-model")
		synced := serverRunner.Datastore.PoolHasSynced() && modelExist != nil
		return synced
	}, 10*time.Second, 10*time.Millisecond)

	return func() {
		_ = testEnv.Stop()
		_ = k8sClient.DeleteAllOf(context.Background(), &v1alpha2.InferencePool{})
		_ = k8sClient.DeleteAllOf(context.Background(), &v1alpha2.InferenceModel{})
	}
}

// readDocuments reads documents from file.
func readDocuments(fp string) ([][]byte, error) {
	b, err := os.ReadFile(fp)
	if err != nil {
		return nil, err
	}

	docs := [][]byte{}
	reader := k8syaml.NewYAMLReader(bufio.NewReader(bytes.NewReader(b)))
	for {
		// Read document
		doc, err := reader.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}
		docs = append(docs, doc)
	}
	return docs, nil
}

func makeMetadata(endpoint string) *structpb.Struct {
	return &structpb.Struct{
		Fields: map[string]*structpb.Value{
			runserver.DefaultDestinationEndpointHintMetadataNamespace: {
				Kind: &structpb.Value_StructValue{
					StructValue: &structpb.Struct{
						Fields: map[string]*structpb.Value{
							runserver.DefaultDestinationEndpointHintKey: {
								Kind: &structpb.Value_StringValue{
									StringValue: endpoint,
								},
							},
						},
					},
				},
			},
		},
	}
}

// registerMetricsHandler is a simplified version of metrics endpoint handler
// without Authentication for integration tests.
func registerMetricsHandler(mgr manager.Manager, port int) error {
	metrics.Register()

	// Init HTTP server.
	h := promhttp.HandlerFor(
		legacyregistry.DefaultGatherer,
		promhttp.HandlerOpts{},
	)

	mux := http.NewServeMux()
	mux.Handle("/metrics", h)

	srv := &http.Server{
		Addr:    net.JoinHostPort("", strconv.Itoa(port)),
		Handler: mux,
	}

	if err := mgr.Add(&manager.Server{
		Name:   "metrics",
		Server: srv,
	}); err != nil {
		return err
	}
	return nil
}

// inject options that allow multiple test runs to run
// https://github.com/kubernetes-sigs/controller-runtime/issues/2937
func managerTestOptions(namespace, name string) ctrl.Options {
	return ctrl.Options{
		Scheme: scheme,
		Cache: cache.Options{
			ByObject: map[client.Object]cache.ByObject{
				&corev1.Pod{}: {
					Namespaces: map[string]cache.Config{
						namespace: {},
					},
				},
				&v1alpha2.InferencePool{}: {
					Namespaces: map[string]cache.Config{
						namespace: {
							FieldSelector: fields.SelectorFromSet(fields.Set{
								"metadata.name": name,
							}),
						},
					},
				},
				&v1alpha2.InferenceModel{}: {
					Namespaces: map[string]cache.Config{
						namespace: {},
					},
				},
			},
		},
		Controller: config.Controller{
			SkipNameValidation: boolPointer(true),
		},
	}
}

func boolPointer(b bool) *bool {
	return &b
}
