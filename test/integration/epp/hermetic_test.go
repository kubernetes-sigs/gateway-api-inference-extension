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
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/testing/protocmp"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	k8syaml "k8s.io/apimachinery/pkg/util/yaml"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	metricsutils "k8s.io/component-base/metrics/testutil"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	k8sclient "sigs.k8s.io/controller-runtime/pkg/client"
	crconfig "sigs.k8s.io/controller-runtime/pkg/config"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	crmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/yaml"

	"sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/saturationdetector"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/filter"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/server"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	epptestutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
	integrationutils "sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

const (
	// Test Infrastructure
	testPoolName    = "vllm-llama3-8b-instruct-pool"
	testNamespace   = "default"
	testMetricsPort = 8889
	testPort        = server.DefaultGrpcPort

	// Model Names
	modelMyModel         = "my-model"
	modelMyModelTarget   = "my-model-12345"
	modelSQLLora         = "sql-lora"
	modelSQLLoraTarget   = "sql-lora-1fdg2"
	modelSheddable       = "sql-lora-sheddable"
	modelSheddableTarget = "sql-lora-1fdg3"
	modelDirect          = "direct-model"
)

var (
	testGRPCAddress = fmt.Sprintf("localhost:%d", server.DefaultGrpcPort)
	serverRunner    *server.ExtProcServerRunner
	k8sClient       k8sclient.Client
	testEnv         *envtest.Environment
	scheme          = runtime.NewScheme()
	logger          = logutil.NewTestLogger().V(logutil.VERBOSE)
)

func TestMain(m *testing.M) {
	cleanup := BeforeSuite()
	code := m.Run()
	cleanup()
	os.Exit(code)
}

type label struct {
	name,
	value string
}

func labelsToString(labels []label) string {
	var sb strings.Builder
	i := 0
	for _, l := range labels {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(fmt.Sprintf("%s=%q", l.name, l.value))
		i++
	}
	return sb.String()
}

func inferenceModelRequestTotal(labels []label) string {
	return fmt.Sprintf(`
		# HELP inference_model_request_total [ALPHA] Counter of inference model requests broken out for each model and target model.
		# TYPE inference_model_request_total counter
		inference_model_request_total{%s} 1
		`, labelsToString(labels),
	)
}

func inferencePoolReadyPods(v int, labels []label) string {
	return fmt.Sprintf(`
		# HELP inference_pool_ready_pods [ALPHA] The number of ready pods in the inference server pool.
		# TYPE inference_pool_ready_pods gauge
		inference_pool_ready_pods{%s} %d
		`, labelsToString(labels), v,
	)
}

func TestFullDuplexStreamed_KubeInferenceModelRequest(t *testing.T) {
	tests := []struct {
		name              string
		requests          []*extProcPb.ProcessingRequest
		pods              map[*backend.Pod]*backendmetrics.MetricsState
		wantResponses     []*extProcPb.ProcessingResponse
		wantMetrics       map[string]string
		wantErr           bool
		immediateResponse *extProcPb.ImmediateResponse
	}{
		// Request flow tests
		{
			name:     "select lower queue and kv cache, no active lora",
			requests: integrationutils.GenerateStreamedRequestSet(logger, "test1", modelMyModel, nil),
			// Pod 1 will be picked because it has relatively low queue size and low KV cache.
			pods: newPodStates(
				podState{index: 0, queueSize: 3, kvCacheUsage: 0.2},
				podState{index: 1, queueSize: 0, kvCacheUsage: 0.1},
				podState{index: 2, queueSize: 10, kvCacheUsage: 0.2},
			),
			wantMetrics: map[string]string{
				"inference_model_request_total": inferenceModelRequestTotal([]label{
					{"model_name", modelMyModel},
					{"target_model_name", modelMyModelTarget},
				}),
				"inference_pool_ready_pods": inferencePoolReadyPods(3, []label{
					{"name", testPoolName},
				}),
			},
			wantErr: false,
			wantResponses: integrationutils.NewRequestBufferedResponse(
				"192.168.1.2:8000",
				fmt.Sprintf(`{"max_tokens":100,"model":%q,"prompt":"test1","temperature":0}`, modelMyModelTarget),
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      "hi",
						RawValue: []byte("mom"),
					},
				},
			),
		},
		{
			name: "invalid json; return body",
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
						RequestBody: &extProcPb.HttpBody{Body: []byte("no healthy upstream"), EndOfStream: true},
					},
				},
			},
			// Pod 1 will be picked because it has relatively low queue size, the requested model active, and low KV cache.
			pods: newPodStates(
				podState{index: 0, queueSize: 0, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar"}},
				podState{index: 1, queueSize: 0, kvCacheUsage: 0.1, activeModels: []string{"foo", modelSQLLoraTarget}},
				podState{index: 2, queueSize: 10, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar"}},
			),
			wantErr: false,
			wantResponses: integrationutils.NewImmediateErrorResponse(
				envoyTypePb.StatusCode_BadRequest,
				"inference gateway: BadRequest - Error unmarshaling request body: no healthy upstream",
			),
		},
		{
			name:     "select active lora, low queue",
			requests: integrationutils.GenerateStreamedRequestSet(logger, "test2", modelSQLLora, nil),
			// Pod 1 will be picked because it has relatively low queue size, the requested model active, and low KV cache.
			pods: newPodStates(
				podState{index: 0, queueSize: 0, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar"}},
				podState{index: 1, queueSize: 0, kvCacheUsage: 0.1, activeModels: []string{"foo", modelSQLLoraTarget}},
				podState{index: 2, queueSize: 10, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar"}},
			),

			wantMetrics: map[string]string{
				"inference_model_request_total": inferenceModelRequestTotal([]label{
					{"model_name", modelSQLLora},
					{"target_model_name", modelSQLLoraTarget},
				}),
			},
			wantErr: false,
			wantResponses: integrationutils.NewRequestBufferedResponse(
				"192.168.1.2:8000",
				fmt.Sprintf(`{"max_tokens":100,"model":%q,"prompt":"test2","temperature":0}`, modelSQLLoraTarget),
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      "hi",
						RawValue: []byte("mom"),
					},
				},
			),
		},
		{
			name:     "select no lora despite active model, avoid excessive queue size",
			requests: integrationutils.GenerateStreamedRequestSet(logger, "test3", modelSQLLora, nil),
			// Pod 2 will be picked despite NOT having the requested model active as it is above the affinity for queue size.
			// Also it is critical, so we should still admit the request despite all queue sizes being greater than the queue
			// size threshold.
			pods: newPodStates(
				podState{index: 0, queueSize: 10, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar"}},
				podState{index: 1, queueSize: 200, kvCacheUsage: 0.1, activeModels: []string{"foo", modelSQLLoraTarget}},
				podState{index: 2, queueSize: 6, kvCacheUsage: 0.2, activeModels: []string{"foo"}},
			),
			wantMetrics: map[string]string{
				"inference_model_request_total": inferenceModelRequestTotal([]label{
					{"model_name", modelSQLLora},
					{"target_model_name", modelSQLLoraTarget},
				}),
			},
			wantErr: false,
			wantResponses: integrationutils.NewRequestBufferedResponse(
				"192.168.1.3:8000",
				fmt.Sprintf(`{"max_tokens":100,"model":%q,"prompt":"test3","temperature":0}`, modelSQLLoraTarget),
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      "hi",
						RawValue: []byte("mom"),
					},
				},
			),
		},
		{
			name:     "noncritical and all models past threshold, shed request",
			requests: integrationutils.GenerateStreamedRequestSet(logger, "test4", modelSheddable, nil),
			// pod 0: excluded; above queue size threshold
			// pod 1: excluded; above KV cache threshold
			// pod 2: excluded; above queue size threshold
			pods: newPodStates(
				podState{index: 0, queueSize: 6, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar", modelSheddableTarget}},
				podState{index: 1, queueSize: 0, kvCacheUsage: 0.85, activeModels: []string{"foo", modelSheddableTarget}},
				podState{index: 2, queueSize: 10, kvCacheUsage: 0.9, activeModels: []string{"foo", modelSheddableTarget}},
			),
			wantErr:     false,
			wantMetrics: map[string]string{},
			wantResponses: integrationutils.NewImmediateErrorResponse(
				envoyTypePb.StatusCode_TooManyRequests,
				"inference gateway: InferencePoolResourceExhausted - system saturated, non-critical request dropped",
			),
		},
		{
			name:     "noncritical, but one server has capacity, do not shed",
			requests: integrationutils.GenerateStreamedRequestSet(logger, "test5", modelSheddable, nil),
			// Pod 1 will be picked because it has relatively low queue size and low KV cache.
			pods: newPodStates(
				podState{index: 0, queueSize: 4, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar", modelSheddableTarget}},
				podState{index: 1, queueSize: 0, kvCacheUsage: 0.85, activeModels: []string{"foo", modelSheddableTarget}},
				podState{index: 2, queueSize: 10, kvCacheUsage: 0.9, activeModels: []string{"foo", modelSheddableTarget}},
			),
			wantMetrics: map[string]string{
				"inference_model_request_total": inferenceModelRequestTotal([]label{
					{"model_name", modelSheddable},
					{"target_model_name", modelSheddableTarget},
				}),
			},
			wantErr: false,
			wantResponses: integrationutils.NewRequestBufferedResponse(
				"192.168.1.2:8000",
				fmt.Sprintf(`{"max_tokens":100,"model":%q,"prompt":"test5","temperature":0}`, modelSheddableTarget),
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      "hi",
						RawValue: []byte("mom"),
					},
				},
			),
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
			// Pod 1 will be picked because it has relatively low queue size and low KV cache.
			pods: newPodStates(
				podState{index: 0, queueSize: 4, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar", modelSheddableTarget}},
				podState{index: 1, queueSize: 0, kvCacheUsage: 0.85, activeModels: []string{"foo", modelSheddableTarget}},
				podState{index: 2, queueSize: 10, kvCacheUsage: 0.9, activeModels: []string{"foo", modelSheddableTarget}},
			),
			wantMetrics: map[string]string{
				"inference_model_request_total": inferenceModelRequestTotal([]label{
					{"model_name", modelSheddable},
					{"target_model_name", modelSheddableTarget},
				}),
			},
			wantErr: false,
			wantResponses: integrationutils.NewRequestBufferedResponse(
				"192.168.1.2:8000",
				fmt.Sprintf(`{"max_tokens":100,"model":%q,"prompt":"test6","temperature":0}`, modelSheddableTarget),
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      "hi",
						RawValue: []byte("mom"),
					},
				},
			),
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
			// pod 0: selected
			// pod 1: excluded; above KV cache threshold
			// pod 2: excluded; above queue size threshold
			pods: newPodStates(
				podState{index: 0, queueSize: 4, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar", modelSheddableTarget}},
				podState{index: 1, queueSize: 0, kvCacheUsage: 0.85, activeModels: []string{"foo", modelSheddableTarget}},
				podState{index: 2, queueSize: 10, kvCacheUsage: 0.9, activeModels: []string{"foo", modelSheddableTarget}},
			),
			wantMetrics: map[string]string{
				"inference_model_request_total": inferenceModelRequestTotal([]label{
					{"model_name", modelDirect},
					{"target_model_name", modelDirect},
				}),
			},
			wantErr: false,
			wantResponses: integrationutils.NewRequestBufferedResponse(
				"192.168.1.2:8000",
				fmt.Sprintf(`{"max_tokens":100,"model":%q,"prompt":"test6","temperature":0}`, modelDirect),
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      "hi",
						RawValue: []byte("mom"),
					},
				},
			),
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
			// pod 0: selected
			// pod 1: excluded; above KV cache threshold
			// pod 2: excluded; above queue size threshold
			pods: newPodStates(
				podState{index: 0, queueSize: 4, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar", modelSheddableTarget}},
				podState{index: 1, queueSize: 0, kvCacheUsage: 0.85, activeModels: []string{"foo", modelSheddableTarget}},
				podState{index: 2, queueSize: 10, kvCacheUsage: 0.9, activeModels: []string{"foo", modelSheddableTarget}},
			),
			wantErr: false,
			wantResponses: integrationutils.NewResponseBufferedResponse(
				fmt.Sprintf(`{"max_tokens":100,"model":%q,"prompt":"test6","temperature":0}`, modelSheddable),
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      "x-went-into-resp-headers",
						RawValue: []byte("true"),
					},
				},
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      "content-type",
						RawValue: []uint8("application/json"),
					},
				},
			),
		},
		{
			name: "Response is invalid json; return body",
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
						ResponseBody: &extProcPb.HttpBody{Body: []byte("no healthy upstream"), EndOfStream: true},
					},
				},
			},
			// pod 0: selected
			// pod 1: excluded; above KV cache threshold
			// pod 2: excluded; above queue size threshold
			pods: newPodStates(
				podState{index: 0, queueSize: 4, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar", modelSheddableTarget}},
				podState{index: 1, queueSize: 0, kvCacheUsage: 0.85, activeModels: []string{"foo", modelSheddableTarget}},
				podState{index: 2, queueSize: 10, kvCacheUsage: 0.9, activeModels: []string{"foo", modelSheddableTarget}},
			),
			wantErr: false,
			wantResponses: integrationutils.NewResponseBufferedResponse(
				"no healthy upstream",
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      "x-went-into-resp-headers",
						RawValue: []byte("true"),
					},
				},
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      "content-type",
						RawValue: []uint8("application/json"),
					},
				},
			),
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
			// pod 0: selected
			// pod 1: excluded; above KV cache threshold
			// pod 2: excluded; above queue size threshold
			pods: newPodStates(
				podState{index: 0, queueSize: 4, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar", modelSheddableTarget}},
				podState{index: 1, queueSize: 0, kvCacheUsage: 0.85, activeModels: []string{"foo", modelSheddableTarget}},
				podState{index: 2, queueSize: 10, kvCacheUsage: 0.9, activeModels: []string{"foo", modelSheddableTarget}},
			),
			wantErr: false,
			wantResponses: integrationutils.NewResponseBufferedResponse(
				fmt.Sprintf(`{"max_tokens":100,"model":%q,"prompt":"test6","temperature":0}`, modelSheddable),
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      "x-went-into-resp-headers",
						RawValue: []byte("true"),
					},
				},
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      "content-type",
						RawValue: []uint8("application/json"),
					},
				},
			),
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
							Body:        []byte("data: {\"id\":\"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9\",\"object\":\"text_completion\",\"created\":1741379018,\"model\":\"food-review-1\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"total_tokens\":17,\"completion_tokens\":10}}\ndata: [DONE]"),
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
				integrationutils.NewResponseHeaders(
					&configPb.HeaderValueOption{
						Header: &configPb.HeaderValue{
							Key:      "x-went-into-resp-headers",
							RawValue: []byte("true"),
						},
					},
					&configPb.HeaderValueOption{
						Header: &configPb.HeaderValue{
							Key:      "content-type",
							RawValue: []byte("text/event-stream"),
						},
					},
					&configPb.HeaderValueOption{
						Header: &configPb.HeaderValue{
							Key:      "status",
							RawValue: []byte("200"),
						},
					},
				),
				integrationutils.NewResponseStreamChunk(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[{"index":0,"text":"NEVER","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`, false),
				integrationutils.NewResponseStreamChunk(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[{"index":0,"text":"GONNA","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`, false),
				integrationutils.NewResponseStreamChunk(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[{"index":0,"text":"GIVE","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`, false),
				integrationutils.NewResponseStreamChunk(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[{"index":0,"text":"YOU","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`, false),
				integrationutils.NewResponseStreamChunk(`data: {"id":"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9","object":"text_completion","created":1741379018,"model":"food-review-1","choices":[{"index":0,"text":"UP","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}`, false),
				integrationutils.NewResponseStreamChunk("data: {\"id\":\"cmpl-0fee233f-7d56-404a-acd3-4dad775d03d9\",\"object\":\"text_completion\",\"created\":1741379018,\"model\":\"food-review-1\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"total_tokens\":17,\"completion_tokens\":10}}\ndata: [DONE]", false),
				integrationutils.NewResponseStreamChunk("", true),
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
			wantResponses: []*extProcPb.ProcessingResponse{},
			pods: newPodStates(
				podState{index: 0, queueSize: 4, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar", modelSheddableTarget}},
			),
			wantMetrics: map[string]string{},
		},
		{
			name: "select active lora with subsetting tag, all pods available",
			requests: integrationutils.GenerateStreamedRequestSet(
				logger,
				"test2",
				modelSQLLora,
				[]string{"192.168.1.1:8000", "192.168.1.2:8000", "192.168.1.3:8000"}),
			// Pod 1 will be picked because it has relatively low queue size, the requested model active, low KV cache, and within subset.
			pods: newPodStates(
				podState{index: 0, queueSize: 0, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar"}},
				podState{index: 1, queueSize: 0, kvCacheUsage: 0.1, activeModels: []string{"foo", modelSQLLoraTarget}},
				podState{index: 2, queueSize: 10, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar"}},
			),

			wantMetrics: map[string]string{
				"inference_model_request_total": inferenceModelRequestTotal([]label{
					{"model_name", modelSQLLora},
					{"target_model_name", modelSQLLoraTarget},
				}),
			},
			wantErr: false,
			wantResponses: integrationutils.NewRequestBufferedResponse(
				"192.168.1.2:8000",
				fmt.Sprintf(`{"max_tokens":100,"model":%q,"prompt":"test2","temperature":0}`, modelSQLLoraTarget),
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      "hi",
						RawValue: []byte("mom"),
					},
				},
			),
		},
		{
			name: "select active lora with subsetting tag, some pods match",
			requests: integrationutils.GenerateStreamedRequestSet(
				logger,
				"test2",
				modelSQLLora,
				[]string{"192.168.1.3:8000"}),
			// Pod 3 has high queue and kv cache utilization, but it will still be picked because it is the only one matching subsetting target.
			pods: newPodStates(
				podState{index: 0, queueSize: 0, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar"}},
				podState{index: 1, queueSize: 0, kvCacheUsage: 0.1, activeModels: []string{"foo", modelSQLLoraTarget}},
				podState{index: 2, queueSize: 10, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar"}},
			),

			wantMetrics: map[string]string{
				"inference_model_request_total": inferenceModelRequestTotal([]label{
					{"model_name", modelSQLLora},
					{"target_model_name", modelSQLLoraTarget},
				}),
			},
			wantErr: false,
			wantResponses: integrationutils.NewRequestBufferedResponse(
				"192.168.1.3:8000",
				fmt.Sprintf(`{"max_tokens":100,"model":%q,"prompt":"test2","temperature":0}`, modelSQLLoraTarget),
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      "hi",
						RawValue: []byte("mom"),
					},
				},
			),
		},
		{
			name: "select active lora with subsetting tag, no pods available",
			requests: integrationutils.GenerateStreamedRequestSet(
				logger,
				"test2",
				modelSQLLora,
				[]string{"192.168.1.4:8000", "192.168.1.5:8000", "192.168.1.6:8000"}),
			// No pods will be picked as none are within the subset.
			pods: newPodStates(
				podState{index: 0, queueSize: 0, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar"}},
				podState{index: 1, queueSize: 0, kvCacheUsage: 0.1, activeModels: []string{"foo", modelSQLLoraTarget}},
				podState{index: 2, queueSize: 10, kvCacheUsage: 0.2, activeModels: []string{"foo", "bar"}},
			),

			wantMetrics: map[string]string{},
			wantErr:     true,
			wantResponses: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_ImmediateResponse{
						ImmediateResponse: &extProcPb.ImmediateResponse{
							Status: &envoyTypePb.HttpStatus{
								Code: envoyTypePb.StatusCode_ServiceUnavailable,
							},
							Body: []byte("inference gateway: ServiceUnavailable - failed to find candidate pods for serving the request"),
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, cleanup := setUpHermeticServer(t, test.pods)
			t.Cleanup(cleanup)
			responses, err := integrationutils.StreamedRequest(t, client, test.requests, len(test.wantResponses))

			if err != nil && !test.wantErr {
				t.Errorf("Unexpected error, got: %v, want error: %v", err, test.wantErr)
			}
			if diff := cmp.Diff(test.wantResponses, responses,
				protocmp.Transform(),
				protocmp.SortRepeated(func(a, b *configPb.HeaderValueOption) bool {
					return a.GetHeader().GetKey() < b.GetHeader().GetKey()
				}),
			); diff != "" {
				t.Errorf("Unexpected response, (-want +got): %v", diff)
			}

			if len(test.wantMetrics) != 0 {
				for metricName, value := range test.wantMetrics {
					if err := metricsutils.GatherAndCompare(crmetrics.Registry, strings.NewReader(value), metricName); err != nil {
						t.Error(err)
					}
				}
			}
			metrics.Reset()
		})
	}
}

func setUpHermeticServer(t *testing.T, podAndMetrics map[*backend.Pod]*backendmetrics.MetricsState) (client extProcPb.ExternalProcessor_ProcessClient, cleanup func()) {
	// Reconfigure the TestPodMetricsClient.
	res := map[types.NamespacedName]*backendmetrics.MetricsState{}
	for pod, metrics := range podAndMetrics {
		res[pod.NamespacedName] = metrics
	}
	serverRunner.TestPodMetricsClient.SetRes(res)

	serverCtx, stopServer := context.WithCancel(context.Background())

	// TODO: this should be consistent with the inference pool
	podLabels := map[string]string{
		"app": testPoolName,
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

	// Create a grpc connection
	conn, err := grpc.NewClient(testGRPCAddress, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		logutil.Fatal(logger, err, "Failed to connect", "address", testGRPCAddress)
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

func fakePod(index int) *backend.Pod {
	return &backend.Pod{
		NamespacedName: types.NamespacedName{Name: fmt.Sprintf("pod-%v", index), Namespace: testNamespace},
		Address:        fmt.Sprintf("192.168.1.%d", index+1),
		Labels:         make(map[string]string, 0),
	}
}

// podState is a descriptor for a pod's simulated metrics.
type podState struct {
	index        int
	queueSize    int
	kvCacheUsage float64
	activeModels []string
}

// newPodStates generates the backend metrics map required by the test setup.
func newPodStates(states ...podState) map[*backend.Pod]*backendmetrics.MetricsState {
	res := make(map[*backend.Pod]*backendmetrics.MetricsState)
	for _, s := range states {
		pod := fakePod(s.index)
		activeModelsMap := make(map[string]int)
		for _, model := range s.activeModels {
			activeModelsMap[model] = 1
		}
		res[pod] = &backendmetrics.MetricsState{
			WaitingQueueSize:    s.queueSize,
			KVCacheUsagePercent: s.kvCacheUsage,
			ActiveModels:        activeModelsMap,
			WaitingModels:       make(map[string]int),
		}
	}
	return res
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

	metrics.Register()
	// Register metrics handler.
	// Metrics endpoint is enabled in 'config/default/kustomization.yaml'. The Metrics options configure the server.
	// More info:
	// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.19.1/pkg/metrics/server
	// - https://book.kubebuilder.io/reference/metrics.html
	metricsServerOptions := metricsserver.Options{
		BindAddress:    fmt.Sprintf(":%d", testMetricsPort),
		FilterProvider: filters.WithAuthenticationAndAuthorization,
	}
	mgr, err := server.NewManagerWithOptions(cfg, managerTestOptions(testNamespace, testPoolName, metricsServerOptions))
	if err != nil {
		logutil.Fatal(logger, err, "Failed to create controller manager")
	}

	serverRunner = server.NewDefaultExtProcServerRunner()
	serverRunner.TestPodMetricsClient = &backendmetrics.FakePodMetricsClient{}
	pmf := backendmetrics.NewPodMetricsFactory(serverRunner.TestPodMetricsClient, 10*time.Millisecond)
	// Adjust from defaults
	serverRunner.PoolNamespacedName = types.NamespacedName{Name: testPoolName, Namespace: testNamespace}
	serverRunner.Datastore = datastore.NewDatastore(context.Background(), pmf)

	loraAffinityFilter := filter.NewLoraAffinityFilter(config.Conf.LoraAffinityThreshold)
	leastQueueFilter := filter.NewLeastQueueFilter()
	leastKvCacheFilter := filter.NewLeastKVCacheFilter()

	lowLatencyFilter := &filter.DecisionTreeFilter{
		Current: filter.NewLowQueueFilter(config.Conf.QueueingThresholdLoRA),
		NextOnSuccess: &filter.DecisionTreeFilter{
			Current: loraAffinityFilter,
			NextOnSuccessOrFailure: &filter.DecisionTreeFilter{
				Current: leastQueueFilter,
				NextOnSuccessOrFailure: &filter.DecisionTreeFilter{
					Current: leastKvCacheFilter,
				},
			},
		},
		NextOnFailure: &filter.DecisionTreeFilter{
			Current: leastQueueFilter,
			NextOnSuccessOrFailure: &filter.DecisionTreeFilter{
				Current: loraAffinityFilter,
				NextOnSuccessOrFailure: &filter.DecisionTreeFilter{
					Current: leastKvCacheFilter,
				},
			},
		},
	}

	defaultProfile := framework.NewSchedulerProfile().
		WithFilters(lowLatencyFilter).
		WithPicker(picker.NewRandomPicker(picker.DefaultMaxNumOfEndpoints))

	profileHandler := profile.NewSingleProfileHandler()

	schedulerConfig := scheduling.NewSchedulerConfig(profileHandler, map[string]*framework.SchedulerProfile{"default": defaultProfile})
	scheduler := scheduling.NewSchedulerWithConfig(schedulerConfig)

	sdConfig := &saturationdetector.Config{
		QueueDepthThreshold:       saturationdetector.DefaultQueueDepthThreshold,
		KVCacheUtilThreshold:      saturationdetector.DefaultKVCacheUtilThreshold,
		MetricsStalenessThreshold: saturationdetector.DefaultMetricsStalenessThreshold,
	}
	detector := saturationdetector.NewDetector(sdConfig, serverRunner.Datastore, logger.WithName("saturation-detector"))
	serverRunner.SaturationDetector = detector
	serverRunner.Director = requestcontrol.NewDirectorWithConfig(serverRunner.Datastore, scheduler, detector, requestcontrol.NewConfig())
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
		modelExist := serverRunner.Datastore.ModelGet(modelMyModel)
		synced := serverRunner.Datastore.PoolHasSynced() && modelExist != nil
		return synced
	}, 10*time.Second, 10*time.Millisecond)

	return func() {
		_ = testEnv.Stop()
		_ = k8sClient.DeleteAllOf(context.Background(), &v1.InferencePool{})
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

// inject options that allow multiple test runs to run
// https://github.com/kubernetes-sigs/controller-runtime/issues/2937
func managerTestOptions(namespace, name string, metricsServerOptions metricsserver.Options) ctrl.Options {
	return ctrl.Options{
		Scheme: scheme,
		Cache: cache.Options{
			ByObject: map[k8sclient.Object]cache.ByObject{
				&corev1.Pod{}: {
					Namespaces: map[string]cache.Config{
						namespace: {},
					},
				},
				&v1.InferencePool{}: {
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
		Controller: crconfig.Controller{
			SkipNameValidation: boolPointer(true),
		},
		Metrics: metricsServerOptions,
	}
}

func boolPointer(b bool) *bool {
	return &b
}
