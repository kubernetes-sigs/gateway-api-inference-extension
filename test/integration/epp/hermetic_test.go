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

// Package epp contains hermetic integration tests for the External Processing Proxy (EPP), faking the backend pods to
// allow for precise control over their metrics and state.
package epp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
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
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/testing/protocmp"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
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
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/yaml"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/saturationdetector"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/scorer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/server"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
	epptestutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
	integrationutils "sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

const (
	testPoolName  = "vllm-llama3-8b-instruct-pool"
	testNamespace = "default"

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
	name, value string
}

func labelsToString(labels []label) string {
	var sb strings.Builder
	for i, l := range labels {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(fmt.Sprintf("%s=%q", l.name, l.value))
	}
	return sb.String()
}

func inferenceObjectiveRequestTotal(labels []label) string {
	return fmt.Sprintf(`
		# HELP inference_objective_request_total [ALPHA] Counter of inference objective requests broken out for each model and target model.
		# TYPE inference_objective_request_total counter
		inference_objective_request_total{%s} 1
		`, labelsToString(labels))
}

func inferencePoolReadyPods(v int, labels []label) string {
	return fmt.Sprintf(`
		# HELP inference_pool_ready_pods [ALPHA] The number of ready pods in the inference server pool.
		# TYPE inference_pool_ready_pods gauge
		inference_pool_ready_pods{%s} %d
		`, labelsToString(labels), v)
}

// newRequestTotal creates the expected Prometheus metric string for the `inference_objective_request_total` counter.
func newRequestTotal(model, target string) string {
	return inferenceObjectiveRequestTotal([]label{{"model_name", model}, {"target_model_name", target}})
}

// newReadyPods creates the expected Prometheus metric string for the `inference_pool_ready_pods` gauge.
func newReadyPods(count int) string {
	return inferencePoolReadyPods(count, []label{{"name", testPoolName}})
}

// expectRouteTo constructs the full set of expected gRPC responses for a successful request that is routed to a
// specific backend endpoint.
func expectRouteTo(endpoint, targetModel, prompt string) []*extProcPb.ProcessingResponse {
	bodyJSON := map[string]interface{}{
		"max_tokens":  100,
		"model":       targetModel,
		"prompt":      prompt,
		"temperature": 0,
	}
	bodyBytes, _ := json.Marshal(bodyJSON)

	return integrationutils.NewRequestBufferedResponse(
		endpoint,
		string(bodyBytes),
		&configPb.HeaderValueOption{Header: &configPb.HeaderValue{Key: "hi", RawValue: []byte("mom")}},
		&configPb.HeaderValueOption{Header: &configPb.HeaderValue{
			Key:      requtil.RequestIdHeaderKey,
			RawValue: []byte("test-request-id"),
		}},
	)
}

// testHarness encapsulates the setup and teardown for a single hermetic test run.
// It ensures that each test case runs with its own isolated environment, including a dedicated server runner, gRPC
// client, and Kubernetes resources.
type testHarness struct {
	t      *testing.T
	runner *server.ExtProcServerRunner
	client extProcPb.ExternalProcessor_ProcessClient
	cancel func()
}

// newTestHarness creates and initializes all components for a test.
// It's the factory for creating a fully isolated test environment (sans controller-runtime Manager and its global
// metrics registry).
func newTestHarness(
	t *testing.T,
	podAndMetrics map[*backend.Pod]*backendmetrics.MetricsState,
	sdConfig *saturationdetector.Config,

	uniqueSuffix string,
) *testHarness {
	runner, client, serverCancel, conn, clientCancel := setupTestInfrastructure(t, podAndMetrics, sdConfig, uniqueSuffix)
	return &testHarness{
		t:      t,
		runner: runner,
		client: client,
		cancel: func() {
			clientCancel()
			conn.Close()
			serverCancel()

			for pod := range podAndMetrics {
				podObj := epptestutil.MakePod(pod.NamespacedName.Name).
					Namespace(pod.NamespacedName.Namespace).Complete().ObjRef()
				if err := k8sClient.Delete(context.Background(), podObj); err != nil {
					t.Logf("Failed to delete pod %s: %v", podObj.GetName(), err)
				}
			}
		},
	}
}

func TestFullDuplexStreamed_KubeInferenceObjectiveRequest(t *testing.T) {
	defaultPods := []podState{{index: 0}}
	type testCase struct {
		name          string
		requests      []*extProcPb.ProcessingRequest
		pods          []podState
		wantResponses []*extProcPb.ProcessingResponse
		wantMetrics   map[string]string
		wantErr       bool
	}

	// runTestCases is a generic engine for executing a slice of test cases.
	// It handles the boilerplate of setting up the test harness, running the test,  validating the response, and checking
	// metrics for each case.
	runTestCases := func(t *testing.T, testCases []testCase, sdConfig *saturationdetector.Config) {
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				// These integration tests are run serially.
				// The controller-runtime Manager registers metrics to a global registry.
				// We ensure correctness by resetting the global metrics for each test run.
				metrics.Reset()
				t.Cleanup(metrics.Reset)

				uniqueSuffix := uuid.NewString()[:8]
				h := newTestHarness(t, newPodStates(uniqueSuffix, tc.pods...), sdConfig, uniqueSuffix)
				t.Cleanup(h.cancel)

				responses, err := integrationutils.StreamedRequest(t, h.client, tc.requests, len(tc.wantResponses))
				validateResponse(t, err, tc.wantErr, tc.wantResponses, responses)

				errs := make(map[string]error)
				assert.Eventually(t, func() bool {
					for metricName, value := range tc.wantMetrics {
						if err := metricsutils.GatherAndCompare(
							crmetrics.Registry,
							strings.NewReader(value),
							metricName,
						); err != nil {
							errs[metricName] = err
							return false
						}
					}
					return true
				}, 5*time.Second, 100*time.Millisecond, "failed to match all expected metrics")
				if len(errs) > 0 {
					for metricName, err := range errs {
						t.Logf("Metric comparison failed for %s: %v", metricName, err)
					}
				}
			})
		}
	}

	t.Run("RequestRouting", func(t *testing.T) {
		testCases := []testCase{
			{
				name: "selects pod with lower queue and kv cache",
				requests: integrationutils.GenerateStreamedRequestSetWithHeaders(logger, "test1", modelMyModel,
					modelMyModelTarget, nil, nil),
				pods: []podState{
					{index: 0, queueSize: 3, kvCacheUsage: 0.2},
					{index: 1, queueSize: 0, kvCacheUsage: 0.1},
					{index: 2, queueSize: 10, kvCacheUsage: 0.2},
				},
				wantResponses: expectRouteTo("192.168.1.2:8000", modelMyModelTarget, "test1"),
				wantMetrics: map[string]string{
					"inference_objective_request_total": newRequestTotal(modelMyModel, modelMyModelTarget),
					"inference_pool_ready_pods":         newReadyPods(3),
				},
			},
			{
				name: "selects pod with active lora and low queue",
				requests: integrationutils.GenerateStreamedRequestSetWithHeaders(logger, "test2", modelSQLLora,
					modelSQLLoraTarget, nil, nil),
				pods: []podState{
					{index: 0, queueSize: 0, kvCacheUsage: 0.2, activeModels: []string{"foo"}},
					{index: 1, queueSize: 0, kvCacheUsage: 0.1, activeModels: []string{"foo", modelSQLLoraTarget}},
					{index: 2, queueSize: 10, kvCacheUsage: 0.2, activeModels: []string{"foo"}},
				},
				wantResponses: expectRouteTo("192.168.1.2:8000", modelSQLLoraTarget, "test2"),
				wantMetrics: map[string]string{
					"inference_objective_request_total": newRequestTotal(modelSQLLora, modelSQLLoraTarget),
				},
			},
			{
				name: "selects pod with lora affinity despite higher kv cache",
				requests: integrationutils.GenerateStreamedRequestSetWithHeaders(logger, "test3", modelSQLLora,
					modelSQLLoraTarget, nil, nil),
				pods: []podState{
					{index: 0, queueSize: 10, kvCacheUsage: 0.2},
					{index: 1, queueSize: 10, kvCacheUsage: 0.4, activeModels: []string{modelSQLLoraTarget}},
					{index: 2, queueSize: 10, kvCacheUsage: 0.3},
				},
				wantResponses: expectRouteTo("192.168.1.2:8000", modelSQLLoraTarget, "test3"),
				wantMetrics: map[string]string{"inference_objective_request_total": newRequestTotal(modelSQLLora,
					modelSQLLoraTarget)},
			},
			{
				name: "routes to least-saturated pod when all pods are under high load",
				requests: integrationutils.GenerateStreamedRequestSetWithHeaders(logger, "test4", modelSQLLora,
					modelSQLLoraTarget, nil, nil),
				pods: []podState{
					{index: 0, queueSize: 6, kvCacheUsage: 0.2, activeModels: []string{modelSQLLoraTarget}},
					{index: 1, queueSize: 0, kvCacheUsage: 0.85},
					{index: 2, queueSize: 10, kvCacheUsage: 0.9},
				},
				wantResponses: expectRouteTo("192.168.1.1:8000", modelSQLLoraTarget, "test4"),
				wantMetrics: map[string]string{
					"inference_objective_request_total": newRequestTotal(modelSQLLora, modelSQLLoraTarget),
				},
			},
			{
				name: "passthrough for models not defined in objectives",
				requests: integrationutils.GenerateStreamedRequestSetWithHeaders(logger, "test6", modelDirect, modelDirect, nil,
					map[string]string{metadata.ModelNameRewriteKey: modelDirect}),
				pods: []podState{
					{index: 0, queueSize: 0, kvCacheUsage: 0.1},
					{index: 1, queueSize: 5, kvCacheUsage: 0.85},
					{index: 2, queueSize: 10, kvCacheUsage: 0.9},
				},
				wantResponses: expectRouteTo("192.168.1.1:8000", modelDirect, "test6"),
				wantMetrics: map[string]string{
					"inference_objective_request_total": newRequestTotal(modelDirect, modelDirect),
				},
			},
			{
				name: "routes request with multi-chunk body",
				requests: integrationutils.GenerateStreamedRequestSetWithHeaders(logger, "test", modelMyModel,
					modelMyModelTarget, nil, nil),
				pods: defaultPods,
				wantResponses: integrationutils.NewRequestBufferedResponse(
					"192.168.1.1:8000",
					`{"max_tokens":100,"model":"my-model-12345","prompt":"test","temperature":0}`,
					&configPb.HeaderValueOption{Header: &configPb.HeaderValue{Key: "hi", RawValue: []byte("mom")}},
					&configPb.HeaderValueOption{Header: &configPb.HeaderValue{
						Key:      requtil.RequestIdHeaderKey,
						RawValue: []byte("test-request-id"),
					}},
				),
				wantMetrics: map[string]string{
					"inference_objective_request_total": newRequestTotal(modelMyModel, modelMyModelTarget),
				},
			},
		}
		runTestCases(t, testCases, nil)
	})

	t.Run("ResponseHandling", func(t *testing.T) {
		testCases := []testCase{
			{
				name: "buffers and rewrites multi-chunk json response",
				requests: []*extProcPb.ProcessingRequest{
					{Request: &extProcPb.ProcessingRequest_RequestHeaders{RequestHeaders: &extProcPb.HttpHeaders{
						Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
							{Key: metadata.ObjectiveKey, Value: modelSheddable},
							{Key: metadata.ModelNameRewriteKey, Value: modelSheddableTarget},
							{Key: requtil.RequestIdHeaderKey, Value: "test-static-id-1"},
						},
						}, EndOfStream: true}}},
					{Request: &extProcPb.ProcessingRequest_ResponseHeaders{ResponseHeaders: &extProcPb.HttpHeaders{
						Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
							{Key: "content-type", Value: "application/json"},
						},
						}}}},
					{Request: &extProcPb.ProcessingRequest_ResponseBody{ResponseBody: &extProcPb.HttpBody{
						Body: []byte(`{"model":"` + modelSheddable + `", "prompt": "test"}`), EndOfStream: false},
					}},
					{Request: &extProcPb.ProcessingRequest_ResponseBody{ResponseBody: &extProcPb.HttpBody{
						Body: []byte(`}`), EndOfStream: true},
					}},
				},
				pods: defaultPods,
				wantResponses: []*extProcPb.ProcessingResponse{
					{
						Response: &extProcPb.ProcessingResponse_RequestHeaders{RequestHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{SetHeaders: []*configPb.HeaderValueOption{
									{Header: &configPb.HeaderValue{
										Key:      metadata.DestinationEndpointKey,
										RawValue: []byte("192.168.1.1:8000"),
									}},
									{Header: &configPb.HeaderValue{
										Key:      requtil.RequestIdHeaderKey,
										RawValue: []byte("test-static-id-1"),
									}},
								}},
							},
						}},
						DynamicMetadata: integrationutils.MakeMetadata("192.168.1.1:8000"),
					},
					integrationutils.NewResponseHeaders(
						&configPb.HeaderValueOption{Header: &configPb.HeaderValue{
							Key:      "content-type",
							RawValue: []byte("application/json"),
						}},
						&configPb.HeaderValueOption{Header: &configPb.HeaderValue{
							Key:      "x-went-into-resp-headers",
							RawValue: []byte("true"),
						}},
					),
					integrationutils.NewResponseStreamChunk(`{"model":"`+modelSheddable+`","prompt":"test"}`, true),
				},
			},
			{
				name: "handles invalid json in response body",
				requests: []*extProcPb.ProcessingRequest{
					{Request: &extProcPb.ProcessingRequest_RequestHeaders{RequestHeaders: &extProcPb.HttpHeaders{
						Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
							{Key: metadata.ObjectiveKey, Value: modelSheddable},
							{Key: metadata.ModelNameRewriteKey, Value: modelSheddableTarget},
							{Key: requtil.RequestIdHeaderKey, Value: "test-static-id-2"},
						},
						}, EndOfStream: true}}},
					{Request: &extProcPb.ProcessingRequest_ResponseHeaders{ResponseHeaders: &extProcPb.HttpHeaders{
						Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
							{Key: "content-type", Value: "application/json"},
						}},
					}}},
					{Request: &extProcPb.ProcessingRequest_ResponseBody{ResponseBody: &extProcPb.HttpBody{
						Body: []byte(`not valid json`), EndOfStream: true,
					}}},
				},
				pods: defaultPods,
				wantResponses: []*extProcPb.ProcessingResponse{
					{
						Response: &extProcPb.ProcessingResponse_RequestHeaders{RequestHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{SetHeaders: []*configPb.HeaderValueOption{
									{Header: &configPb.HeaderValue{
										Key:      metadata.DestinationEndpointKey,
										RawValue: []byte("192.168.1.1:8000"),
									}},
									{Header: &configPb.HeaderValue{
										Key:      requtil.RequestIdHeaderKey,
										RawValue: []byte("test-static-id-2"),
									}},
								}},
							},
						}},
						DynamicMetadata: integrationutils.MakeMetadata("192.168.1.1:8000"),
					},
					integrationutils.NewResponseHeaders(
						&configPb.HeaderValueOption{Header: &configPb.HeaderValue{
							Key:      "content-type",
							RawValue: []byte("application/json"),
						}},
						&configPb.HeaderValueOption{Header: &configPb.HeaderValue{
							Key:      "x-went-into-resp-headers",
							RawValue: []byte("true"),
						}},
					),
					integrationutils.NewResponseStreamChunk(`not valid json`, true),
				},
			},
			{
				name: "handles single chunk response followed by empty EOS chunk",
				requests: []*extProcPb.ProcessingRequest{
					{Request: &extProcPb.ProcessingRequest_RequestHeaders{RequestHeaders: &extProcPb.HttpHeaders{
						Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
							{Key: metadata.ObjectiveKey, Value: modelSheddable},
							{Key: metadata.ModelNameRewriteKey, Value: modelSheddableTarget},
							{Key: requtil.RequestIdHeaderKey, Value: "test-static-id-3"}},
						}, EndOfStream: true}}},
					{Request: &extProcPb.ProcessingRequest_ResponseHeaders{ResponseHeaders: &extProcPb.HttpHeaders{
						Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
							{Key: "content-type", Value: "application/json"},
						}},
					}}},
					{Request: &extProcPb.ProcessingRequest_ResponseBody{ResponseBody: &extProcPb.HttpBody{
						Body: []byte(`{"model":"` + modelSheddableTarget + `"}`), EndOfStream: false,
					}}},
					{Request: &extProcPb.ProcessingRequest_ResponseBody{ResponseBody: &extProcPb.HttpBody{
						Body: []byte(""), EndOfStream: true,
					}}},
				},
				pods: defaultPods,
				wantResponses: []*extProcPb.ProcessingResponse{
					{
						Response: &extProcPb.ProcessingResponse_RequestHeaders{RequestHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{SetHeaders: []*configPb.HeaderValueOption{
									{Header: &configPb.HeaderValue{
										Key: metadata.DestinationEndpointKey, RawValue: []byte("192.168.1.1:8000"),
									}},
									{Header: &configPb.HeaderValue{
										Key: requtil.RequestIdHeaderKey, RawValue: []byte("test-static-id-3"),
									}},
								}},
							},
						}},
						DynamicMetadata: integrationutils.MakeMetadata("192.168.1.1:8000"),
					},
					integrationutils.NewResponseHeaders(
						&configPb.HeaderValueOption{Header: &configPb.HeaderValue{
							Key: "content-type", RawValue: []byte("application/json"),
						}},
						&configPb.HeaderValueOption{Header: &configPb.HeaderValue{
							Key: "x-went-into-resp-headers", RawValue: []byte("true"),
						}},
					),
					integrationutils.NewResponseStreamChunk(`{"model":"`+modelSheddableTarget+`"}`, true),
				},
			},
			{
				name: "passes through and counts tokens in event-stream response",
				requests: []*extProcPb.ProcessingRequest{
					{Request: &extProcPb.ProcessingRequest_RequestHeaders{RequestHeaders: &extProcPb.HttpHeaders{
						Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
							{Key: metadata.ObjectiveKey, Value: modelSheddable},
							{Key: metadata.ModelNameRewriteKey, Value: modelSheddableTarget},
							{Key: requtil.RequestIdHeaderKey, Value: "test-static-id-4"},
						}}, EndOfStream: true,
					}}},
					{Request: &extProcPb.ProcessingRequest_ResponseHeaders{ResponseHeaders: &extProcPb.HttpHeaders{
						Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
							{Key: "content-type", Value: "text/event-stream"},
						}},
					}}},
					{Request: &extProcPb.ProcessingRequest_ResponseBody{ResponseBody: &extProcPb.HttpBody{
						Body:        []byte(`data: {"usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}`),
						EndOfStream: false,
					}}},
					{Request: &extProcPb.ProcessingRequest_ResponseBody{ResponseBody: &extProcPb.HttpBody{
						Body: []byte("\ndata: [DONE]"), EndOfStream: true,
					}}},
				},
				pods: defaultPods,
				wantMetrics: map[string]string{`inference_objective_input_tokens`: `
					# HELP inference_objective_input_tokens [ALPHA] Inference objective input token count distribution for requests in each model.
					# TYPE inference_objective_input_tokens histogram
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="1"} 0
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="8"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="16"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="32"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="64"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="128"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="256"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="512"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="1024"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="2048"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="4096"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="8192"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="16384"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="32778"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="65536"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="131072"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="262144"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="524288"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="1.048576e+06"} 1
					inference_objective_input_tokens_bucket{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3",le="+Inf"} 1
					inference_objective_input_tokens_sum{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3"} 7
					inference_objective_input_tokens_count{model_name="sql-lora-sheddable",target_model_name="sql-lora-1fdg3"} 1
					`},
				wantResponses: []*extProcPb.ProcessingResponse{
					{
						Response: &extProcPb.ProcessingResponse_RequestHeaders{RequestHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{SetHeaders: []*configPb.HeaderValueOption{
									{Header: &configPb.HeaderValue{
										Key:      metadata.DestinationEndpointKey,
										RawValue: []byte("192.168.1.1:8000"),
									}},
									{Header: &configPb.HeaderValue{
										Key:      requtil.RequestIdHeaderKey,
										RawValue: []byte("test-static-id-4"),
									}},
								}},
							},
						}},
						DynamicMetadata: integrationutils.MakeMetadata("192.168.1.1:8000"),
					},
					integrationutils.NewResponseHeaders(
						&configPb.HeaderValueOption{Header: &configPb.HeaderValue{
							Key:      "content-type",
							RawValue: []byte("text/event-stream"),
						}},
						&configPb.HeaderValueOption{Header: &configPb.HeaderValue{
							Key:      "x-went-into-resp-headers",
							RawValue: []byte("true"),
						}},
					),
					integrationutils.NewResponseStreamChunk(`data: {"usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}`, false),
					integrationutils.NewResponseStreamChunk("\ndata: [DONE]", true),
				},
			},
		}
		runTestCases(t, testCases, nil)
	})

	t.Run("Subsetting", func(t *testing.T) {
		testCases := []testCase{
			{
				name: "selects best pod from available subset",
				requests: integrationutils.GenerateStreamedRequestSetWithHeaders(logger, "subset-test", modelSQLLora,
					modelSQLLoraTarget, []string{"192.168.1.1:8000", "192.168.1.2:8000"}, nil),
				pods: []podState{
					{index: 0, queueSize: 5, kvCacheUsage: 0.2},
					{index: 1, queueSize: 0, kvCacheUsage: 0.1, activeModels: []string{modelSQLLoraTarget}},
					{index: 2, queueSize: 0, kvCacheUsage: 0.1},
				},
				wantResponses: expectRouteTo("192.168.1.2:8000", modelSQLLoraTarget, "subset-test"),
			},
			{
				name: "selects only available pod in subset despite high load",
				requests: integrationutils.GenerateStreamedRequestSetWithHeaders(logger, "subset-test", modelMyModel,
					modelMyModelTarget, []string{"192.168.1.3:8000"}, nil),
				pods: []podState{
					{index: 0, queueSize: 0, kvCacheUsage: 0.1},
					{index: 1, queueSize: 0, kvCacheUsage: 0.1},
					{index: 2, queueSize: 10, kvCacheUsage: 0.9},
				},
				wantResponses: expectRouteTo("192.168.1.3:8000", modelMyModelTarget, "subset-test"),
			},
			{
				name: "returns error when no pods match subset",
				requests: integrationutils.GenerateStreamedRequestSetWithHeaders(logger, "subset-test", modelMyModel,
					modelMyModelTarget, []string{"192.168.1.4:8000"}, nil),
				pods:    []podState{{index: 0}, {index: 1}, {index: 2}},
				wantErr: true,
				wantResponses: integrationutils.NewImmediateErrorResponse(
					envoyTypePb.StatusCode_ServiceUnavailable,
					"inference gateway: ServiceUnavailable - failed to find candidate pods for serving the request",
				),
			},
		}
		runTestCases(t, testCases, nil)
	})

	t.Run("ErrorConditions", func(t *testing.T) {
		testCases := []testCase{
			{
				name: "invalid json in request body",
				requests: []*extProcPb.ProcessingRequest{
					{Request: &extProcPb.ProcessingRequest_RequestHeaders{RequestHeaders: &extProcPb.HttpHeaders{
						Headers: &configPb.HeaderMap{},
					}}},
					{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{
						Body: []byte("not json"), EndOfStream: true,
					}}},
				},
				pods:    defaultPods,
				wantErr: true,
				wantResponses: integrationutils.NewImmediateErrorResponse(
					envoyTypePb.StatusCode_BadRequest,
					"inference gateway: BadRequest - Error unmarshaling request body",
				),
			},
			{
				name: "request body is missing model field",
				requests: []*extProcPb.ProcessingRequest{
					{Request: &extProcPb.ProcessingRequest_RequestHeaders{RequestHeaders: &extProcPb.HttpHeaders{
						Headers: &configPb.HeaderMap{},
					}}},
					{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{
						Body: []byte(`{"prompt":"test"}`), EndOfStream: true,
					}}},
				},
				pods:    defaultPods,
				wantErr: true,
				wantResponses: integrationutils.NewImmediateErrorResponse(
					envoyTypePb.StatusCode_BadRequest,
					"inference gateway: BadRequest - model not found in request body",
				),
			},
			{
				name: "no backend pods available in datastore",
				requests: []*extProcPb.ProcessingRequest{
					{Request: &extProcPb.ProcessingRequest_RequestHeaders{RequestHeaders: &extProcPb.HttpHeaders{
						Headers: &configPb.HeaderMap{}, EndOfStream: true,
					}}},
				},
				pods:    nil,
				wantErr: true,
				wantResponses: integrationutils.NewImmediateErrorResponse(
					envoyTypePb.StatusCode_InternalServerError,
					"inference gateway: Internal - no pods available in datastore",
				),
			},
		}
		runTestCases(t, testCases, nil)
	})

	t.Run("RequestTypes", func(t *testing.T) {
		testCases := []testCase{
			{
				name: "simple GET request is passed through",
				requests: []*extProcPb.ProcessingRequest{
					{Request: &extProcPb.ProcessingRequest_RequestHeaders{RequestHeaders: &extProcPb.HttpHeaders{
						Headers: &configPb.HeaderMap{}, EndOfStream: true}}},
				},
				pods:          defaultPods,
				wantResponses: nil, // Expect no modification, just pass-through.
			},
		}
		runTestCases(t, testCases, nil)
	})
}

// setupTestInfrastructure is the core setup engine for a single hermetic test case. It performs the following steps:
//  1. Creates a new controller-runtime manager with a cache scoped to a unique test ID to ensure resource isolation.
//  2. Starts the manager in the background.
//  3. Instantiates a new EPP server runner, configuring it with a fake metrics client and a real scheduler.
//  4. Creates the fake Kubernetes pod objects and injects their simulated metrics into the fake client.
//  5. Starts the EPP server in the background on a free port.
//  6. Waits until the manager's cache has synced and the EPP datastore is populated with the fake pods.
//  7. Creates and returns a gRPC client connected to the test server.
//
// It returns the server runner instance, the gRPC client, a function to stop the server and manager, the gRPC
// connection, and a function to cancel the client context.
func setupTestInfrastructure(
	t *testing.T,
	podAndMetrics map[*backend.Pod]*backendmetrics.MetricsState,
	sdConfig *saturationdetector.Config,
	uniqueSuffix string,
) (
	*server.ExtProcServerRunner,
	extProcPb.ExternalProcessor_ProcessClient,
	context.CancelFunc,
	*grpc.ClientConn,
	context.CancelFunc,
) {
	// --- 1. Create a Manager with a Pod Selector for this Test Run ---
	// We use a unique suffix for each test run to label pods.
	// The manager's cache is configured to only watch pods with this label, ensuring that tests do not interfere with
	// each other's backend pods.
	testRunLabel := "test-run-id"
	podSelector := labels.SelectorFromSet(map[string]string{testRunLabel: uniqueSuffix})
	mgr, err := server.NewManagerWithOptions(testEnv.Config, managerTestOptions(testNamespace, testPoolName, podSelector))
	require.NoError(t, err)

	// --- 2. Start the Manager and EPP Server ---
	// The manager is started in the background to handle Kubernetes watches.
	// The EPP server is configured to run on a random free port to avoid conflicts during parallel test execution.
	managerCtx, stopManager := context.WithCancel(context.Background())
	serverCtx, stopServer := context.WithCancel(context.Background())

	go func() {
		if err := mgr.Start(managerCtx); err != nil {
			if !errors.Is(err, context.Canceled) {
				t.Errorf("Failed to start manager: %v", err)
			}
		}
	}()

	kvCacheUtilizationScorer := scorer.NewKVCacheUtilizationScorer()
	queueingScorer := scorer.NewQueueScorer()
	prefixCacheScorer := prefix.New(context.Background(), prefix.DefaultConfig)
	loraAffinityScorer := scorer.NewLoraAffinityScorer()
	defaultProfile := framework.NewSchedulerProfile().
		WithScorers(
			framework.NewWeightedScorer(kvCacheUtilizationScorer, 1),
			framework.NewWeightedScorer(queueingScorer, 1),
			framework.NewWeightedScorer(prefixCacheScorer, 1),
			framework.NewWeightedScorer(loraAffinityScorer, 1),
		).
		WithPicker(picker.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
	profileHandler := profile.NewSingleProfileHandler()
	schedulerConfig := scheduling.NewSchedulerConfig(profileHandler, map[string]*framework.SchedulerProfile{
		"default": defaultProfile,
	})
	scheduler := scheduling.NewSchedulerWithConfig(schedulerConfig)

	runner := server.NewDefaultExtProcServerRunner()
	grpcAddress, err := integrationutils.GetFreePort()
	require.NoError(t, err)
	runner.GrpcPort = grpcAddress.Port
	runner.TestPodMetricsClient = &backendmetrics.FakePodMetricsClient{}
	pmf := backendmetrics.NewPodMetricsFactory(runner.TestPodMetricsClient, 10*time.Millisecond)

	runner.PoolGKNN = common.GKNN{
		NamespacedName: types.NamespacedName{Namespace: testNamespace, Name: testPoolName},
		GroupKind:      schema.GroupKind{Group: v1.GroupVersion.Group, Kind: "InferencePool"},
	}
	runner.Datastore = datastore.NewDatastore(context.Background(), pmf)
	runner.SecureServing = false

	if err := runner.SetupWithManager(context.Background(), mgr); err != nil {
		t.Fatalf("Failed to setup test-local server runner: %v", err)
	}

	// --- 3. Configure the Director and its Dependencies ---
	// The core EPP logic (scheduler, saturation detector) is wired up here.
	// This allows tests to enable/disable features and provide custom configurations.
	if sdConfig == nil {
		sdConfig = &saturationdetector.Config{
			QueueDepthThreshold:       saturationdetector.DefaultQueueDepthThreshold,
			KVCacheUtilThreshold:      saturationdetector.DefaultKVCacheUtilThreshold,
			MetricsStalenessThreshold: saturationdetector.DefaultMetricsStalenessThreshold,
		}
	}
	detector := saturationdetector.NewDetector(sdConfig, logger.WithName("saturation-detector"))
	runner.SaturationDetector = detector
	runner.Director = requestcontrol.NewDirectorWithConfig(runner.Datastore, scheduler, detector,
		requestcontrol.NewConfig())

	// --- 4. Create Fake Backend Pods and Metrics ---
	// The test harness creates fake Kubernetes pod objects and injects a fake metrics client into the server runner.
	// This gives each test precise control over the perceived state of the backend.
	res := map[types.NamespacedName]*backendmetrics.MetricsState{}
	for pod, metrics := range podAndMetrics {
		res[pod.NamespacedName] = metrics
	}
	runner.TestPodMetricsClient.SetRes(res)

	podLabels := map[string]string{
		"app":        testPoolName,
		testRunLabel: uniqueSuffix,
	}
	for pod := range podAndMetrics {
		podObj := epptestutil.MakePod(pod.NamespacedName.Name).
			Namespace(pod.NamespacedName.Namespace).
			ReadyCondition().
			Labels(podLabels).
			IP(pod.Address).
			Complete().
			ObjRef()
		copy := podObj.DeepCopy()
		if err := k8sClient.Create(context.Background(), copy); err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}
		copy.Status = podObj.Status
		if err := k8sClient.Status().Update(context.Background(), copy); err != nil {
			t.Fatalf("Failed to update pod status: %v", err)
		}
	}

	go func() {
		if err := runner.AsRunnable(logger.WithName("ext-proc")).Start(serverCtx); err != nil {
			if !errors.Is(err, context.Canceled) {
				t.Errorf("Failed to start ext-proc server: %v", err)
			}
		}
	}()

	// --- 5. Wait for Datastore Sync ---
	// We must wait for the controller-runtime cache to sync with the fake API server to ensure the EPP's datastore has
	// the correct view of pods and objectives before the test begins.
	// This is a critical step. We must block until the manager's cache has synced and propagated the pod and objective
	// resources to the EPP's datastore. Otherwise, the test will start before the EPP server is aware of any backend
	// pods, leading to guaranteed failures.
	assert.EventuallyWithT(t, func(t *assert.CollectT) {
		synced := runner.Datastore.PoolHasSynced()
		assert.True(t, synced, "Pool should be synced")
		assert.Len(t, runner.Datastore.PodList(backendmetrics.AllPodsPredicate), len(podAndMetrics), "Datastore not synced")
		assert.NotNil(t, runner.Datastore.ObjectiveGet(modelSheddable), "InferenceObjective not synced")
	}, 10*time.Second, 100*time.Millisecond)

	// --- 6. Set up the gRPC Client ---
	// Finally, a gRPC client is created to communicate with the EPP server.
	conn, err := grpc.NewClient(grpcAddress.String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to connect to %s: %v", testGRPCAddress, err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	client, err := extProcPb.NewExternalProcessorClient(conn).Process(ctx)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	stopAll := func() {
		stopServer()
		stopManager()
	}

	return runner, client, stopAll, conn, cancel
}

func fakePod(index int, uniqueSuffix string) *backend.Pod {
	return &backend.Pod{
		NamespacedName: types.NamespacedName{Name: fmt.Sprintf("pod-%d-%s", index, uniqueSuffix), Namespace: testNamespace},
		Address:        fmt.Sprintf("192.168.1.%d", index+1),
	}
}

// podState is a simplified descriptor for a pod's simulated metrics, used to define test scenarios.
type podState struct {
	index        int
	queueSize    int
	kvCacheUsage float64
	activeModels []string
}

// newPodStates is a test helper that converts a slice of simplified podState descriptors into the detailed
// `map[*backend.Pod]*backendmetrics.MetricsState` required by the fake metrics client.
func newPodStates(uniqueSuffix string, states ...podState) map[*backend.Pod]*backendmetrics.MetricsState {
	res := make(map[*backend.Pod]*backendmetrics.MetricsState)
	for _, s := range states {
		pod := fakePod(s.index, uniqueSuffix)
		actveModelsMap := make(map[string]int)
		for _, model := range s.activeModels {
			actveModelsMap[model] = 1
		}
		res[pod] = &backendmetrics.MetricsState{
			WaitingQueueSize:    s.queueSize,
			KVCacheUsagePercent: s.kvCacheUsage,
			ActiveModels:        actveModelsMap,
			WaitingModels:       make(map[string]int),
		}
	}
	return res
}

// validateResponse centralizes the validation logic for test responses. It correctly handles the expected `io.EOF`
// error for immediate responses and performs a detailed diff of the received protobuf messages against the expected
// ones.
func validateResponse(t *testing.T, err error, wantErr bool, wantResponses, responses []*extProcPb.ProcessingResponse) {
	if wantErr {
		// For immediate error responses, the server often closes the stream, resulting in `io.EOF`.
		// A nil error is also acceptable if the server sends the response then waits for the client to close.
		if err != nil {
			require.ErrorIs(t, err, io.EOF, "Expected EOF or nil error for immediate response stream")
		}
	} else {
		require.NoError(t, err)
	}

	if diff := cmp.Diff(wantResponses, responses,
		protocmp.Transform(),
		protocmp.SortRepeated(func(a, b *configPb.HeaderValueOption) bool {
			return a.GetHeader().GetKey() < b.GetHeader().GetKey()
		}),
	); diff != "" {
		t.Errorf("Unexpected response, (-want +got): %v", diff)
	}
}

// BeforeSuite sets up the hermetic test environment for the entire package. It starts a fake API server using envtest,
// creates a Kubernetes client, and pre-loads the Custom Resource Definitions (CRDs) and a common set of custom
// resources (like `InferencePool` and `InferenceObjective`) that are required by the test cases.
func BeforeSuite() func() {
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
	utilruntime.Must(v1.Install(scheme))

	k8sClient, err = k8sclient.New(cfg, k8sclient.Options{Scheme: scheme})
	if err != nil {
		logutil.Fatal(logger, err, "Failed to start k8s Client")
	}

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
		if err := k8sClient.Create(context.Background(), obj); err != nil {
			logutil.Fatal(logger, err, "Unable to create object", "object", obj.GetName())
		}
	}

	ctrl.SetLogger(logger)
	metrics.Register() // Register global metrics once for the entire test suite.

	logger.Info("Hermetic test suite setup complete")
	return func() {
		_ = testEnv.Stop()
	}
}

func readDocuments(fp string) ([][]byte, error) {
	b, err := os.ReadFile(fp)
	if err != nil {
		return nil, err
	}
	var docs [][]byte
	reader := k8syaml.NewYAMLReader(bufio.NewReader(bytes.NewReader(b)))
	for {
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

// managerTestOptions configures the controller-runtime manager for test isolation. Its most critical job is to
// configure the cache to only watch pods that have a specific `test-run-id` label.
// This ensures that the manager's cache for one test run is completely isolated from the pods created in another,
// preventing test interference. It also disables the metrics server to avoid port conflicts.
func managerTestOptions(namespace, name string, podSelector labels.Selector) ctrl.Options {
	return ctrl.Options{
		Scheme: scheme,
		Cache: cache.Options{
			ByObject: map[k8sclient.Object]cache.ByObject{
				&corev1.Pod{}: {
					Namespaces: map[string]cache.Config{
						namespace: {
							LabelSelector: podSelector,
						},
					},
				},
				&v1.InferencePool{}: {
					Namespaces: map[string]cache.Config{
						namespace: {
							FieldSelector: fields.SelectorFromSet(fields.Set{"metadata.name": name}),
						},
					},
				},
				&v1alpha2.InferenceObjective{}: {Namespaces: map[string]cache.Config{namespace: {}}},
			},
		},
		Controller: crconfig.Controller{SkipNameValidation: boolPointer(true)},
		Metrics: metricsserver.Options{
			BindAddress: "0", // Disable the metrics server in tests
		},
	}
}

func boolPointer(b bool) *bool {
	return &b
}
