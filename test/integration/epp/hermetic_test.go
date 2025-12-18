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
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zapcore"
	"google.golang.org/protobuf/testing/protocmp"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/yaml"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
	"sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

const (
	testPoolName         = "vllm-llama3-8b-instruct-pool"
	modelMyModel         = "my-model"
	modelMyModelTarget   = "my-model-12345"
	modelSQLLora         = "sql-lora"
	modelSQLLoraTarget   = "sql-lora-1fdg2"
	modelSheddable       = "sql-lora-sheddable"
	modelSheddableTarget = "sql-lora-1fdg3"
	modelDirect          = "direct-model"
	modelToBeWritten     = "model-to-be-rewritten"
	modelAfterRewrite    = "rewritten-model"
)

// Global State (Initialized in TestMain)
var (
	k8sClient     client.Client
	testEnv       *envtest.Environment
	testScheme    = runtime.NewScheme()
	logger        = zap.New(zap.UseDevMode(true), zap.Level(zapcore.Level(logutil.DEFAULT)))
	baseResources []*unstructured.Unstructured
)

func TestMain(m *testing.M) {
	ctrl.SetLogger(logger)

	// 1. EnvTest Setup (API Server + Etcd)
	testEnv = &envtest.Environment{
		CRDDirectoryPaths:     []string{filepath.Join("..", "..", "..", "config", "crd", "bases")},
		ErrorIfCRDPathMissing: true,
	}
	cfg, err := testEnv.Start()
	if err != nil {
		panic(fmt.Sprintf("failed to start test environment: %v", err))
	}

	// 2. Client & Scheme Registration
	utilruntime.Must(clientgoscheme.AddToScheme(testScheme))
	utilruntime.Must(v1alpha2.Install(testScheme))
	utilruntime.Must(v1.Install(testScheme))
	k8sClient, err = client.New(cfg, client.Options{Scheme: testScheme})
	if err != nil {
		panic(err)
	}

	// 3. Global Metric Registration
	// Necessary because we cannot parallelize tests using the global registry.
	metrics.Register()

	// 4. Pre-parse Base Resources
	// We load the YAML once here to avoid unnecessary I/O in every test case.
	baseResources = loadBaseResources()

	code := m.Run()

	_ = testEnv.Stop()
	os.Exit(code)
}

func TestFullDuplexStreamed_KubeInferenceObjectiveRequest(t *testing.T) {
	tests := []struct {
		name          string
		requests      []*extProcPb.ProcessingRequest
		pods          []podState
		wantResponses []*extProcPb.ProcessingResponse
		wantMetrics   map[string]string
		waitForModel  string
	}{
		// --- Standard Routing Logic ---
		{
			name:     "select lower queue and kv cache",
			requests: integration.ReqLLM(logger, "test1", modelMyModel, modelMyModelTarget),
			pods: []podState{
				P(0, 3, 0.2),
				P(1, 0, 0.1), // Winner (Low Queue, Low KV)
				P(2, 10, 0.2),
			},
			wantResponses: ExpectRouteTo("192.168.1.2:8000", modelMyModelTarget, "test1"),
			wantMetrics: map[string]string{
				"inference_objective_request_total": cleanMetric(metricReqTotal(modelMyModel, modelMyModelTarget)),
				"inference_pool_ready_pods":         cleanMetric(metricReadyPods(3)),
			},
		},
		{
			name:     "select active lora, low queue",
			requests: integration.ReqLLM(logger, "test2", modelSQLLora, modelSQLLoraTarget),
			pods: []podState{
				P(0, 0, 0.2, "foo", "bar"),
				P(1, 0, 0.1, "foo", modelSQLLoraTarget), // Winner (Has LoRA)
				P(2, 10, 0.2, "foo", "bar"),
			},
			wantResponses: ExpectRouteTo("192.168.1.2:8000", modelSQLLoraTarget, "test2"),
			wantMetrics: map[string]string{
				"inference_objective_request_total": cleanMetric(metricReqTotal(modelSQLLora, modelSQLLoraTarget)),
			},
		},
		{
			name:     "select lora despite higher kv cache (affinity)",
			requests: integration.ReqLLM(logger, "test3", modelSQLLora, modelSQLLoraTarget),
			pods: []podState{
				P(0, 10, 0.2, "foo", "bar"),
				P(1, 10, 0.4, "foo", modelSQLLoraTarget), // Winner (Affinity overrides KV)
				P(2, 10, 0.3, "foo"),
			},
			wantResponses: ExpectRouteTo("192.168.1.2:8000", modelSQLLoraTarget, "test3"),
			wantMetrics: map[string]string{
				"inference_objective_request_total": cleanMetric(metricReqTotal(modelSQLLora, modelSQLLoraTarget)),
			},
		},
		{
			name:     "do not shed requests by default",
			requests: integration.ReqLLM(logger, "test4", modelSQLLora, modelSQLLoraTarget),
			pods: []podState{
				P(0, 6, 0.2, "foo", "bar", modelSQLLoraTarget), // Winner (Lowest saturated)
				P(1, 0, 0.85, "foo"),
				P(2, 10, 0.9, "foo"),
			},
			wantResponses: ExpectRouteTo("192.168.1.1:8000", modelSQLLoraTarget, "test4"),
			wantMetrics: map[string]string{
				"inference_objective_request_total": cleanMetric(metricReqTotal(modelSQLLora, modelSQLLoraTarget)),
			},
		},

		// --- Error Handling & Edge Cases ----
		{
			name: "invalid json body",
			requests: integration.ReqRaw(
				map[string]string{"hi": "mom"},
				"no healthy upstream",
			),
			pods: []podState{
				P(0, 0, 0.2, "foo", "bar"),
			},
			wantResponses: ExpectReject(
				envoyTypePb.StatusCode_BadRequest,
				"inference gateway: BadRequest - Error unmarshaling request body",
			),
		},
		{
			name: "split body across chunks",
			requests: integration.ReqRaw(
				map[string]string{
					"hi":                         "mom",
					metadata.ObjectiveKey:        modelSheddable,
					metadata.ModelNameRewriteKey: modelSheddableTarget,
					requtil.RequestIdHeaderKey:   "test-request-id",
				},
				`{"max_tokens":100,"model":"sql-lo`,
				`ra-sheddable","prompt":"test6","temperature":0}`,
			),
			pods: []podState{
				P(0, 4, 0.2, "foo", "bar", modelSheddableTarget),
				P(1, 4, 0.85, "foo", modelSheddableTarget),
			},
			wantResponses: ExpectRouteTo("192.168.1.1:8000", modelSheddableTarget, "test6"),
			wantMetrics: map[string]string{
				"inference_objective_request_total": cleanMetric(metricReqTotal(modelSheddable, modelSheddableTarget)),
			},
		},
		{
			name:     "no backend pods available",
			requests: integration.ReqHeaderOnly(map[string]string{"content-type": "application/json"}),
			pods:     nil,
			wantResponses: ExpectReject(envoyTypePb.StatusCode_InternalServerError,
				"inference gateway: Internal - no pods available in datastore"),
		},
		{
			name: "request missing model field",
			requests: integration.ReqRaw(
				map[string]string{"content-type": "application/json"},
				`{"hello":"world"}`,
			),
			wantResponses: ExpectReject(envoyTypePb.StatusCode_BadRequest,
				"inference gateway: BadRequest - model not found in request body"),
		},

		// --- Subsetting & Metadata ---
		{
			name: "subsetting: select best from subset",
			// Only pods in the subset list are eligible.
			requests: ReqSubset("test2", modelSQLLora, modelSQLLoraTarget,
				"192.168.1.1:8000", "192.168.1.2:8000", "192.168.1.3:8000"),
			pods: []podState{
				P(0, 0, 0.2, "foo"),
				P(1, 0, 0.1, "foo", modelSQLLoraTarget), // Winner (Low Queue + Matches Subset)
				P(2, 10, 0.2, "foo"),
			},
			wantResponses: ExpectRouteTo("192.168.1.2:8000", modelSQLLoraTarget, "test2"),
		},
		{
			name:     "subsetting: partial match",
			requests: ReqSubset("test2", modelSQLLora, modelSQLLoraTarget, "192.168.1.3:8000"),
			pods: []podState{
				P(0, 0, 0.2, "foo"),
				P(1, 0, 0.1, "foo", modelSQLLoraTarget),
				P(2, 10, 0.2, "foo"), // Winner (Matches Subset, despite load)
			},
			wantResponses: ExpectRouteTo("192.168.1.3:8000", modelSQLLoraTarget, "test2"),
		},
		{
			name:     "subsetting: no pods match",
			requests: ReqSubset("test2", modelSQLLora, modelSQLLoraTarget, "192.168.1.99:8000"),
			pods: []podState{
				P(0, 0, 0.2, "foo"),
				P(1, 0, 0.1, "foo", modelSQLLoraTarget),
			},
			wantResponses: ExpectReject(envoyTypePb.StatusCode_ServiceUnavailable,
				"inference gateway: ServiceUnavailable - failed to find candidate pods for serving the request"),
		},

		// --- Request Modification (Passthrough & Rewrite) ---
		{
			name: "passthrough: model not in objectives",
			requests: integration.ReqRaw(
				map[string]string{
					"hi":                         "mom",
					metadata.ObjectiveKey:        modelDirect,
					metadata.ModelNameRewriteKey: modelDirect,
					requtil.RequestIdHeaderKey:   "test-request-id",
				},
				`{"max_tokens":100,"model":"direct-`,
				`model","prompt":"test6","temperature":0}`,
			),
			pods: []podState{
				P(0, 4, 0.2, "foo", "bar", modelSheddableTarget),
			},
			wantResponses: ExpectRouteTo("192.168.1.1:8000", modelDirect, "test6"),
			wantMetrics: map[string]string{
				"inference_objective_request_total": cleanMetric(metricReqTotal(modelDirect, modelDirect)),
			},
		},
		{
			name:     "rewrite request model",
			requests: integration.ReqLLM(logger, "test-rewrite", modelToBeWritten, modelToBeWritten),
			pods: []podState{
				P(0, 0, 0.1, "foo", modelAfterRewrite),
			},
			wantResponses: ExpectRouteTo("192.168.1.1:8000", modelAfterRewrite, "test-rewrite"),
			wantMetrics: map[string]string{
				"inference_objective_request_total": cleanMetric(metricReqTotal(modelToBeWritten, modelAfterRewrite)),
			},
		},
		{
			name: "protocol: simple GET (header only)",
			requests: integration.ReqHeaderOnly(map[string]string{
				"content-type": "text/event-stream",
				"status":       "200",
			}),
			pods:          []podState{P(0, 0, 0, "foo")},
			wantResponses: nil,
		},

		// --- Response Processing (Buffering & Streaming) ---
		{
			name: "response buffering: multi-chunk JSON",
			requests: ReqResponseOnly(
				map[string]string{"content-type": "application/json"},
				`{"max_tokens":100,"model":"sql-lo`,
				`ra-sheddable","prompt":"test6","temperature":0}`,
			),
			pods: []podState{P(0, 4, 0.2, modelSheddableTarget)},
			wantResponses: ExpectBufferResp(
				fmt.Sprintf(`{"max_tokens":100,"model":%q,"prompt":"test6","temperature":0}`, modelSheddable),
				"application/json"),
		},
		{
			name: "response buffering: invalid JSON",
			requests: ReqResponseOnly(
				map[string]string{"content-type": "application/json"},
				"no healthy upstream",
			),
			pods:          []podState{P(0, 4, 0.2, modelSheddableTarget)},
			wantResponses: ExpectBufferResp("no healthy upstream", "application/json"),
		},
		{
			name: "response buffering: empty EOS chunk (JSON)",
			requests: ReqResponseOnly(
				map[string]string{"content-type": "application/json"},
				`{"max_tokens":100,"model":"sql-lora-sheddable","prompt":"test6","temperature":0}`,
				"",
			),
			pods: []podState{P(0, 4, 0.2, modelSheddableTarget)},
			wantResponses: ExpectBufferResp(
				fmt.Sprintf(`{"max_tokens":100,"model":%q,"prompt":"test6","temperature":0}`, modelSheddable),
				"application/json"),
		},
		{
			name: "response streaming: SSE token counting",
			requests: ReqResponseOnly(
				map[string]string{"content-type": "text/event-stream", "status": "200"},
				// Chunk 1: Simulate a standard data chunk.
				`data: {}`,
				// Chunk 2: Usage data + DONE signal.
				`data: {"usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}`+"\n"+`data: [DONE]`,
				"", // EndOfStream
			),
			pods:         []podState{P(0, 4, 0.2, modelSheddableTarget)},
			waitForModel: modelSheddable,
			wantResponses: ExpectStreamResp(
				`data: {}`,
				`data: {"usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}`+"\n"+`data: [DONE]`,
				"",
			),
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
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Resolve Model to Sync.
			modelToSync := tc.waitForModel
			if modelToSync == "" {
				modelToSync = modelMyModel
			}

			h := NewTestHarness(t, context.Background()).
				WithBaseResources().
				WithPods(tc.pods).
				WaitForSync(len(tc.pods), modelToSync)

			// Wait for metrics to settle to avoid race conditions where Datastore has pods but Scheduler/Metrics collector
			// hasn't processed them yet (causing random scheduling or missing metrics).
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

// loadBaseResources parses the YAML manifest once at startup.
func loadBaseResources() []*unstructured.Unstructured {
	path := filepath.Join("..", "..", "testdata", "inferencepool-with-model-hermetic.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		panic(fmt.Sprintf("failed to read manifest %s: %v", path, err))
	}

	var objs []*unstructured.Unstructured
	decoder := yaml.NewYAMLOrJSONDecoder(strings.NewReader(string(data)), 4096)
	for {
		u := &unstructured.Unstructured{}
		if err := decoder.Decode(u); err != nil {
			if err.Error() == "EOF" {
				break
			}
			panic(fmt.Sprintf("failed to decode YAML: %v", err))
		}
		objs = append(objs, u)
	}
	return objs
}

// cleanMetric removes indentation from multiline metric strings and ensures a trailing newline exists, which is
// required by the Prometheus text parser.
func cleanMetric(s string) string {
	lines := strings.Split(s, "\n")
	var cleaned []string
	for _, l := range lines {
		trimmed := strings.TrimSpace(l)
		if trimmed != "" {
			cleaned = append(cleaned, trimmed)
		}
	}
	return strings.Join(cleaned, "\n") + "\n"
}
