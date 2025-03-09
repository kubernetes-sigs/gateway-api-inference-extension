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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	k8syaml "k8s.io/apimachinery/pkg/util/yaml"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/component-base/metrics/legacyregistry"
	metricsutils "k8s.io/component-base/metrics/testutil"
	ctrl "sigs.k8s.io/controller-runtime"
	k8sclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/server"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/server"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	utiltesting "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
	"sigs.k8s.io/yaml"
)

const (
	port        = runserver.DefaultGrpcPort
	metricsPort = 8888
)

var (
	serverRunner *runserver.ExtProcServerRunner
	k8sClient    k8sclient.Client
	testEnv      *envtest.Environment
	scheme       = runtime.NewScheme()
	logger       = logutil.NewTestLogger().V(logutil.VERBOSE)
)

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
				fakePod(0): &backendmetrics.Metrics{
					WaitingQueueSize:    3,
					KVCacheUsagePercent: 0.2,
				},
				fakePod(1): &backendmetrics.Metrics{
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.1,
				},
				fakePod(2): &backendmetrics.Metrics{
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
				fakePod(0): &backendmetrics.Metrics{
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo": 1,
						"bar": 1,
					},
				},
				fakePod(1): &backendmetrics.Metrics{
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.1,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg2": 1,
					},
				},
				fakePod(2): &backendmetrics.Metrics{
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
				fakePod(0): &backendmetrics.Metrics{
					WaitingQueueSize:    10,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo": 1,
						"bar": 1,
					},
				},
				fakePod(1): &backendmetrics.Metrics{
					WaitingQueueSize:    200,
					KVCacheUsagePercent: 0.1,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg2": 1,
					},
				},
				fakePod(2): &backendmetrics.Metrics{
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
				fakePod(0): &backendmetrics.Metrics{
					WaitingQueueSize:    6,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo":            1,
						"bar":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(1): &backendmetrics.Metrics{
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.85,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(2): &backendmetrics.Metrics{
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
				fakePod(0): &backendmetrics.Metrics{
					WaitingQueueSize:    4,
					KVCacheUsagePercent: 0.2,
					ActiveModels: map[string]int{
						"foo":            1,
						"bar":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(1): &backendmetrics.Metrics{
					WaitingQueueSize:    0,
					KVCacheUsagePercent: 0.85,
					ActiveModels: map[string]int{
						"foo":            1,
						"sql-lora-1fdg3": 1,
					},
				},
				fakePod(2): &backendmetrics.Metrics{
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

	// Set up global k8sclient and extproc server runner with test environment config
	cleanup := BeforeSuit(t)
	defer cleanup()

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, cleanup := setUpHermeticServer(t, test.pods)
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

func setUpHermeticServer(t *testing.T, podAndMetrics map[backendmetrics.Pod]*backendmetrics.Metrics) (client extProcPb.ExternalProcessor_ProcessClient, cleanup func()) {
	// Reconfigure the TestPodMetricsClient.
	res := map[types.NamespacedName]*backendmetrics.Metrics{}
	for pod, metrics := range podAndMetrics {
		res[pod.NamespacedName] = metrics
	}
	serverRunner.TestPodMetricsClient.SetRes(res)

	serverCtx, stopServer := context.WithCancel(context.Background())

	// TODO: this should be consistent with the inference pool
	podLabels := map[string]string{
		"app": "vllm-llama2-7b-pool",
	}

	for pod := range podAndMetrics {
		pod := utiltesting.MakePod(pod.NamespacedName.Name).
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
			pod := utiltesting.MakePod(pod.NamespacedName.Name).
				Namespace(pod.NamespacedName.Namespace).Complete().ObjRef()

			if err := k8sClient.Delete(context.Background(), pod); err != nil {
				logutil.Fatal(logger, err, "Failed to delete pod", "pod", fakePod)
			}
		}
		// wait a little until the goroutines actually exit
		time.Sleep(5 * time.Second)
	}
}

func fakePod(index int) backendmetrics.Pod {
	return backendmetrics.Pod{
		NamespacedName: types.NamespacedName{Name: fmt.Sprintf("pod-%v", index), Namespace: "default"},
		Address:        fmt.Sprintf("192.168.1.%d", index+1),
	}
}

// Sets up a test environment and returns the runner struct
func BeforeSuit(t *testing.T) func() {
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
	utilruntime.Must(v1alpha2.AddToScheme(scheme))

	k8sClient, err = k8sclient.New(cfg, k8sclient.Options{Scheme: scheme})
	if err != nil {
		logutil.Fatal(logger, err, "Failed to start k8s Client")
	} else if k8sClient == nil {
		logutil.Fatal(logger, nil, "No error, but returned kubernetes client is nil", "config", cfg)
	}

	// Init runtime.
	ctrl.SetLogger(logger)

	mgr, err := server.NewDefaultManager("default", "vllm-llama2-7b-pool", cfg)
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
	serverRunner.PoolName = "vllm-llama2-7b-pool"
	serverRunner.Datastore = datastore.NewDatastore(context.Background(), pmf)
	serverRunner.SecureServing = false

	if err := serverRunner.SetupWithManager(context.Background(), mgr); err != nil {
		logutil.Fatal(logger, err, "Failed to setup server runner")
	}

	// Start the controller manager in go routine, not blocking
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
		inferenceModel := &v1alpha2.InferenceModel{}
		if err = yaml.Unmarshal(doc, inferenceModel); err != nil {
			logutil.Fatal(logger, err, "Can't unmarshal object", "document", doc)
		}
		if inferenceModel.Kind == "InferenceModel" {
			logger.Info("Creating inference model", "model", inferenceModel)
			if err := k8sClient.Create(context.Background(), inferenceModel); err != nil {
				logutil.Fatal(logger, err, "Unable to create inferenceModel", "modelName", inferenceModel.Name)
			}
		}
	}
	for _, doc := range docs {
		inferencePool := &v1alpha2.InferencePool{}
		if err = yaml.Unmarshal(doc, inferencePool); err != nil {
			logutil.Fatal(logger, err, "Can't unmarshal object", "document", doc)
		}
		if inferencePool.Kind == "InferencePool" {
			logger.Info("Creating inference pool", "pool", inferencePool)
			if err := k8sClient.Create(context.Background(), inferencePool); err != nil {
				logutil.Fatal(logger, err, "Unable to create inferencePool", "poolName", inferencePool.Name)
			}
		}
	}

	assert.EventuallyWithT(t, func(t *assert.CollectT) {
		modelExist := serverRunner.Datastore.ModelGet("my-model")
		synced := serverRunner.Datastore.PoolHasSynced() && modelExist != nil
		assert.True(t, synced, "Timeout waiting for the pool and models to sync")
	}, 10*time.Second, 10*time.Millisecond)

	return func() {
		_ = testEnv.Stop()
	}
}

func sendRequest(t *testing.T, client extProcPb.ExternalProcessor_ProcessClient, req *extProcPb.ProcessingRequest) (*extProcPb.ProcessingResponse, error) {
	t.Logf("Sending request: %v", req)
	if err := client.Send(req); err != nil {
		t.Logf("Failed to send request %+v: %v", req, err)
		return nil, err
	}

	res, err := client.Recv()
	if err != nil {
		t.Logf("Failed to receive: %v", err)
		return nil, err
	}
	t.Logf("Received request %+v", res)
	return res, err
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
