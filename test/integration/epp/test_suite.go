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
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/structpb"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/utils/ptr"
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
	utiltesting "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
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
	utilruntime.Must(v1alpha2.AddToScheme(scheme))

	k8sClient, err = k8sclient.New(cfg, k8sclient.Options{Scheme: scheme})
	if err != nil {
		logutil.Fatal(logger, err, "Failed to start k8s Client")
	} else if k8sClient == nil {
		logutil.Fatal(logger, nil, "No error, but returned kubernetes client is nil", "config", cfg)
	}

	// Init runtime.
	ctrl.SetLogger(logger)

	mgr, err := server.NewManagerWithOptions(cfg, managerTestOptions("default", "vllm-llama2-7b-pool"))
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

	// Start the controller manager in a go routine, not blocking
	go func() {
		if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
			logutil.Fatal(logger, err, "Failed to start manager")
		}
	}()

	logger.Info("Setting up hermetic ExtProc server")

	ns := "default"
	pool := utiltesting.MakeInferencePool("vllm-llama2-7b-pool").
		Namespace(ns).
		TargetPortNumber(8000).
		Selector(map[string]string{"app": "vllm-llama2-7b-pool"}).
		ExtensionRef("epp").
		ObjRef()
	if err := k8sClient.Create(context.Background(), pool); err != nil {
		logutil.Fatal(logger, err, "Unable to create inferencePool", "pool", pool.Name)
	}

	models := []*v1alpha2.InferenceModel{
		utiltesting.MakeInferenceModel("sample").
			Namespace(ns).
			ModelName("sql-lora").
			Criticality(v1alpha2.Critical).
			PoolName(pool.Name).
			TargetModel("sql-lora-1fdg2").
			ObjRef(),
		utiltesting.MakeInferenceModel("sheddable").
			Namespace(ns).
			ModelName("sql-lora-sheddable").
			Criticality(v1alpha2.Sheddable).
			PoolName(pool.Name).
			TargetModel("sql-lora-1fdg3").
			ObjRef(),
		utiltesting.MakeInferenceModel("generic").
			Namespace(ns).
			ModelName("my-model").
			Criticality(v1alpha2.Critical).
			PoolName(pool.Name).
			TargetModel("my-model-12345").
			ObjRef(),
		utiltesting.MakeInferenceModel("direct-model").
			Namespace(ns).
			ModelName("direct-model").
			Criticality(v1alpha2.Critical).
			PoolName(pool.Name).
			ObjRef(),
	}
	for i := range models {
		logger.Info("Creating inference model", "model", models[i])
		if err := k8sClient.Create(context.Background(), models[i]); err != nil {
			logutil.Fatal(logger, err, "Unable to create inferenceModel", "modelName", models[i].Name)
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

func streamedRequest(t *testing.T, client extProcPb.ExternalProcessor_ProcessClient, requests []*extProcPb.ProcessingRequest, expectedResponses int) ([]*extProcPb.ProcessingResponse, error) {
	for _, req := range requests {
		t.Logf("Sending request: %v", req)
		if err := client.Send(req); err != nil {
			t.Logf("Failed to send request %+v: %v", req, err)
			return nil, err
		}
		// Brief pause for the goroutines to execute sequentially and populate the internal pipe channels sequentially
		// without the pause there can be a race condition where a goroutine from a subsequent request is able to populate
		// the pipe writer channel before a previous chunk. This is simply due to everything running in memory, this would
		// not happen in a real world environment with non-zero latency.
		time.Sleep(1 * time.Millisecond)
	}
	responses := []*extProcPb.ProcessingResponse{}

	// Make an incredible simple timeout func in the case where
	// there is less than the expected amount of responses; bail and fail.
	var simpleTimeout bool
	go func() {
		time.Sleep(10 * time.Second)
		simpleTimeout = true
	}()

	for range expectedResponses {
		if simpleTimeout {
			break
		}
		res, err := client.Recv()
		if err != nil && err != io.EOF {
			t.Logf("Failed to receive: %v", err)
			return nil, err
		}
		t.Logf("Received request %+v", res)
		responses = append(responses, res)
	}
	return responses, nil
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
			SkipNameValidation: ptr.To(true),
		},
	}
}
