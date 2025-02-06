// Package test contains e2e tests for the ext proc while faking the backend pods.
package integration

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"testing"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/testing/protocmp"
	"google.golang.org/protobuf/types/known/structpb"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	runserver "inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/server"
	extprocutils "inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/test"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	k8syaml "k8s.io/apimachinery/pkg/util/yaml"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	klog "k8s.io/klog/v2"
	k8sclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/yaml"
)

const (
	port = runserver.DefaultGrpcPort
)

var (
	serverRunner *runserver.ExtProcServerRunner
	k8sClient    k8sclient.Client
	testEnv      *envtest.Environment
	scheme       = runtime.NewScheme()
)

func SKIPTestHandleRequestBody(t *testing.T) {
	tests := []struct {
		name        string
		req         *extProcPb.ProcessingRequest
		pods        []*backend.PodMetrics
		models      map[string]*v1alpha1.InferenceModel
		wantHeaders []*configPb.HeaderValueOption
		wantBody    []byte
		wantErr     bool
	}{
		{
			name: "success",
			req:  extprocutils.GenerateRequest("my-model"),
			models: map[string]*v1alpha1.InferenceModel{
				"my-model": {
					Spec: v1alpha1.InferenceModelSpec{
						ModelName: "my-model",
						TargetModels: []v1alpha1.TargetModel{
							{
								Name:   "my-model-v1",
								Weight: pointer(100),
							},
						},
					},
				},
			},
			// pod-1 will be picked because it has relatively low queue size, with the requested
			// model being active, and has low KV cache.
			pods: []*backend.PodMetrics{
				{
					Pod: extprocutils.FakePod(0),
					Metrics: backend.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0.2,
						ActiveModels: map[string]int{
							"foo": 1,
							"bar": 1,
						},
					},
				},
				{
					Pod: extprocutils.FakePod(1),
					Metrics: backend.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0.1,
						ActiveModels: map[string]int{
							"foo":         1,
							"my-model-v1": 1,
						},
					},
				},
				{
					Pod: extprocutils.FakePod(2),
					Metrics: backend.Metrics{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.2,
						ActiveModels: map[string]int{
							"foo": 1,
						},
					},
				},
			},
			wantHeaders: []*configPb.HeaderValueOption{
				{
					Header: &configPb.HeaderValue{
						Key:      runserver.DefaultTargetEndpointKey,
						RawValue: []byte("address-1"),
					},
				},
				{
					Header: &configPb.HeaderValue{
						Key:      "Content-Length",
						RawValue: []byte("73"),
					},
				},
			},
			wantBody: []byte("{\"max_tokens\":100,\"model\":\"my-model-v1\",\"prompt\":\"hello\",\"temperature\":0}"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, cleanup := setUpServer(t, test.pods, test.models)
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
			}
			res, err := sendRequest(t, client, test.req)

			if (err != nil) != test.wantErr {
				t.Fatalf("Unexpected error, got %v, want %v", err, test.wantErr)
			}

			if diff := cmp.Diff(want, res, protocmp.Transform()); diff != "" {
				t.Errorf("Unexpected response, (-want +got): %v", diff)
			}
		})
	}

}

func TestKubeInferenceModelRequest(t *testing.T) {
	tests := []struct {
		name         string
		req          *extProcPb.ProcessingRequest
		wantHeaders  []*configPb.HeaderValueOption
		wantMetadata *structpb.Struct
		wantBody     []byte
		wantErr      bool
	}{
		{
			name: "success",
			req:  extprocutils.GenerateRequest("sql-lora"),
			// pod-1 will be picked because it has relatively low queue size, with the requested
			// model being active, and has low KV cache.
			wantHeaders: []*configPb.HeaderValueOption{
				{
					Header: &configPb.HeaderValue{
						Key:      runserver.DefaultTargetEndpointKey,
						RawValue: []byte("address-1"),
					},
				},
				{
					Header: &configPb.HeaderValue{
						Key:      "Content-Length",
						RawValue: []byte("76"),
					},
				},
			},
			wantMetadata: &structpb.Struct{
				Fields: map[string]*structpb.Value{
					runserver.DefaultTargetEndpointKey: {
						Kind: &structpb.Value_StringValue{
							StringValue: "address-1",
						},
					},
				},
			},
			wantBody: []byte("{\"max_tokens\":100,\"model\":\"sql-lora-1fdg2\",\"prompt\":\"hello\",\"temperature\":0}"),
			wantErr:  false,
		},
	}

	pods := []*backend.PodMetrics{
		{
			Pod: extprocutils.FakePod(0),
			Metrics: backend.Metrics{
				WaitingQueueSize:    0,
				KVCacheUsagePercent: 0.2,
				ActiveModels: map[string]int{
					"foo": 1,
					"bar": 1,
				},
			},
		},
		{
			Pod: extprocutils.FakePod(1),
			Metrics: backend.Metrics{
				WaitingQueueSize:    0,
				KVCacheUsagePercent: 0.1,
				ActiveModels: map[string]int{
					"foo":            1,
					"sql-lora-1fdg2": 1,
				},
			},
		},
		{
			Pod: extprocutils.FakePod(2),
			Metrics: backend.Metrics{
				WaitingQueueSize:    10,
				KVCacheUsagePercent: 0.2,
				ActiveModels: map[string]int{
					"foo": 1,
				},
			},
		},
	}

	// Set up global k8sclient and extproc server runner with test environment config
	BeforeSuit()

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, cleanup := setUpHermeticServer(t, pods)
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

			if err != nil {
				if !test.wantErr {
					t.Errorf("Unexpected error, got: %v, want error: %v", err, test.wantErr)
				}
			} else if diff := cmp.Diff(want, res, protocmp.Transform()); diff != "" {
				t.Errorf("Unexpected response, (-want +got): %v", diff)
			}
		})
	}
}

func setUpServer(t *testing.T, pods []*backend.PodMetrics, models map[string]*v1alpha1.InferenceModel) (client extProcPb.ExternalProcessor_ProcessClient, cleanup func()) {
	t.Logf("Setting up ExtProc server")
	server := extprocutils.StartExtProc(port, time.Second, time.Second, pods, models)

	address := fmt.Sprintf("localhost:%v", port)
	// Create a grpc connection
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to %v: %v", address, err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	client, err = extProcPb.NewExternalProcessorClient(conn).Process(ctx)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	return client, func() {
		cancel()
		conn.Close()
		server.GracefulStop()
	}
}

func setUpHermeticServer(t *testing.T, pods []*backend.PodMetrics) (client extProcPb.ExternalProcessor_ProcessClient, cleanup func()) {
	t.Logf("Setting up hermetic ExtProc server")
	klog.InitFlags(nil)
	flag.Parse()
	// Configure klog verbosity levels to print ext proc logs.
	_ = flag.Lookup("v").Value.Set("3")

	// Unmarshal CRDs from file into structs
	manifestsPath := filepath.Join("..", "testdata", "inferencepool-with-model-hermetic.yaml")
	docs, err := readDocuments(manifestsPath)
	if err != nil {
		log.Fatalf("Can't read object manifests at path %v, %v", manifestsPath, err)
	}

	for _, doc := range docs {
		inferenceModel := &v1alpha1.InferenceModel{}
		if err = yaml.Unmarshal(doc, inferenceModel); err != nil {
			log.Fatalf("Can't unmarshal object: %v", doc)
		}
		if inferenceModel.Kind == "InferenceModel" {
			t.Logf("Creating inference model: %+v", inferenceModel)
			if err := k8sClient.Create(context.Background(), inferenceModel); err != nil {
				log.Fatalf("unable to create inferenceModel %v: %v", inferenceModel.Name, err)
			}
		}
	}
	for _, doc := range docs {
		inferencePool := &v1alpha1.InferencePool{}
		if err = yaml.Unmarshal(doc, inferencePool); err != nil {
			log.Fatalf("Can't unmarshal object: %v", doc)
		}
		if inferencePool.Kind == "InferencePool" {
			t.Logf("Creating inference pool: %+v", inferencePool)
			if err := k8sClient.Create(context.Background(), inferencePool); err != nil {
				log.Fatalf("unable to create inferencePool %v: %v", inferencePool.Name, err)
			}
		}
	}

	ps := make(backend.PodSet)
	pms := make(map[backend.Pod]*backend.PodMetrics)
	for _, pod := range pods {
		ps[pod.Pod] = true
		pms[pod.Pod] = pod
	}
	pmc := &backend.FakePodMetricsClient{Res: pms}

	server := serverRunner.Start(backend.NewK8sDataStore(backend.WithPods(pods)), pmc)
	if err != nil {
		log.Fatalf("Ext-proc failed with the err: %v", err)
	}

	// Wait the reconciler to populate the datastore.
	time.Sleep(10 * time.Second)

	address := fmt.Sprintf("localhost:%v", port)
	// Create a grpc connection
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to %v: %v", address, err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	client, err = extProcPb.NewExternalProcessorClient(conn).Process(ctx)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	return client, func() {
		cancel()
		conn.Close()
		server.GracefulStop()
	}
}

// Sets up a test environment and returns the runner struct
func BeforeSuit() {
	// Set up mock k8s API Client
	testEnv = &envtest.Environment{
		CRDDirectoryPaths:     []string{filepath.Join("..", "..", "config", "crd", "bases")},
		ErrorIfCRDPathMissing: true,
	}
	cfg, err := testEnv.Start()

	if err != nil {
		log.Fatalf("Failed to start test environment, cfg: %v error: %v", cfg, err)
	}

	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(v1alpha1.AddToScheme(scheme))

	k8sClient, err = k8sclient.New(cfg, k8sclient.Options{Scheme: scheme})
	if err != nil {
		log.Fatalf("Failed to start k8s Client: %v", err)
	} else if k8sClient == nil {
		log.Fatalf("No error, but returned kubernetes client is nil, cfg: %v", cfg)
	}

	serverRunner = runserver.NewDefaultExtProcServerRunner()
	// Adjust from defaults
	serverRunner.PoolName = "vllm-llama2-7b-pool"
	serverRunner.Scheme = scheme
	serverRunner.Config = cfg
	serverRunner.Datastore = backend.NewK8sDataStore()

	serverRunner.Setup()

	// Start the controller manager in go routine, not blocking
	go func() {
		serverRunner.StartManager()
	}()
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
func pointer(v int32) *int32 {
	return &v
}
