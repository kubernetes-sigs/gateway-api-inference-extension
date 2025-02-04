package test

import (
	"encoding/json"
	"fmt"
	"net"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/handlers"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/scheduling"
	klog "k8s.io/klog/v2"
)

func StartExtProc(port int, refreshPodsInterval, refreshMetricsInterval time.Duration, pods []*backend.PodMetrics, models map[string]*v1alpha1.InferenceModel) *grpc.Server {
	ps := make(backend.PodSet)
	pms := make(map[string]*backend.PodMetrics)
	for _, pod := range pods {
		ps[pod.Pod] = true
		pms[pod.Pod.Name] = pod
	}
	pmc := &backend.FakePodMetricsClient{Res: pms}
	pp := backend.NewProvider(pmc, backend.NewK8sDataStore())
	if err := pp.Init(refreshPodsInterval, refreshMetricsInterval); err != nil {
		klog.Fatalf("failed to initialize: %v", err)
	}
	return startExtProc(port, pp, models)
}

// startExtProc starts an extProc server with fake pods.
func startExtProc(port int, pp *backend.Provider, models map[string]*v1alpha1.InferenceModel) *grpc.Server {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		klog.Fatalf("failed to listen: %v", err)
	}

	s := grpc.NewServer()

	extProcPb.RegisterExternalProcessorServer(s, handlers.NewServer(pp, scheduling.NewScheduler(pp), "target-pod", &backend.FakeDataStore{Res: models}))

	klog.Infof("Starting gRPC server on port :%v", port)
	reflection.Register(s)
	go func() {
		err := s.Serve(lis)
		if err != nil {
			klog.Fatalf("Ext-proc failed with the err: %v", err)
		}
	}()
	return s
}

func GenerateRequest(model string) *extProcPb.ProcessingRequest {
	j := map[string]interface{}{
		"model":       model,
		"prompt":      "hello",
		"max_tokens":  100,
		"temperature": 0,
	}

	llmReq, err := json.Marshal(j)
	if err != nil {
		klog.Fatal(err)
	}
	req := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{Body: llmReq},
		},
	}
	return req
}

func FakePod(index int) backend.Pod {
	address := fmt.Sprintf("address-%v", index)
	pod := backend.Pod{
		Name:    fmt.Sprintf("pod-%v", index),
		Address: address,
	}
	return pod
}
