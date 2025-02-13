package test

import (
	"encoding/json"
	"fmt"
	"net"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
	klog "k8s.io/klog/v2"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/handlers"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/scheduling"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

func StartExtProc(port int, refreshPodsInterval, refreshMetricsInterval, refreshPrometheusMetricsInterval time.Duration, pods []*backend.PodMetrics, models map[string]*v1alpha1.InferenceModel) *grpc.Server {
	ps := make(backend.PodSet)
	pms := make(map[backend.Pod]*backend.PodMetrics)
	for _, pod := range pods {
		ps[pod.Pod] = true
		pms[pod.Pod] = pod
	}
	pmc := &backend.FakePodMetricsClient{Res: pms}
	pp := backend.NewProvider(pmc, backend.NewK8sDataStore(backend.WithPods(pods)))
	if err := pp.Init(refreshPodsInterval, refreshMetricsInterval, refreshPrometheusMetricsInterval); err != nil {
		logutil.Fatal(err, "Failed to initialize")
	}
	return startExtProc(port, pp, models)
}

// startExtProc starts an extProc server with fake pods.
func startExtProc(port int, pp *backend.Provider, models map[string]*v1alpha1.InferenceModel) *grpc.Server {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		logutil.Fatal(err, "Failed to listen", "port", port)
	}

	s := grpc.NewServer()

	extProcPb.RegisterExternalProcessorServer(s, handlers.NewServer(pp, scheduling.NewScheduler(pp), "target-pod", &backend.FakeDataStore{Res: models}))

	klog.InfoS("gRPC server starting", "port", port)
	reflection.Register(s)
	go func() {
		err := s.Serve(lis)
		if err != nil {
			logutil.Fatal(err, "Ext-proc failed with the err")
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
		logutil.Fatal(err, "Failed to unmarshal LLM request")
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
