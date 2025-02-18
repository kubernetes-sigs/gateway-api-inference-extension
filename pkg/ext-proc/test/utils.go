package test

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/handlers"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/scheduling"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
	utiltesting "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/testing"
)

func StartExtProc(
	ctx context.Context,
	port int,
	refreshPodsInterval, refreshMetricsInterval, refreshPrometheusMetricsInterval time.Duration,
	pods []*backend.PodMetrics,
	models map[string]*v1alpha1.InferenceModel,
) *grpc.Server {
	logger := log.FromContext(ctx)
	pms := make(map[types.NamespacedName]*backend.PodMetrics)
	for _, pod := range pods {
		pms[pod.NamespacedName] = pod
	}
	pmc := &backend.FakePodMetricsClient{Res: pms}
	datastore := backend.NewDatastore()
	for _, m := range models {
		datastore.ModelSet(m)
	}
	for _, pm := range pods {
		pod := utiltesting.MakePod(pm.NamespacedName.Name, pm.NamespacedName.Namespace).
			ReadyCondition().
			IP(pm.Address).
			Obj()
		datastore.PodUpdateOrAddIfNotExist(&pod)
		datastore.PodUpdateMetricsIfExist(pm.NamespacedName, &pm.Metrics)
	}
	pp := backend.NewProvider(pmc, datastore)
	if err := pp.Init(ctx, refreshMetricsInterval, refreshPrometheusMetricsInterval); err != nil {
		logutil.Fatal(logger, err, "Failed to initialize")
	}
	return startExtProc(logger, port, datastore)
}

// startExtProc starts an extProc server with fake pods.
func startExtProc(logger logr.Logger, port int, datastore backend.Datastore) *grpc.Server {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		logutil.Fatal(logger, err, "Failed to listen", "port", port)
	}

	s := grpc.NewServer()

	extProcPb.RegisterExternalProcessorServer(s, handlers.NewServer(scheduling.NewScheduler(datastore), "target-pod", datastore))

	logger.Info("gRPC server starting", "port", port)
	reflection.Register(s)
	go func() {
		err := s.Serve(lis)
		if err != nil {
			logutil.Fatal(logger, err, "Ext-proc failed with the err")
		}
	}()
	return s
}

func GenerateRequest(logger logr.Logger, prompt, model string) *extProcPb.ProcessingRequest {
	j := map[string]interface{}{
		"model":       model,
		"prompt":      prompt,
		"max_tokens":  100,
		"temperature": 0,
	}

	llmReq, err := json.Marshal(j)
	if err != nil {
		logutil.Fatal(logger, err, "Failed to unmarshal LLM request")
	}
	req := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{Body: llmReq},
		},
	}
	return req
}

func FakePodMetrics(index int, metrics backend.Metrics) *backend.PodMetrics {
	address := fmt.Sprintf("address-%v", index)
	pod := backend.PodMetrics{
		Pod: backend.Pod{
			NamespacedName: types.NamespacedName{Name: fmt.Sprintf("pod-%v", index)},
			Address:        address,
		},
		Metrics: metrics,
	}
	return &pod
}
