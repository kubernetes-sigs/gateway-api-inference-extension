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
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	utiltesting "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

func StartExtProc(
	ctx context.Context,
	port int,
	refreshPodsInterval, refreshMetricsInterval, refreshPrometheusMetricsInterval time.Duration,
	pods []*datastore.PodMetrics,
	models map[string]*v1alpha2.InferenceModel,
) *grpc.Server {
	logger := log.FromContext(ctx)
	pms := make(map[types.NamespacedName]*datastore.PodMetrics)
	for _, pod := range pods {
		pms[pod.NamespacedName] = pod
	}
	pmc := &backend.FakePodMetricsClient{Res: pms}
	datastore := datastore.NewDatastore()
	for _, m := range models {
		datastore.ModelSetIfOlder(m)
	}
	for _, pm := range pods {
		pod := utiltesting.MakePod(pm.NamespacedName.Name).
			Namespace(pm.NamespacedName.Namespace).
			ReadyCondition().
			IP(pm.Address).
			ObjRef()
		datastore.PodUpdateOrAddIfNotExist(pod)
		datastore.PodUpdateMetricsIfExist(pm.NamespacedName, &pm.Metrics)
	}
	pp := backend.NewProvider(pmc, datastore)
	if err := pp.Init(ctx, refreshMetricsInterval, refreshPrometheusMetricsInterval); err != nil {
		logutil.Fatal(logger, err, "Failed to initialize")
	}
	return startExtProc(logger, port, datastore)
}

// startExtProc starts an extProc server with fake pods.
func startExtProc(logger logr.Logger, port int, datastore datastore.Datastore) *grpc.Server {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		logutil.Fatal(logger, err, "Failed to listen", "port", port)
	}

	s := grpc.NewServer()

	extProcPb.RegisterExternalProcessorServer(s, handlers.NewServer(scheduling.NewScheduler(datastore), "", "target-pod", datastore))

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

func FakePodMetrics(index int, metrics datastore.Metrics) *datastore.PodMetrics {
	address := fmt.Sprintf("192.168.1.%d", index+1)
	pod := datastore.PodMetrics{
		Pod: datastore.Pod{
			NamespacedName: types.NamespacedName{Name: fmt.Sprintf("pod-%v", index), Namespace: "default"},
			Address:        address,
		},
		Metrics: metrics,
	}
	return &pod
}
