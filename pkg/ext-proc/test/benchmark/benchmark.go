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

package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/bojand/ghz/printer"
	"github.com/bojand/ghz/runner"
	"github.com/go-logr/logr"
	"github.com/jhump/protoreflect/desc"
	uberzap "go.uber.org/zap"
	"google.golang.org/protobuf/proto"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/datastore"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/server"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/test"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

var (
	svrAddr       = flag.String("server_address", fmt.Sprintf("localhost:%d", runserver.DefaultGrpcPort), "Address of the ext proc server")
	totalRequests = flag.Int("total_requests", 100000, "number of requests to be sent for load test")
	// Flags when running a local ext proc server.
	numFakePods                      = flag.Int("num_fake_pods", 200, "number of fake pods when running a local ext proc server")
	numModelsPerPod                  = flag.Int("num_models_per_pod", 5, "number of fake models per pod when running a local ext proc server")
	localServer                      = flag.Bool("local_server", true, "whether to start a local ext proc server")
	refreshPodsInterval              = flag.Duration("refreshPodsInterval", 10*time.Second, "interval to refresh pods")
	refreshMetricsInterval           = flag.Duration("refreshMetricsInterval", 50*time.Millisecond, "interval to refresh metrics via polling pods")
	refreshPrometheusMetricsInterval = flag.Duration("refreshPrometheusMetricsInterval", 5*time.Second, "interval to flush prometheus metrics")
)

const (
	port = runserver.DefaultGrpcPort
)

func main() {
	if err := run(); err != nil {
		os.Exit(1)
	}
}

func run() error {
	opts := zap.Options{
		Development: true,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()
	logger := zap.New(zap.UseFlagOptions(&opts), zap.RawZapOpts(uberzap.AddCaller()))
	ctx := log.IntoContext(context.Background(), logger)

	if *localServer {
		test.StartExtProc(ctx, port, *refreshPodsInterval, *refreshMetricsInterval, *refreshPrometheusMetricsInterval, fakePods(), fakeModels())
		time.Sleep(time.Second) // wait until server is up
		logger.Info("Server started")
	}

	report, err := runner.Run(
		"envoy.service.ext_proc.v3.ExternalProcessor.Process",
		*svrAddr,
		runner.WithInsecure(true),
		runner.WithBinaryDataFunc(generateRequestFunc(logger)),
		runner.WithTotalRequests(uint(*totalRequests)),
	)
	if err != nil {
		logger.Error(err, "Runner failed")
		return err
	}

	printer := printer.ReportPrinter{
		Out:    os.Stdout,
		Report: report,
	}

	printer.Print("summary")
	return nil
}

func generateRequestFunc(logger logr.Logger) func(mtd *desc.MethodDescriptor, callData *runner.CallData) []byte {
	return func(mtd *desc.MethodDescriptor, callData *runner.CallData) []byte {
		numModels := *numFakePods * (*numModelsPerPod)
		req := test.GenerateRequest(logger, "hello", modelName(int(callData.RequestNumber)%numModels))
		data, err := proto.Marshal(req)
		if err != nil {
			logutil.Fatal(logger, err, "Failed to marshal request", "request", req)
		}
		return data
	}
}

func fakeModels() map[string]*v1alpha1.InferenceModel {
	models := map[string]*v1alpha1.InferenceModel{}
	for i := range *numFakePods {
		for j := range *numModelsPerPod {
			m := modelName(i*(*numModelsPerPod) + j)
			models[m] = &v1alpha1.InferenceModel{Spec: v1alpha1.InferenceModelSpec{ModelName: m}}
		}
	}

	return models
}

func fakePods() []*datastore.PodMetrics {
	pms := make([]*datastore.PodMetrics, 0, *numFakePods)
	for i := 0; i < *numFakePods; i++ {
		pms = append(pms, test.FakePodMetrics(i, fakeMetrics(i)))
	}

	return pms
}

// fakeMetrics adds numModelsPerPod number of adapters to the pod metrics.
func fakeMetrics(podNumber int) datastore.Metrics {
	metrics := datastore.Metrics{
		ActiveModels: make(map[string]int),
	}
	for i := 0; i < *numModelsPerPod; i++ {
		metrics.ActiveModels[modelName(podNumber*(*numModelsPerPod)+i)] = 0
	}
	return metrics
}

func modelName(i int) string {
	return fmt.Sprintf("adapter-%v", i)
}
