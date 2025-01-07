package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/bojand/ghz/printer"
	"github.com/bojand/ghz/runner"
	"github.com/jhump/protoreflect/desc"
	"google.golang.org/protobuf/proto"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/test"
	klog "k8s.io/klog/v2"
)

var (
	svrAddr       = flag.String("server_address", "localhost:9002", "Address of the ext proc server")
	totalRequests = flag.Int("total_requests", 100000, "number of requests to be sent for load test")
	// Flags when running a local ext proc server.
	numFakePods            = flag.Int("num_fake_pods", 200, "number of fake pods when running a local ext proc server")
	numModelsPerPod        = flag.Int("num_models_per_pod", 5, "number of fake models per pod when running a local ext proc server")
	localServer            = flag.Bool("local_server", true, "whether to start a local ext proc server")
	refreshPodsInterval    = flag.Duration("refreshPodsInterval", 10*time.Second, "interval to refresh pods")
	refreshMetricsInterval = flag.Duration("refreshMetricsInterval", 50*time.Millisecond, "interval to refresh metrics")
)

const (
	port = 9002
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	if *localServer {
		test.StartExtProc(port, *refreshPodsInterval, *refreshMetricsInterval, fakePods(), fakeModels())
		time.Sleep(time.Second) // wait until server is up
		klog.Info("Server started")
	}

	report, err := runner.Run(
		"envoy.service.ext_proc.v3.ExternalProcessor.Process",
		*svrAddr,
		runner.WithInsecure(true),
		runner.WithBinaryDataFunc(generateRequest),
		runner.WithTotalRequests(uint(*totalRequests)),
	)
	if err != nil {
		klog.Fatal(err)
	}

	printer := printer.ReportPrinter{
		Out:    os.Stdout,
		Report: report,
	}

	printer.Print("summary")
}

func generateRequest(mtd *desc.MethodDescriptor, callData *runner.CallData) []byte {
	numModels := *numFakePods * (*numModelsPerPod)
	req := test.GenerateRequest(modelName(int(callData.RequestNumber) % numModels))
	data, err := proto.Marshal(req)
	if err != nil {
		klog.Fatal("marshaling error: ", err)
	}
	return data
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

func fakePods() []*backend.PodMetrics {
	pms := make([]*backend.PodMetrics, 0, *numFakePods)
	for i := 0; i < *numFakePods; i++ {
		metrics := fakeMetrics(i)
		pod := test.FakePod(i)
		pms = append(pms, &backend.PodMetrics{Pod: pod, Metrics: metrics})
	}

	return pms
}

// fakeMetrics adds numModelsPerPod number of adapters to the pod metrics.
func fakeMetrics(podNumber int) backend.Metrics {
	metrics := backend.Metrics{
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
