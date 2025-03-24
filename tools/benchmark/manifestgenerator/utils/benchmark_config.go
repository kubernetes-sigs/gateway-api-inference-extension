package utils

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
	klog "k8s.io/klog/v2"

	benchmarkpb "benchmark-catalog/proto"
)

var (
	// The benchmark rates of a single accelerator can handle for llama-2-7b model.
	acceleratorQPSLlama2_7b = map[string][]float32{
		// Latency start to grow at 5, through peaks at 6
		"nvidia-l4": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		// Latency starts to grow around 32, throughput peaks at 34
		"nvidia-tesla-a100": {2, 4, 6, 8, 10, 12, 16, 24, 28, 30, 32, 34, 36, 40},
		"nvidia-h100-80gb":  {10, 20, 30, 40, 60, 70, 80, 90, 100},
	}
	acceleratorQPSLlama3_8b = map[string][]float32{
		// Latency start to grow at 5, through peaks at 6
		"nvidia-l4": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		// Latency starts to grow around 32, throughput peaks at 34
		"nvidia-tesla-a100": {2, 4, 6, 8, 10, 12, 16, 24, 28, 30, 32, 34, 36, 40},
		// Latency starts to grow at 100, throughput peaks at 110
		"nvidia-h100-80gb": {10, 20, 30, 40, 60, 80, 90, 100, 120, 140, 180},
	}
	acceleratorQPSGemma2_27b = map[string][]float32{
		// Latency starts to grow at 8, throughput peaks at 16
		"nvidia-h100-80gb": {2, 4, 6, 10, 12, 16, 20, 24, 28, 32},
	}
)

func ReadBenchmarks(file string) ([]*benchmarkpb.Benchmark, error) {
	klog.V(1).Infof("Reading benchmark from file %s", file)
	res := []*benchmarkpb.Benchmark{}
	// Read the config file
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}

	bs := &benchmarkpb.Benchmarks{}
	if prototext.Unmarshal(data, bs); err != nil {
		return nil, fmt.Errorf("failed to unmarshall %v: %v", file, err)
	}
	klog.V(1).Infof("Read %v raw benchmarks", len(bs.GetBenchmarks()))
	klog.V(2).Infof("Raw benchmarks: %+v", bs.GetBenchmarks())
	// Build a map of benchmark names to benchmark
	raw := make(map[string]*benchmarkpb.Benchmark, len(bs.Benchmarks))
	for _, benchmark := range bs.Benchmarks {
		if _, ok := raw[benchmark.Name]; ok {
			return nil, fmt.Errorf("Duplicate benchmark: %v", benchmark.Name)
		}
		raw[benchmark.Name] = benchmark
	}
	processed := make(map[string]*benchmarkpb.Benchmark, len(bs.Benchmarks))

	for _, benchmark := range bs.Benchmarks {
		klog.V(2).Infof("Before processing benchmark %v: %v: %+v", benchmark.Name, len(processed), processed)
		updated, err := processBenchmark(benchmark, raw, processed)
		klog.V(2).Infof("After processing benchmark %v: %v: %+v", benchmark.Name, len(processed), processed)
		if err != nil {
			return nil, err
		}
		res = append(res, updated)
	}

	return res, nil
}

func ReadBenchmark(file string) (*benchmarkpb.Benchmark, error) {
	// Read the config file
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}

	// Parse the config and override the defaults
	b := &benchmarkpb.Benchmark{}
	if err := prototext.Unmarshal(data, b); err != nil {
		return nil, fmt.Errorf("failed to unmarshall %v: %v", file, err)
	}

	if err := applyDefaults(b); err != nil {
		return nil, fmt.Errorf("failed to apply defaults: %v", err)
	}

	if err := validateBenchmark(b); err != nil {
		return nil, fmt.Errorf("failed to validate %v", file)
	}

	return b, nil
}

func processBenchmark(b *benchmarkpb.Benchmark, raw, processed map[string]*benchmarkpb.Benchmark) (*benchmarkpb.Benchmark, error) {
	klog.V(2).Infof("Processing benchmark %v: %+v", b.GetName(), b)
	updated := proto.Clone(b).(*benchmarkpb.Benchmark)
	if b.GetBaseBenchmarkName() != "" {
		klog.V(2).Infof("[Benchmark=%v, base=%v]", b.GetName(), b.GetBaseBenchmarkName())
		rawBase, ok := raw[b.GetBaseBenchmarkName()]
		if !ok {
			return nil, fmt.Errorf("couldn't find base benchmark %v", b.GetBaseBenchmarkName())
		}
		if _, ok := processed[b.GetBaseBenchmarkName()]; !ok {
			klog.V(2).Infof("[Benchmark=%v, base=%v], base hasn't been processed", b.GetName(), b.GetBaseBenchmarkName())
			processedBase, err := processBenchmark(rawBase, raw, processed)
			if err != nil {
				return nil, err
			}
			klog.V(2).Infof("Updating processed benchmark map: %v", processedBase.GetName())
			processed[processedBase.GetName()] = processedBase
		} else {
			klog.V(2).Infof("[Benchmark=%v, base=%v] Base has already been processed", b.GetName(), b.GetBaseBenchmarkName())
		}
		processedBase := processed[b.GetBaseBenchmarkName()]
		updated = applyBaseBenchmark(b, processedBase)
	}

	if err := applyDefaults(updated); err != nil {
		return nil, fmt.Errorf("failed to apply defaults: %v", err)
	}

	if err := validateBenchmark(updated); err != nil {
		return nil, fmt.Errorf("failed to validate %v", updated.GetName())
	}
	klog.V(2).Infof("Updated benchmark %v: %+v", b.GetName(), b)
	processed[updated.GetName()] = updated
	return updated, nil
}

func SaveBenchmark(b *benchmarkpb.Benchmark, file string, override bool) error {
	out, err := prototext.Marshal(b)
	if err != nil {
		return fmt.Errorf("error marshalling to pbtxt: %v", err)
	}
	klog.V(1).Infof("Saving proto file %q", file)

	if err := SaveFile(file+".pbtxt", out, override); err != nil {
		return fmt.Errorf("error saving file %q: %v", file, err)
	}

	yaml, err := protobufToYAML(&benchmarkpb.Helm{Global: b})
	if err != nil {
		return fmt.Errorf("error converting proto to yaml: %v", err)
	}
	return SaveFile(file+".yaml", yaml, override)
}

func applyBaseBenchmark(benchmark, base *benchmarkpb.Benchmark) *benchmarkpb.Benchmark {
	klog.V(2).Infof("Applying base benchmark %v to %v", base.GetName(), benchmark.GetName())
	updated := proto.Clone(base).(*benchmarkpb.Benchmark)
	// Hack: Do not inherit request rates from base. Usually request rates needs to be updated.
	updated.GetConfig().GetBenchmarkTool().GetLpg().RequestRates = ""
	proto.Merge(updated, benchmark)
	return updated
}

func validateBenchmark(b *benchmarkpb.Benchmark) error {
	return nil
}

func applyDefaults(b *benchmarkpb.Benchmark) error {
	b.GetConfig().GetLoadBalancer().GatewayEnabled = b.GetConfig().GetLoadBalancer().GetGateway() != nil
	b.GetConfig().GetLoadBalancer().K8SServiceEnabled = b.GetConfig().GetLoadBalancer().GetK8SService() != nil
	b.GetConfig().GetLoadBalancer().GatewayEnvoyEnabled = b.GetConfig().GetLoadBalancer().GetGateway().GetEnvoy() != nil
	b.GetConfig().GetLoadBalancer().GatewayGkeGatewayEnabled = b.GetConfig().GetLoadBalancer().GetGateway().GetGkeGateway() != nil
	b.GetConfig().GetLoadBalancer().GatewayEnvoyEppEnabled = b.GetConfig().GetLoadBalancer().GetGateway().GetEnvoy().GetEpp() != nil
	b.GetConfig().GetLoadBalancer().GatewayEnvoyLbPolicyEnabled = b.GetConfig().GetLoadBalancer().GetGateway().GetEnvoy().GetLbPolicy() != ""

	applyVLLMDefaults(b.GetConfig().GetModelServer().GetVllm())
	if err := applyBenchmarkToolDefaults(b); err != nil {
		return err
	}
	return nil
}

func applyBenchmarkToolDefaults(b *benchmarkpb.Benchmark) error {
	lpg := b.GetConfig().GetBenchmarkTool().GetLpg()
	if lpg == nil {
		return nil
	}

	if err := applyRequestRatesDefaults(b); err != nil {
		return err
	}

	return nil
}

func applyRequestRatesDefaults(b *benchmarkpb.Benchmark) error {
	lpg := b.GetConfig().GetBenchmarkTool().GetLpg()
	if lpg.GetRequestRates() != "" {
		klog.V(2).Infof("Request rates specified, skipping defaults : %v", lpg.GetRequestRates())
		return nil
	}

	klog.V(2).Infof("Applying default request rates to %v", b.GetName())
	// Apply default request rates
	accelerator := b.GetConfig().GetModelServer().GetAccelerator()
	qps, err := defaultRates(accelerator, b.GetConfig().GetModelServer().GetVllm().GetModel())
	if err != nil {
		return err
	}
	tp, err := strconv.Atoi(b.GetConfig().GetModelServer().GetVllm().GetTensorParallelism())
	if err != nil {
		return fmt.Errorf("failed to convert tensor TensorParallelism to int")
	}
	numAccelerators := b.GetConfig().GetModelServer().GetReplicas() * int32(tp)
	numModels := len(strings.Split(lpg.GetModels(), ","))
	klog.V(2).Infof("[Benchmark=%v] num models=%v, num accelerators=%v", b.GetName(), numModels, numAccelerators)

	rates := make([]string, 0, len(qps))
	for _, baseRate := range qps {
		newRate := baseRate * float32(numAccelerators) / float32(numModels)
		rates = append(rates, fmt.Sprintf("%.1f", newRate))
	}
	lpg.RequestRates = strings.Join(rates, ",")
	klog.V(2).Infof("[Benchmark=%v]Set request rates to %v", b.GetName(), lpg.RequestRates)
	return nil
}

func defaultRates(accelerator, model string) ([]float32, error) {
	switch model {
	case "meta-llama/Llama-2-7b-hf":
		qps, ok := acceleratorQPSLlama2_7b[accelerator]
		if !ok {
			return nil, fmt.Errorf("unknown accelerator type: %v", accelerator)
		}
		return qps, nil
	case "meta-llama/Llama-3.1-8B-Instruct":
		qps, ok := acceleratorQPSLlama3_8b[accelerator]
		if !ok {
			return nil, fmt.Errorf("unknown accelerator type: %v", accelerator)
		}
		return qps, nil
	case "google/gemma-2-27b":
		qps, ok := acceleratorQPSGemma2_27b[accelerator]
		if !ok {
			return nil, fmt.Errorf("unknown accelerator type: %v", accelerator)
		}
		return qps, nil
	default:
		return nil, fmt.Errorf("unsupported model: %v", model)
	}
}

func applyVLLMDefaults(v *benchmarkpb.VLLM) {
	if v == nil {
		return
	}
	if v.GetTensorParallelism() == "" {
		v.TensorParallelism = "1"
	}
	if v.GetV1() == "" {
		v.V1 = "0"
	}
}
