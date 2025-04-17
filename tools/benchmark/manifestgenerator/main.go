package main

import (
	"bytes"
	"flag"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"

	"benchmark-catalog/manifestgenerator/utils"
	benchmarkpb "benchmark-catalog/proto"

	klog "k8s.io/klog/v2"
)

var (
	catalogDir        = flag.String("catalogDir", "../catalog/", "catalog path containing all kustomize components")
	benchmarkFilePath = flag.String("benchmarkFilePath", "benchmarks.pbtxt", "prototxt file of a SINGLE benchmark to run, ignored when benchmarks is provided")
	benchmarks        = flag.String("benchmarks", "", "prototxt file of the benchmarks to run under the catalogDir")
	outputRootDir     = flag.String("outputRootDir", "../output", "root directory to store output files")
	manifestTypes     = flag.String("manifestTypes", "ModelServer,LoadBalancer", "comma separated list of manifest types of {ModelServer, LoadBalancer, BenchmarkTool}. NOTE: Do not generate BenchmarkTool manifest until the LoadBalancer is deployed.")
	runID             = flag.String("runID", "default", "ID of the run, which can be shared across multiple benchmarks")
	override          = flag.Bool("override", false, "whether to override existing benchmark and manifest files")
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()
	// Print all flag values
	flags := "Flags: "
	flag.VisitAll(func(f *flag.Flag) {
		flags += fmt.Sprintf("%s=%v; ", f.Name, f.Value)
	})
	klog.V(1).Info(flags)

	var bs []*benchmarkpb.Benchmark
	var err error
	// Apply defaulting and validation on input benchmark file, and save the output benchmark file.
	if *benchmarks != "" {
		benchmarksFile := filepath.Join(*catalogDir, *benchmarks)
		klog.Infof("Reading source benchmarks file: %v", benchmarksFile)
		bs, err = utils.ReadBenchmarks(benchmarksFile)
		if err != nil {
			klog.Fatalf("Failed to read benchmarks: %v", err)
		}
		klog.Infof("Read %v benchmarks", len(bs))
	} else { // Read single benchmark file instead
		klog.Infof("Reading single benchmark: %v", *benchmarkFilePath)
		b, err := utils.ReadBenchmark(*benchmarkFilePath)
		if err != nil {
			klog.Fatalf("Failed to read benchmark: %v", err)
		}
		bs = append(bs, b)
	}

	for _, b := range bs {
		processBenchmark(b)
	}
}

func processBenchmark(b *benchmarkpb.Benchmark) {
	gen := &HelmGenerator{}

	klog.V(2).Infof("Processing benchmark %v: %+v", b.GetName(), b)
	benchmarkNameWithRunID := b.GetName() + "-" + *runID
	namespace := benchmarkNameWithRunID
	outputDir := filepath.Join(*outputRootDir, *runID, benchmarkNameWithRunID)
	outputBenchmarkFile := filepath.Join(outputDir, "benchmark")
	// Configure namespaces for things (mostly object references) that cannot be directly overriden by Kustomize.
	klog.V(2).Infof("Setting namespace to %v", namespace)
	b.GetConfig().Namespace = namespace
	if strings.Contains(*manifestTypes, "BenchmarkTool") {
		// Configure the IP for benchmark, based on LoadBalancer configuration
		if b.GetConfig().GetLoadBalancer().GetK8SService() != nil {
			ip := fmt.Sprintf("model-server-service.%v.svc.cluster.local", namespace)
			klog.V(2).Infof("Setting IP to %v", ip)
			b.GetConfig().GetBenchmarkTool().GetLpg().Ip = ip
		}
		if b.GetConfig().GetLoadBalancer().GetGateway() != nil {
			command := fmt.Sprintf("kubectl get service -n envoy-gateway-system -l gateway.envoyproxy.io/owning-gateway-name=inference-gateway,gateway.envoyproxy.io/owning-gateway-namespace=%s | grep envoy | awk '{print $1}'", namespace)
			klog.V(2).Infof("Running command: %v", command)
			gwSvc, err := runBashCommand(command)
			if err != nil {
				klog.Fatalf("Failed to run command to get Gateway IP: %v", err)
			}
			klog.V(2).Infof("Gateway service IP:%v", gwSvc)
			ip := fmt.Sprintf("%v.envoy-gateway-system.svc.cluster.local", gwSvc)
			klog.V(2).Infof("Setting IP to %v", ip)
			b.GetConfig().GetBenchmarkTool().GetLpg().Ip = ip
		}
	}
	utils.SaveBenchmark(b, outputBenchmarkFile, true)

	// Generate manifests.
	bc := b.GetConfig()
	klog.V(2).Infof("Benchmark config: %+v", bc)
	if strings.Contains(*manifestTypes, "ModelServer") {
		gen.GenerateOneManifestType(bc.GetModelServer(), namespace, *catalogDir, "ModelServer", outputDir, *override)
	}
	if strings.Contains(*manifestTypes, "LoadBalancer") {
		gen.GenerateOneManifestType(bc.GetLoadBalancer(), namespace, *catalogDir, "LoadBalancer", outputDir, *override)
	}
	if strings.Contains(*manifestTypes, "BenchmarkTool") {
		gen.GenerateOneManifestType(bc.GetBenchmarkTool(), namespace, *catalogDir, "BenchmarkTool", outputDir, *override)
	}
}

type Generator interface {
	// GenerateOneManifestType generates the manifest yaml for a particular manifest type in the
	// benchmark config, such as the ModelServer. The output will be saved to /outputDir/manifests/manifestType.yaml.
	// the catalogManifestFolder is the workspace dir for kustomize.
	GenerateOneManifestType(msg any, namespace, catalogManifestFolder, manifestType, outputDir string, override bool)
}

type HelmGenerator struct{}

func (h *HelmGenerator) GenerateOneManifestType(msg any, namespace, catalogManifestFolder, manifestType, outputDir string, override bool) {
	chartPath := filepath.Join(catalogManifestFolder, "charts", manifestType)
	_, err := runBashCommand(fmt.Sprintf("helm dependency update %s", chartPath))
	if err != nil {
		klog.Fatalf("Failed to update helm dependencies: %v", err)
	}
	valuesFile := filepath.Join(outputDir, "benchmark.yaml")
	// Example: helm template BenchmarkTool {catalogManifestFolder}/charts/BenchmarkTool -n default -f BenchmarkTool/values.yaml
	helmCommand := fmt.Sprintf("helm template %s %s -n %s -f %s", strings.ToLower(manifestType), chartPath, namespace, valuesFile)

	// Run the helm command
	output, err := runBashCommand(helmCommand)
	if err != nil {
		klog.Fatalf("Failed to run helm command: %v", err)
	}

	// Save the output to a file
	outputFile := filepath.Join(outputDir, "manifests", manifestType+".yaml")
	if err := utils.SaveFile(outputFile, []byte(output), override); err != nil {
		klog.Fatalf("Failed to save helm output: %v", err)
	}
}

func runBashCommand(command string) (string, error) {
	klog.V(1).Infof("Running command: %s", command)
	// Create a new command
	cmd := exec.Command("bash", "-c", command)

	// Create a buffer to capture the output
	var out bytes.Buffer
	cmd.Stdout = &out

	// Run the command
	err := cmd.Run()
	if err != nil {
		return "", err
	}

	// Trim whitespace from the output and return it
	return strings.TrimSpace(out.String()), nil
}
