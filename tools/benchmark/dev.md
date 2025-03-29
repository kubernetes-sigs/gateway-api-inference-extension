## The manifest generator

[code](./manifestgenerator)

The manifestgenerator takes in the `Benchmark` proto as input, and generates benchmark manifests for each part of the config (ModelServer, LoadBalancer, BenchmarkTool), using Helm.

### Benchmark inheritance

Each benchmark MUST have a name, and optionally the name of its base benchmark. The scope of the inheritance is limited to the benchmarks in the same pbtxt file.

### Determine the LoadBalancer address

The address of the LoadBalancer is usually known at runtime after the load balancer is deployed. In the case of EPP, we wait for the corresponding Envoy service to be ready. If we benchmark against a public gateway IP, we may wait for the gateway IP to be available. Therefore, we can specify the manifestgenerator to generate the `ModelServer` and `LoadBalancer` manifest types first, then call it again to generate the manifest for `BenchmarkTool`, after the `LoadBalancer` is ready.

### Auto set the request rate

The tool can automatically choose a curated list of request rates based on # accelerator, accelerator type and model, if the request rates are not specified by the user.

## FAQs

### How to add a new config field

1. Update proto/benchmark.proto.
1. Regenerate the go file: `protoc --go_out=. --go_opt=paths=source_relative benchmark.proto`
1. For Helm generator, edit corresponding helm templates to parse the new field.

### How to add a new accelerator type or model

Update `applyRequestRatesDefaults` in `pkg/utils/benchmark_config.go` for the new accelerator type and/or model.


