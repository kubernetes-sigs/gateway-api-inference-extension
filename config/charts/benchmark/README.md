# Benchmark

A chart to deploy the benchmark tool on top of vLLM model server deployment done via [getting started guide](https://gateway-api-inference-extension.sigs.k8s.io/guides/#getting-started-with-gateway-api-inference-extension)


## Install

To install benchmark tool 

```txt
$ helm install benchmark-tool ./config/charts/benchmark \
    --set moderlServingEndpoint.mode=gateway \
    --set moderlServingEndpoint.name=inference-gateway \
    --set moderlServingEndpoint.namespace=default
```

## Uninstall

Run the following command to uninstall the chart:

```txt
$ helm uninstall benchmark-tool
```

## Configuration

The following table list the configurable parameters of the chart.

| **Parameter Name**                          | **Description**                                                                                    |
|---------------------------------------------|----------------------------------------------------------------------------------------------------|
| `benchmark.requestRates`                    | Comma separated list of number of requests per second. For each request rate benchmarking would be done against the vLLM deployment.                                             |
| `benchmark.timeSeconds`                     | Number of prompts will be calculated following this forumula `requestRate * timeSeconds` for each requestRate.                                                                  |
| `benchmark.maxNumPrompts`                   | Maximum number of prompts to process. Will be considered when `requestRates` is not set.                                                                                                |
| `benchmark.tokenizer`                       | Name or path of the tokenizer.                                                                                          |
| `benchmark.models`                          | Comma separated list of models to benchmark.                                                                                          |
| `benchmark.backend`                         | Model serving backend. Default: vllm                                                                                                |
| `benchmark.port`                            | Model serving backend server's port                                                                                                | 
| `benchmark.inputLength`                     | Maximum number of input tokens for filtering the benchmark dataset.                                                                                            |
| `benchmark.outputLength`                    | Maximum number of output tokens for filtering the benchmark dataset. |
| `benchmark.filePrefix`                      | Prefix to use for benchmark result's output file .  |
| `benchmark.trafficSplit`                    | Comma-separated list of traffic split proportions for the models, e.g. '0.9,0.1'. Sum must equal 1.0.                                                                                 |
| `benchmark.scrapeServerMetrics`             | Whether to scrape server metrics.                                                                                            |
| `benchmark.saveAggregatedResult`            | Whether to aggregate results of all models and save the result.                                                                                             |
| `benchmark.streamRequest`                   | Whether to stream the request. Needed for TTFT metric                                                                                              |
| `benchmark.trafficSplit`                    | Comma-separated list of traffic split proportions for the models, e.g. '0.9,0.1'. Sum must equal 1.0.                                                                                 |
| `benchmark.trafficSplit`                    | Comma-separated list of traffic split proportions for the models, e.g. '0.9,0.1'. Sum must equal 1.0.                                                                                 |
| `benchmark.trafficSplit`                    | Comma-separated list of traffic split proportions for the models, e.g. '0.9,0.1'. Sum must equal 1.0.                                                                                 |
| `benchmark.trafficSplit`                    | Comma-separated list of traffic split proportions for the models, e.g. '0.9,0.1'. Sum must equal 1.0.                                                                                 |
| `moderlServingEndpoint.mode`                | Mode in which you want the LPG tool to consume the model serving endpoint for benchmarking. Options are gateway or service                                                                      |                        
| `moderlServingEndpoint.name`                | Provide model serving endpoint's resource name. i.e. name of inference gateway or load balancer service name                                                                          |                        
| `moderlServingEndpoint.namespace`           | Namespace of  moderlServingEndpoint resource. i.e. namespace of inference gateway or load balancer service name                                                                          |                        
