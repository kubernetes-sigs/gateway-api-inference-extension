# InferencePool

A chart to deploy an InferencePool and a corresponding EndpointPicker (epp) deployment.  

## Install

To install an InferencePool named `vllm-llama3-8b-instruct`  that selects from endpoints with label `app: vllm-llama3-8b-instruct` and listening on port `8000`, you can run the following command:

```txt
$ helm install vllm-llama3-8b-instruct ./config/charts/inferencepool \
  --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
```

To install via the latest published chart in staging  (--version v0 indicates latest dev version), you can run the following command:

```txt
$ helm install vllm-llama3-8b-instruct \
  --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
  --set provider.name=[none|gke] \
  oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool --version v0
```

Note that the provider name is needed to deploy provider-specific resources. If no provider is specified, then only the InferencePool object and the EPP are deployed.

### Install for Triton TensorRT-LLM

Use `--set inferencePool.modelServerType=triton-tensorrt-llm` to install for Triton TensorRT-LLM, e.g.,

```txt
$ helm install triton-llama3-8b-instruct \
  --set inferencePool.modelServers.matchLabels.app=triton-llama3-8b-instruct \
  --set inferencePool.modelServerType=triton-tensorrt-llm \
  --set provider.name=[none|gke] \
  oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool --version v0
```

## Uninstall

Run the following command to uninstall the chart:

```txt
$ helm uninstall pool-1
```

## Configuration

The following table list the configurable parameters of the chart.

| **Parameter Name**                          | **Description**                                                                                                        |
|---------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| `inferencePool.targetPortNumber`            | Target port number for the vllm backends, will be used to scrape metrics by the inference extension. Defaults to 8000. |
| `inferencePool.modelServerType`            | Type of the model servers in the pool, valid options are [vllm, triton-tensorrt-llm], default is vllm. |
| `inferencePool.modelServers.matchLabels`    | Label selector to match vllm backends managed by the inference pool.                                                   |
| `inferenceExtension.replicas`               | Number of replicas for the endpoint picker extension service. Defaults to `1`.                                         |
| `inferenceExtension.image.name`             | Name of the container image used for the endpoint picker.                                                              |
| `inferenceExtension.image.hub`              | Registry URL where the endpoint picker image is hosted.                                                                |
| `inferenceExtension.image.tag`              | Image tag of the endpoint picker.                                                                                      |
| `inferenceExtension.image.pullPolicy`       | Image pull policy for the container. Possible values: `Always`, `IfNotPresent`, or `Never`. Defaults to `Always`.      |
| `inferenceExtension.extProcPort`            | Port where the endpoint picker service is served for external processing. Defaults to `9002`.                          |
| `provider.name`                             | Name of the Inference Gateway implementation being used. Possible values: `gke`. Defaults to `none`.                   |

## Notes

This chart will only deploy an InferencePool and its corresponding EndpointPicker extension. Before install the chart, please make sure that the inference extension CRDs are installed in the cluster. For more details, please refer to the [getting started guide](https://gateway-api-inference-extension.sigs.k8s.io/guides/).
