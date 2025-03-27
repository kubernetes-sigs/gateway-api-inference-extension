# InferencePool

A chart to deploy an InferencePool and a corresponding EndpointPicker (epp) deployment.  


## Install

To install an InferencePool named `vllm-llama2-7b`  that selects from endpoints with label `app: vllm-llama2-7b` and listening on port `8000`, you can run the following command:

```txt
$ helm install vllm-llama2-7b ./config/charts/inferencepool \
  --set inferencePool.name=vllm-llama2-7b \
  --set inferencePool.modelServers.matchLabels.app=vllm-llama2-7b \
  --set inferencePool.targetPortNumber=8000
```

where `inferencePool.targetPortNumber` is the pod that vllm backends served on and `inferencePool.modelServers.matchLabels` is the selector to match the vllm backends.

To install via the latest published chart in staging  (--version v0 indicates latest dev version), you can run the following command:

```txt
$ helm install vllm-llama2-7b \
  --set inferencePool.name=vllm-llama2-7b \
  --set inferencePool.modelServers.matchLabels.app=vllm-llama2-7b \
  --set inferencePool.targetPortNumber=8000 \
  oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool --version v0
```

## Uninstall

Run the following command to uninstall the chart:

```txt
$ helm uninstall pool-1
```

## Configuration

The following table list the configurable parameters of the chart.

| **Parameter Name**                          | **Description**                                                                                                   |
|---------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `inferencePool.name`                        | Name for the InferencePool, and inference extension will be named as `${inferencePool.name}-epp`.                |
| `inferencePool.targetPortNumber`            | Target port number for the vllm backends, will be used to scrape metrics by the inference extension.             |
| `inferencePool.modelServers.matchLabels`    | Label selector to match vllm backends managed by the inference pool.                                             |
| `inferenceExtension.replicas`               | Number of replicas for the inference extension service. Defaults to `1`.                                           |
| `inferenceExtension.image.name`             | Name of the container image used for the inference extension.                                                    |
| `inferenceExtension.image.hub`              | Registry URL where the inference extension image is hosted.                                                     |
| `inferenceExtension.image.tag`              | Image tag of the inference extension.                                                                             |
| `inferenceExtension.image.pullPolicy`       | Image pull policy for the container. Possible values: `Always`, `IfNotPresent`, or `Never`. Defaults to `Always`. |
| `inferenceExtension.extProcPort`            | Port where the inference extension service is served for external processing. Defaults to `9002`.                  |

## Notes

This chart will only deploy an InferencePool and its corresponding EndpointPicker extension. Before install the chart, please make sure that the inference extension CRDs are installed in the cluster. For more details, please refer to the [getting started guide](https://gateway-api-inference-extension.sigs.k8s.io/guides/).
