# Gateway Api Inference Extension

A chart to deploy the inference extension and a InferencePool managed by the extension.

## Install

Suppose now a vllm service with label `app: vllm-llama2-7b` and served on port `8000` is deployed in `default` namespace in the cluster.

To deploy the inference extension, you can run the following command:

```txt
$ helm install my-release . -n default \
    --set inferencePool.targetPortNumber=8000 \
    --set inferencePool.selector.app=vllm-llama2-7b
```

Or you can change the `values.yaml` to:

```yaml
inferencePool:
  name: pool-1
  targetPortNumber: 8000
  selector:
    app: vllm-llama2-7b
```

where `inferencePool.targetPortNumber` is the pod that vllm backends served on and `inferencePool.selector` is the selector to match the vllm backends. And then run:

```txt
$ helm install my-release .
```

## Uninstall

Run the following command to uninstall the chart:

```txt
$ helm uninstall my-release
```

## Configuration

The following table list the configurable parameters of the chart.

| **Parameter Name**                          | **Description**                                                                                                   |
|---------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `inferenceExtension.replicas`               | Number of replicas for the inference extension service. Defaults to `1`.                                           |
| `inferenceExtension.image.name`             | Name of the container image used for the inference extension.                                                    |
| `inferenceExtension.image.hub`              | Registry URL where the inference extension image is hosted.                                                     |
| `inferenceExtension.image.tag`              | Image tag of the inference extension.                                                                             |
| `inferenceExtension.image.pullPolicy`       | Image pull policy for the container. Possible values: `Always`, `IfNotPresent`, or `Never`. Defaults to `Always`. |
| `inferenceExtension.extProcPort`            | Port where the inference extension service is served for external processing. Defaults to `9002`.                  |
| `inferencePool.name`                        | Name for the InferencePool, and inference extension will be named as `${inferencePool.name}-epp`.                |
| `inferencePool.targetPortNumber`            | Target port number for the vllm backends, will be used to scrape metrics by the inference extension.             |
| `inferencePool.selector`                     | Label selector to match vllm backends managed by the inference pool.                                             |

## Notes

This chart will only deploy the inference extension and InferencePool, before install the chart, please make sure that the inference extension CRDs have already been installed in the cluster. And You need to apply traffic policies to route traffic to the inference extension from the gateway after the inference extension is deployed.

For more details, please refer to the [website](https://gateway-api-inference-extension.sigs.k8s.io/guides/).