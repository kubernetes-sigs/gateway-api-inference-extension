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

### Install with Custom Environment Variables

To set custom environment variables for the EndpointPicker deployment, you can define them as free-form YAML in the `values.yaml` file:

```yaml
inferenceExtension:
  env:
    - name: FEATURE_FLAG_ENABLED
      value: "true"
    - name: CUSTOM_ENV_VAR
      value: "custom_value"
    - name: POD_IP
      valueFrom:
        fieldRef:
          fieldPath: status.podIP
```

Then apply it with:

```txt
$ helm install vllm-llama3-8b-instruct ./config/charts/inferencepool -f values.yaml
```

### Install with Additional Ports

To expose additional ports (e.g., for ZMQ), you can define them in the `values.yaml` file:

```yaml
inferenceExtension:
  extraContainerPorts:
    - name: zmq
      containerPort: 5557
      protocol: TCP
  extraServicePorts: # if need to expose the port for external communication
    - name: zmq
      port: 5557
      protocol: TCP
```

Then apply it with:

```txt
$ helm install vllm-llama3-8b-instruct ./config/charts/inferencepool -f values.yaml
```

### Install for Triton TensorRT-LLM

Use `--set inferencePool.modelServerType=triton-tensorrt-llm` to install for Triton TensorRT-LLM, e.g.,

```txt
$ helm install triton-llama3-8b-instruct \
  --set inferencePool.modelServers.matchLabels.app=triton-llama3-8b-instruct \
  --set inferencePool.modelServerType=triton-tensorrt-llm \
  --set provider.name=[none|gke] \
  oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool --version v0
```

### Install with High Availability (HA)

To deploy the EndpointPicker in a high-availability (HA) active-passive configuration, you can enable leader election. When enabled, the EPP deployment will have multiple replicas, but only one "leader" replica will be active and ready to process traffic at any given time. If the leader pod fails, another pod will be elected as the new leader, ensuring service continuity.

To enable HA, set `inferenceExtension.flags.has-enable-leader-election` to `true` and increase the number of replicas in your `values.yaml` file:

```yaml
inferenceExtension:
  replicas: 3
  has-enable-leader-election: true
```

Then apply it with:

```txt
helm install vllm-llama3-8b-instruct ./config/charts/inferencepool -f values.yaml
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
| `inferenceExtension.env`                    | List of environment variables to set in the endpoint picker container as free-form YAML. Defaults to `[]`.             |
| `inferenceExtension.extraContainerPorts`    | List of additional container ports to expose. Defaults to `[]`.                                                        |
| `inferenceExtension.extraServicePorts`      | List of additional service ports to expose. Defaults to `[]`.                                                          |
| `inferenceExtension.flags`                  | List of flags which are passed through to endpoint picker. Example flags, enable-pprof, grpc-port etc. Refer [runner.go](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/cmd/epp/runner/runner.go) for complete list.                                                            |
| `inferenceExtension.affinity`               | Affinity for the endpoint picker. Defaults to `{}`.                                                                    |
| `inferenceExtension.tolerations`            | Tolerations for the endpoint picker. Defaults to `[]`.                                                                 |
| `inferenceExtension.flags.has-enable-leader-election` | Enable leader election for high availability. When enabled, only one EPP pod (the leader) will be ready to serve traffic.       |
| `provider.name`                             | Name of the Inference Gateway implementation being used. Possible values: `gke`. Defaults to `none`.                   |

## Notes

This chart will only deploy an InferencePool and its corresponding EndpointPicker extension. Before install the chart, please make sure that the inference extension CRDs are installed in the cluster. For more details, please refer to the [getting started guide](https://gateway-api-inference-extension.sigs.k8s.io/guides/).
