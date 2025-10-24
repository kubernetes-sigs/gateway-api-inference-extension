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
  --set provider.name=[none|gke|istio] \
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

### Install with Custom EPP Plugins Configuration

To set custom EPP plugin config, you can pass it as an inline yaml. For example:

```yaml
  inferenceExtension:
    pluginsCustomConfig:
      custom-plugins.yaml: |
        apiVersion: inference.networking.x-k8s.io/v1alpha1
        kind: EndpointPickerConfig
        plugins:
        - type: custom-scorer
          parameters:
            custom-threshold: 64
        schedulingProfiles:
        - name: default
          plugins:
          - pluginRef: custom-scorer
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
  --set provider.name=[none|gke|istio] \
  oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool --version v0
```

### Install with High Availability (HA)

To deploy the EndpointPicker in a high-availability (HA) active-passive configuration set replicas to be greater than one. In such a setup, only one "leader" replica will be active and ready to process traffic at any given time. If the leader pod fails, another pod will be elected as the new leader, ensuring service continuity.

To enable HA, set `inferenceExtension.replicas` to a number greater than 1.

* Via `--set` flag:

  ```txt
  helm install vllm-llama3-8b-instruct \
  --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
  --set inferenceExtension.replicas=3 \
  --set provider=[none|gke] \
  oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool --version v0
  ```

* Via `values.yaml`:

  ```yaml
  inferenceExtension:
    replicas: 3
  ```

  Then apply it with:

  ```txt
  helm install vllm-llama3-8b-instruct ./config/charts/inferencepool -f values.yaml
  ```

### Install with Monitoring

To enable metrics collection and monitoring for the EndpointPicker, you can configure Prometheus ServiceMonitor creation:

```yaml
inferenceExtension:
  monitoring:
    interval: "10s"
    prometheus:
      enabled: false
      auth:
        enabled: true
        secretName: inference-gateway-sa-metrics-reader-secret
      extraLabels: {}
```

**Note:** Prometheus monitoring requires the Prometheus Operator and ServiceMonitor CRD to be installed in the cluster.

For GKE environments, you need to set `provider.name` to `gke` firstly. This will create the necessary `PodMonitoring` and RBAC resources for metrics collection.

If you are using a GKE Autopilot cluster, you also need to set `provider.gke.autopilot` to `true`.

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

| **Parameter Name**                                         | **Description**                                                                                                                                                                                                                                    |
|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `inferencePool.targetPortNumber`                           | Target port number for the vllm backends, will be used to scrape metrics by the inference extension. Defaults to 8000.                                                                                                                             |
| `inferencePool.modelServerType`                            | Type of the model servers in the pool, valid options are [vllm, triton-tensorrt-llm], default is vllm.                                                                                                                                             |
| `inferencePool.modelServers.matchLabels`                   | Label selector to match vllm backends managed by the inference pool.                                                                                                                                                                               |
| `inferenceExtension.replicas`                              | Number of replicas for the endpoint picker extension service. If More than one replica is used, EPP will run in HA active-passive mode. Defaults to `1`.                                                                                           |
| `inferenceExtension.image.name`                            | Name of the container image used for the endpoint picker.                                                                                                                                                                                          |
| `inferenceExtension.image.hub`                             | Registry URL where the endpoint picker image is hosted.                                                                                                                                                                                            |
| `inferenceExtension.image.tag`                             | Image tag of the endpoint picker.                                                                                                                                                                                                                  |
| `inferenceExtension.image.pullPolicy`                      | Image pull policy for the container. Possible values: `Always`, `IfNotPresent`, or `Never`. Defaults to `Always`.                                                                                                                                  |
| `inferenceExtension.env`                                   | List of environment variables to set in the endpoint picker container as free-form YAML. Defaults to `[]`.                                                                                                                                         |
| `inferenceExtension.extraContainerPorts`                   | List of additional container ports to expose. Defaults to `[]`.                                                                                                                                                                                    |
| `inferenceExtension.extraServicePorts`                     | List of additional service ports to expose. Defaults to `[]`.                                                                                                                                                                                      |
| `inferenceExtension.flags`                                 | List of flags which are passed through to endpoint picker. Example flags, enable-pprof, grpc-port etc. Refer [runner.go](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/cmd/epp/runner/runner.go) for complete list. |
| `inferenceExtension.affinity`                              | Affinity for the endpoint picker. Defaults to `{}`.                                                                                                                                                                                                |
| `inferenceExtension.tolerations`                           | Tolerations for the endpoint picker. Defaults to `[]`.                                                                                                                                                                                             |
| `inferenceExtension.monitoring.interval`                   | Metrics scraping interval for monitoring. Defaults to `10s`.                                                                                                                                                                                       |
| `inferenceExtension.monitoring.prometheus.enabled`         | Enable Prometheus ServiceMonitor creation for EPP metrics collection. Defaults to `false`.                                                                                                                                                         |
| `inferenceExtension.monitoring.gke.enabled`                | **DEPRECATED**: This field is deprecated and will be removed in the next release.  Enable GKE monitoring resources (`PodMonitoring` and RBAC). Defaults to `false`.                                                                                |
| `inferenceExtension.monitoring.prometheus.auth.enabled`    | Enable auth for Prometheus metrics endpoint. Defaults is `true`                                                                                                                                                                                    |
| `inferenceExtension.monitoring.prometheus.auth.secretName` | Name of the service account token secret for metrics authentication. Defaults to `inference-gateway-sa-metrics-reader-secret`.                                                                                                                     |
| `inferenceExtension.monitoring.prometheus.extraLabels`     | Extra labels added to ServiceMonitor.                                                                                                                                                                                                              |
| `inferenceExtension.pluginsCustomConfig`                   | Custom config that is passed to EPP as inline yaml.                                                                                                                                                                                                |
| `inferenceExtension.tracing.enabled`                       | Enables or disables OpenTelemetry tracing globally for the EndpointPicker.                                                                                                                                                                         |
| `inferenceExtension.tracing.otelExporterEndpoint`          | OpenTelemetry collector endpoint.                                                                                                                                                                                                                  |
| `inferenceExtension.tracing.sampling.sampler`              | The trace sampler to use. Currently, only `parentbased_traceidratio` is supported. This sampler respects the parent span’s sampling decision when present, and applies the configured ratio for root spans.                                        |
| `inferenceExtension.tracing.sampling.samplerArg`           | Sampler-specific argument. For `parentbased_traceidratio`, this defines the base sampling rate for new traces (root spans), as a float string in the range [0.0, 1.0]. For example, "0.1" enables 10% sampling.                                    |
| `provider.name`                                            | Name of the Inference Gateway implementation being used. Possible values: [`none`, `gke`, or `istio`]. Defaults to `none`.                                                                                                                         |
| `provider.gke.autopilot`                                   | Set to `true` if the cluster is a GKE Autopilot cluster. This is only used if `provider.name` is `gke`. Defaults to `false`.                                                                                                                       |

### Provider Specific Configuration

This section should document any Gateway provider specific values configurations.

#### GKE

These are the options available to you with `provider.name` set to `gke`:

| **Parameter Name**                          | **Description**                                                                                                        |
|---------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| `gke.monitoringSecret.name`                 | The name of the monitoring secret to be used. Defaults to `inference-gateway-sa-metrics-reader-secret`.                |
| `gke.monitoringSecret.namespace`            | The namespace that the monitoring secret lives in. Defaults to `default`.                                              |


#### Istio

These are the options available to you with `provider.name` set to `istio`:

| **Parameter Name**                          | **Description**                                                                                                        |
|---------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| `istio.destinationRule.host`            | Custom host value for the destination rule. If not set this will use the default value which is derrived from the epp service name and release namespace to gerenate a valid service address. |
| `istio.destinationRule.trafficPolicy.connectionPool`            | Configure the connectionPool level settings of the traffic policy |

#### OpenTelemetry

The EndpointPicker supports OpenTelemetry-based tracing. To enable trace collection, use the following configuration:
```yaml
inferenceExtension:
  tracing:
    enabled: true
    otelExporterEndpoint: "http://localhost:4317"
    sampling:
      sampler: "parentbased_traceidratio"
      samplerArg: "0.1"
```
Make sure that the `otelExporterEndpoint` points to your OpenTelemetry collector endpoint. 
Current only the `parentbased_traceidratio` sampler is supported. You can adjust the base sampling ratio using the `samplerArg` (e.g., 0.1 means 10% of traces will be sampled).

## Notes

This chart will only deploy an InferencePool and its corresponding EndpointPicker extension. Before install the chart, please make sure that the inference extension CRDs are installed in the cluster. For more details, please refer to the [getting started guide](https://gateway-api-inference-extension.sigs.k8s.io/guides/).
