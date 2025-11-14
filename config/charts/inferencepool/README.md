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

### Install with Custom Cmd-line Flags

To set cmd-line flags, you can use the `--set` option to set each flag, e.g.,:

```txt
$ helm install vllm-llama3-8b-instruct \
  --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
  --set inferenceExtension.flags.<FLAG_NAME>=<FLAG_VALUE>
  --set provider.name=[none|gke|istio] \
  oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool --version v0
```

Alternatively, you can define flags in the `values.yaml` file:

```yaml
inferenceExtension:
  flags:
    FLAG_NAME: <FLAG_VALUE>
    v: 3 ## Log verbosity
    ...
```

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
| `inferencePool.apiVersion`                                 | The API version of the InferencePool resource. Defaults to `inference.networking.k8s.io/v1`. This can be changed to `inference.networking.x-k8s.io/v1alpha2` to support older API versions.                                                        |
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
| `inferenceExtension.flags`                                 | map of flags which are passed through to endpoint picker. Example flags, enable-pprof, grpc-port etc. Refer [runner.go](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/cmd/epp/runner/runner.go) for complete list. |
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

#### Jaeger Tracing Backend

GAIE provides an opt-in Jaeger all-in-one deployment as a sub-chart for easy trace collection and visualization. This is particularly useful for development, testing, and understanding how inference requests are processed (filtered, scored) and forwarded to vLLM models.

**Quick Start with Jaeger:**

To install the InferencePool with Jaeger tracing enabled:

```bash
# Update Helm dependencies to fetch Jaeger chart
helm dependency update ./config/charts/inferencepool

# Install with Jaeger enabled
helm install vllm-llama3-8b-instruct ./config/charts/inferencepool \
  --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
  --set inferenceExtension.tracing.enabled=true \
  --set jaeger.enabled=true
```

Or using a `values.yaml` file:

```yaml
inferenceExtension:
  tracing:
    enabled: true
    sampling:
      sampler: "parentbased_traceidratio"
      samplerArg: "1.0"  # 100% sampling for development

jaeger:
  enabled: true
```

Then install:

```bash
helm dependency update ./config/charts/inferencepool
helm install vllm-llama3-8b-instruct ./config/charts/inferencepool -f values.yaml
```

**Accessing Jaeger UI:**

Once deployed, you can access the Jaeger UI to visualize traces:

```bash
# Port-forward to access Jaeger UI
kubectl port-forward svc/vllm-llama3-8b-instruct-jaeger-query 16686:16686

# Open browser to http://localhost:16686
```

In the Jaeger UI, you can:
- Search for traces by service name (`gateway-api-inference-extension`)
- View detailed span information showing filter and scorer execution
- Analyze request routing decisions and latency
- Understand the complete inference request flow

**Configuration Options:**

The Jaeger sub-chart supports the following configuration:

| **Parameter Name**                    | **Description**                                                                                     | **Default**                      |
|---------------------------------------|-----------------------------------------------------------------------------------------------------|----------------------------------|
| `jaeger.enabled`                      | Enable Jaeger all-in-one deployment                                                                 | `false`                          |
| `jaeger.allInOne.enabled`             | Enable all-in-one deployment mode                                                                   | `true`                           |
| `jaeger.allInOne.image.repository`    | Jaeger all-in-one image repository                                                                  | `jaegertracing/all-in-one`       |
| `jaeger.allInOne.image.tag`           | Jaeger image tag                                                                                    | `1.62`                           |
| `jaeger.allInOne.resources.limits`    | Resource limits for Jaeger pod                                                                      | `cpu: 500m, memory: 512Mi`       |
| `jaeger.allInOne.resources.requests`  | Resource requests for Jaeger pod                                                                    | `cpu: 100m, memory: 128Mi`       |
| `jaeger.query.service.type`           | Jaeger UI service type                                                                              | `ClusterIP`                      |
| `jaeger.query.service.port`           | Jaeger UI port                                                                                      | `16686`                          |
| `jaeger.collector.service.otlp.grpc.port` | OTLP gRPC collector port                                                                        | `4317`                           |
| `jaeger.storage.type`                 | Storage backend type (memory, elasticsearch, cassandra, etc.)                                       | `memory`                         |

**Important Notes:**

1. **Development vs Production**: The all-in-one deployment uses in-memory storage and is suitable for development and testing. For production use, consider:
   - Using a persistent storage backend (Elasticsearch, Cassandra, etc.)
   - Deploying Jaeger components separately for better scalability
   - Refer to [Jaeger Production Deployment](https://www.jaegertracing.io/docs/latest/deployment/) for best practices

2. **Automatic Configuration**: When `jaeger.enabled=true`, the OTLP exporter endpoint is automatically configured to point to the Jaeger collector. You don't need to manually set `inferenceExtension.tracing.otelExporterEndpoint`.

3. **Sampling Rate**: For development, you may want to set `samplerArg: "1.0"` to capture all traces. For production, use a lower value like `"0.1"` (10%) to reduce overhead.

4. **Resource Requirements**: Adjust the resource limits based on your trace volume and cluster capacity.

## Notes

This chart will only deploy an InferencePool and its corresponding EndpointPicker extension. Before install the chart, please make sure that the inference extension CRDs are installed in the cluster. For more details, please refer to the [getting started guide](https://gateway-api-inference-extension.sigs.k8s.io/guides/).
