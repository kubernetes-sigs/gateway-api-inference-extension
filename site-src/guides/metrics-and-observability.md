# Metrics & Observability

This guide describes the current state of exposed metrics and how to scrape them, as well as accessing pprof profiles.

## Requirements

=== "EPP"

      To have response metrics, ensure the body mode is set to `Buffered` or `Streamed` (this should be the default behavior for all implementations).

      If you want to include usage metrics for vLLM model server streaming request, send the request with `include_usage`:

      ```
      curl -i ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{
      "model": "food-review",
      "prompt": "whats your fav movie?",
      "max_tokens": 10,
      "temperature": 0,
      "stream": true,
      "stream_options": {"include_usage": true}
      }'
      ```

=== "Dynamic LoRA Adapter Sidecar"

      To have response metrics, ensure the vLLM model server is configured with the dynamic LoRA adapter as a sidecar container and a ConfigMap to configure which models to load/unload. See [this doc](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/tools/dynamic-lora-sidecar#example-configuration) for an example.


## Exposed metrics

### EPP

| **Metric name**                              | **Metric Type**  | <div style="width:200px">**Description**</div>  | <div style="width:250px">**Labels**</div>                                          | **Status**  |
|:---------------------------------------------|:-----------------|:------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:------------|
| inference_objective_request_total                | Counter          | The counter of requests broken out for each model.                | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt; | ALPHA       |
| inference_objective_request_error_total          | Counter          | The counter of requests errors broken out for each model.         | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt; | ALPHA       |
| inference_objective_request_duration_seconds     | Distribution     | Distribution of response latency.                                 | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt; | ALPHA       |
| inference_objective_normalized_time_per_output_token_seconds     | Distribution     | Distribution of ntpot (response latency per output token)                                 | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt; | ALPHA       |
| inference_objective_request_sizes                | Distribution     | Distribution of request size in bytes.                            | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt; | ALPHA       |
| inference_objective_response_sizes               | Distribution     | Distribution of response size in bytes.                           | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt; | ALPHA       |
| inference_objective_input_tokens                 | Distribution     | Distribution of input token count.                                | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt; | ALPHA       |
| inference_objective_output_tokens                | Distribution     | Distribution of output token count.                               | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt; | ALPHA       |
| inference_objective_running_requests                | Gauge     | Number of running requests for each model.             | `model_name`=&lt;model-name&gt;  | ALPHA       |
| inference_pool_average_kv_cache_utilization  | Gauge            | The average kv cache utilization for an inference server pool.    | `name`=&lt;inference-pool-name&gt;                                                 | ALPHA       |
| inference_pool_average_queue_size            | Gauge            | The average number of requests pending in the model server queue. | `name`=&lt;inference-pool-name&gt;                                                 | ALPHA       |
| inference_pool_per_pod_queue_size            | Gauge            | The total number of queue for each model server pod under the inference pool         | `model_server_pod`=&lt;model-server-pod-name&gt; <br> `name`=&lt;inference-pool-name&gt;                             | ALPHA       |
| inference_pool_ready_pods                    | Gauge            | The number of ready pods for an inference server pool.            | `name`=&lt;inference-pool-name&gt;                                                 | ALPHA       |
| inference_extension_info                     | Gauge            | The general information of the current build.                     | `commit`=&lt;hash-of-the-build&gt; <br> `build_ref`=&lt;ref-to-the-build&gt;        | ALPHA       |

### Dynamic LoRA Adapter Sidecar

| **Metric name**            | **Metric Type**  | <div style="width:200px">**Description**</div>   | <div style="width:250px">**Labels**</div> | **Status**  |
|:---------------------------|:-----------------|:-------------------------------------------------|:------------------------------------------|:------------|
| lora_syncer_adapter_status | Gauge            | Status of LoRA adapters (1=loaded, 0=not_loaded) | `adapter_name`=&lt;adapter-id&gt;         | ALPHA       |

### Flow Control Metrics (Experimental)

These metrics provide insights into the experimental flow control layer within the EPP.

| **Metric name** | **Metric Type**  | <div style="width:200px">**Description**</div>  | <div style="width:250px">**Labels**</div> | **Status**  |
|:---|:---|:---|:---|:---|
| inference_extension_flow_control_request_queue_duration_seconds | Distribution | Distribution of the total time requests spend in the flow control layer. This is measured from the moment a request enters the `EnqueueAndWait` function until it reaches a final outcome (e.g., Dispatched, Rejected, Evicted). | `fairness_id`=&lt;flow-id&gt; <br> `priority`=&lt;flow-priority&gt; <br> `outcome`=&lt;QueueOutcome&gt; | ALPHA |
| inference_extension_flow_control_queue_size | Gauge | The current number of requests being actively managed by the flow control layer. This counts requests from the moment they enter the `EnqueueAndWait` function until they reach a final outcome. | `fairness_id`=&lt;flow-id&gt; <br> `priority`=&lt;flow-priority&gt; | ALPHA |

## Scrape Metrics & Pprof profiles

The metrics endpoints are exposed on different ports by default:

- EPP exposes the metrics endpoint at port 9090
- Dynamic LoRA adapter sidecar exposes the metrics endpoint at port 8080

To scrape metrics, the client needs a ClusterRole with the following rule:
`nonResourceURLs: "/metrics", verbs: get`.

Here is one example if the client needs to mound the secret to act as the service account
```
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: inference-gateway-metrics-reader
rules:
- nonResourceURLs:
  - /metrics
  - /debug/pprof/*
  verbs:
  - get
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: inference-gateway-sa-metrics-reader
  namespace: default
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: inference-gateway-sa-metrics-reader-role-binding
  namespace: default
subjects:
- kind: ServiceAccount
  name: inference-gateway-sa-metrics-reader
  namespace: default
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: inference-gateway-metrics-reader
---
apiVersion: v1
kind: Secret
metadata:
  name: inference-gateway-sa-metrics-reader-secret
  namespace: default
  annotations:
    kubernetes.io/service-account.name: inference-gateway-sa-metrics-reader
type: kubernetes.io/service-account-token
```

Then, you can curl the appropriate port as follows. For EPP (port 9090)

```
TOKEN=$(kubectl -n default get secret inference-gateway-sa-metrics-reader-secret  -o jsonpath='{.secrets[0].name}' -o jsonpath='{.data.token}' | base64 --decode)

kubectl -n default port-forward inference-gateway-ext-proc-pod-name  9090

curl -H "Authorization: Bearer $TOKEN" localhost:9090/metrics
```

### Pprof profiles

Currently only the [predefined profiles](https://pkg.go.dev/runtime/pprof#Profile) are supported, CPU profiling will require code changes. Assuming the EPP has been port-forwarded as in the above example, to get the PGN display of the `heap` profile simply run:

```
PROFILE_NAME=heap
curl -H "Authorization: Bearer $TOKEN" localhost:9090/debug/pprof/$PROFILE_NAME -o profile.out
go tool pprof -png profile.out
```
## Setting Up Grafana + Prometheus

### Grafana

A simple grafana deployment can be done with the following commands:

```bash
helm repo add grafana https://grafana.github.io/helm-charts
helm install grafana grafana/grafana --namespace monitoring --create-namespace
```

Get the Grafana URL to visit by running these commands in the same shell:

```bash
kubectl -n monitoring port-forward deploy/grafana 3000:3000
```

Get the generated password for the `admin` user:

```bash
kubectl -n monitoring get secret grafana \
  -o go-template='{% raw %}{{ index .data "admin-password" | base64decode }}{% endraw %}'
```

You can now access the Grafana UI from [http://127.0.0.1:3000](http://127.0.0.1:3000)

### Prometheus

We currently have 2 types of prometheus deployments documented:

1. Self Hosted using the prometheus helm chart
2. Using Google Managed Prometheus

=== "Self-Hosted"

    Create necessary ServiceAccount and RBAC resources:

     ```bash
        kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/observability/prometheus/rbac.yaml
     ```

    Patch the metrics reader ClusterRoleBinding to reference the new ServiceAccount:
    ```bash
       kubectl patch clusterrolebinding inference-gateway-sa-metrics-reader-role-binding \
         --type='json' \
         -p='[{"op": "replace", "path": "/subjects/0/namespace", "value": "monitoring"}]'
    ```

    Add the prometheus-community helm repository:

     ```bash
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
     ```

    Deploy the prometheus helm chart using this command:
     ```bash
        helm install prometheus prometheus-community/prometheus \
         --namespace monitoring \
         -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/observability/prometheus/values.yaml
     ```

    You can add the prometheus data source to grafana following [This Guide](https://grafana.com/docs/grafana/latest/administration/data-source-management/).
    The prometheus server host is by default `http://prometheus-server`

    Notice that the given values file is very simple and will work directly after following the [Getting Started Guide](https://gateway-api-inference-extension.sigs.k8s.io/guides/), you might need to modify it

=== "Google Managed"

    If you run the inference gateway with [Google Managed Prometheus](https://cloud.google.com/stackdriver/docs/managed-prometheus), please follow the [instructions](https://cloud.google.com/stackdriver/docs/managed-prometheus/query)
    to configure Google Managed Prometheus as data source for the grafana dashboard.

## Load Inference Extension dashboard into Grafana

Please follow [grafana instructions](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/import-dashboards/) to load the dashboard json.
The dashboard can be found here [Grafana Dashboard](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/tools/dashboards/inference_gateway.json)

## Prometheus Alerts

The section instructs how to configure prometheus alerts using collected metrics.

### Configure alerts

You can follow this [blog post](https://grafana.com/blog/2020/02/25/step-by-step-guide-to-setting-up-prometheus-alertmanager-with-slack-pagerduty-and-gmail/) for instruction of setting up alerts in your monitoring stacks with Prometheus.

A template alert rule is available at [alert.yaml](../../tools/alerts/alert.yaml). You can modify and append these rules to your existing Prometheus deployment.

#### High Inference Request Latency P99

```yaml
alert: HighInferenceRequestLatencyP99
expr: histogram_quantile(0.99, rate(inference_objective_request_duration_seconds_bucket[5m])) > 10.0 # Adjust threshold as needed (e.g., 10.0 seconds)
for: 5m
annotations:
  title: 'High latency (P99) for model {% raw %}{{ $labels.model_name }}{% endraw %}'
  description: 'The 99th percentile request duration for model {% raw %}{{ $labels.model_name }}{% endraw %} and target model {% raw %}{{ $labels.target_model_name }}{% endraw %} has been consistently above 10.0 seconds for 5 minutes.'
labels:
  severity: 'warning'
```

#### High Inference Error Rate

```yaml
alert: HighInferenceErrorRate
expr: sum by (model_name) (rate(inference_objective_request_error_total[5m])) / sum by (model_name) (rate(inference_objective_request_total[5m])) > 0.05 # Adjust threshold as needed (e.g., 5% error rate)
for: 5m
annotations:
  title: 'High error rate for model {% raw %}{{ $labels.model_name }}{% endraw %}'
  description: 'The error rate for model {% raw %}{{ $labels.model_name }}{% endraw %} and target model {% raw %}{{ $labels.target_model_name }}{% endraw %} has been consistently above 5% for 5 minutes.'
labels:
  severity: 'critical'
  impact: 'availability'
```

#### High Inference Pool Queue Average Size

```yaml
alert: HighInferencePoolAvgQueueSize
expr: inference_pool_average_queue_size > 50 # Adjust threshold based on expected queue size
for: 5m
annotations:
  title: 'High average queue size for inference pool {% raw %}{{ $labels.name }}{% endraw %}'
  description: 'The average number of requests pending in the queue for inference pool {% raw %}{{ $labels.name }}{% endraw %} has been consistently above 50 for 5 minutes.'
labels:
  severity: 'critical'
  impact: 'performance'
```

#### High Inference Pool Average KV Cache

```yaml
alert: HighInferencePoolAvgKVCacheUtilization
expr: inference_pool_average_kv_cache_utilization > 0.9 # 90% utilization
for: 5m
annotations:
  title: 'High KV cache utilization for inference pool {% raw %}{{ $labels.name }}{% endraw %}'
  description: 'The average KV cache utilization for inference pool {% raw %}{{ $labels.name }}{% endraw %} has been consistently above 90% for 5 minutes, indicating potential resource exhaustion.'
labels:
  severity: 'critical'
  impact: 'resource_exhaustion'
```
