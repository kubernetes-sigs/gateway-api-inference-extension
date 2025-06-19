# Metrics

This guide describes the current state of exposed metrics and how to scrape them.

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
      "stream_options": {"include_usage": "true"}
      }'
      ```

=== "Dynamic LoRA Adapter Sidecar"

      To have response metrics, ensure the vLLM model server is configured with the dynamic LoRA adapter as a sidecar container and a ConfigMap to configure which models to load/unload. See [this doc](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/tools/dynamic-lora-sidecar#example-configuration) for an example.


## Exposed metrics

### EPP

| **Metric name**                              | **Metric Type**  | <div style="width:200px">**Description**</div>  | <div style="width:250px">**Labels**</div>                                          | **Status**  |
|:---------------------------------------------|:-----------------|:------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:------------|
| inference_model_request_total                | Counter          | The counter of requests broken out for each model.                | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt; | ALPHA       |
| inference_model_request_error_total          | Counter          | The counter of requests errors broken out for each model.         | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt; | ALPHA       |
| inference_model_request_duration_seconds     | Distribution     | Distribution of response latency.                                 | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt; | ALPHA       |
| normalized_time_per_output_token_seconds     | Distribution     | Distribution of ntpot (response latency per output token)                                 | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt; | ALPHA       |
| inference_model_request_sizes                | Distribution     | Distribution of request size in bytes.                            | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt; | ALPHA       |
| inference_model_response_sizes               | Distribution     | Distribution of response size in bytes.                           | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt; | ALPHA       |
| inference_model_input_tokens                 | Distribution     | Distribution of input token count.                                | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt; | ALPHA       |
| inference_model_output_tokens                | Distribution     | Distribution of output token count.                               | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt; | ALPHA       |
| inference_model_running_requests                | Gauge     | Number of running requests for each model.             | `model_name`=&lt;model-name&gt;  | ALPHA       |
| inference_pool_average_kv_cache_utilization  | Gauge            | The average kv cache utilization for an inference server pool.    | `name`=&lt;inference-pool-name&gt;                                                 | ALPHA       |
| inference_pool_average_queue_size            | Gauge            | The average number of requests pending in the model server queue. | `name`=&lt;inference-pool-name&gt;                                                 | ALPHA       |
| inference_pool_per_pod_queue_size            | Gauge            | The total number of queue for each model server pod under the inference pool         | `model_server_pod`=&lt;model-server-pod-name&gt; <br> `name`=&lt;inference-pool-name&gt;                             | ALPHA       |
| inference_pool_ready_pods                    | Gauge            | The number of ready pods for an inference server pool.            | `name`=&lt;inference-pool-name&gt;                                                 | ALPHA       |
| inference_extension_info                     | Gauge            | The general information of the current build.                     | `commit`=&lt;hash-of-the-build&gt; <br> `build_ref`=&lt;ref-to-the-build&gt;        | ALPHA       |

### Dynamic LoRA Adapter Sidecar

| **Metric name**            | **Metric Type**  | <div style="width:200px">**Description**</div>   | <div style="width:250px">**Labels**</div> | **Status**  |
|:---------------------------|:-----------------|:-------------------------------------------------|:------------------------------------------|:------------|
| lora_syncer_adapter_status | Gauge            | Status of LoRA adapters (1=loaded, 0=not_loaded) | `adapter_name`=&lt;adapter-id&gt;         | ALPHA       |

## Scrape Metrics

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
  apiGroup: rbac.authorization.k8s.io
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

## Prometheus Alerts

The section instructs how to configure prometheus alerts using collected metrics.

### Configure alerts

You can follow this [blog post](https://grafana.com/blog/2020/02/25/step-by-step-guide-to-setting-up-prometheus-alertmanager-with-slack-pagerduty-and-gmail/) for instruction of setting up alerts in your monitoring stacks with Prometheus.

A template alert rule is available at [alert.yaml](../../tools/alerts/alert.yaml). You can modify and append these rules to your existing Prometheus deployment.

#### High Inference Request Latency P99

```yaml
alert: HighInferenceRequestLatencyP99
expr: histogram_quantile(0.99, rate(inference_model_request_duration_seconds_bucket[5m])) > 10.0 # Adjust threshold as needed (e.g., 10.0 seconds)
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
expr: sum by (model_name) (rate(inference_model_request_error_total[5m])) / sum by (model_name) (rate(inference_model_request_total[5m])) > 0.05 # Adjust threshold as needed (e.g., 5% error rate)
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
