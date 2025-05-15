# Metrics

This guide describes the current state of exposed metrics and how to scrape them.

## Requirements

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

## Exposed metrics

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


## Scrape Metrics

Metrics endpoint is exposed at port 9090 by default. To scrape metrics, the client needs a ClusterRole with the following rule:
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
Then, you can curl the 9090 port like following
```
TOKEN=$(kubectl -n default get secret inference-gateway-sa-metrics-reader-secret  -o jsonpath='{.secrets[0].name}' -o jsonpath='{.data.token}' | base64 --decode)

kubectl -n default port-forward inference-gateway-ext-proc-pod-name  9090

curl -H "Authorization: Bearer $TOKEN" localhost:9090/metrics
```