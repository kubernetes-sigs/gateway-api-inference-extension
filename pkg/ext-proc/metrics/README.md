# Documentation

This documentation is the current state of exposed metrics.

## Table of Contents
* [Exposed Metrics](#exposed-metrics)
* [Scrape Metrics](#scrape-metrics)

## Requirements

NOTE: Response metrics are only supported in non-streaming mode, with the follow up [issue](https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/178) to address streaming mode.

Currently you have 2 options:
- If you use response streaming, simply leave the response body processing mode empty in your `EnvoyExtensionPolicy` (default). You won't get response metrics reporting.

- If you don't use streaming, to enable response metrics reporting, you can enable `Buffered` mode for response in `EnvoyExtensionPolicy`.

```
apiVersion: gateway.envoyproxy.io/v1alpha1
kind: EnvoyExtensionPolicy
metadata:
  name: ext-proc-policy
  namespace: default
spec:
  extProc:
    - backendRefs:
      - group: ""
        kind: Service
        name: inference-gateway-ext-proc
        port: 9002
      processingMode:
        request:
          body: Buffered
        response:
          body: Buffered
```

## Exposed metrics

| Metric name | Metric Type  | Description | Labels | Status | 
| ------------|--------------| ----------- | ------ | ------ |
| inference_model_request_total | Counter      | The counter of requests broken out for each model. | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt;  | ALPHA |
| inference_model_request_duration_seconds | Distribution | Distribution of response latency. | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt;  | ALPHA |
| inference_model_request_sizes | Distribution      | Distribution of request size in bytes. | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt;  | ALPHA |
| inference_model_response_sizes | Distribution      | Distribution of response size in bytes. | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt;  | ALPHA |
| inference_model_input_tokens | Distribution      | Distribution of input token count. | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt;  | ALPHA |
| inference_model_output_tokens | Distribution      | Distribution of output token count. | `model_name`=&lt;model-name&gt; <br> `target_model_name`=&lt;target-model-name&gt;  | ALPHA |

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