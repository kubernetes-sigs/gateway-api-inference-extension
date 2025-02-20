# Endpoint Picker Protocol

The Endpoint Picker, or EPP, is a core component of the inference extension. Ultimately it's
responsible for picking an endpoint from the `InferencePool`. A reference implementation can be
found [here](../../../pkg/epp/).

## Proxy Protocol

This is the protocol between the EPP and the proxy (e.g, Envoy).

The EPP MUST implement the Envoy
[external processing service](https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor)protocol.

For each HTTP request, the EPP MUST communicate to the proxy the picked model server endpoint via:

1. Setting the `x-gateway-destination-endpoint` HTTP header to the selected endpoint in <ip:port> format.

2. Set an unstructured entry in the [dynamic_metadata](https://github.com/envoyproxy/go-control-plane/blob/c19bf63a811c90bf9e02f8e0dc1dcef94931ebb4/envoy/service/ext_proc/v3/external_processor.pb.go#L320) field of the ext-proc response. The metadata entry for the picked endpoint MUST be wrapped with an outer key (which represents the metadata namespace) with a default of `envoy.lb`.

The final metadata necessary would look like: 
```go
dynamicMetadata: {
  "envoy.lb": {
    "x-gateway-destination-endpoint": <ip:port>"  
  }
}


Note:
- If the EPP did not communicate the server endpoint via these two methods, it MUST return an error.
- The EPP MUST not set two different values in the header and the inner response metadata value. 

### Why envoy.lb namespace as a default? 
The `envoy.lb` namesapce is a predefined namespace used for subsetting. One common way to use the selected endpoint returned from the server, is [envoy subsets](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/subsets) where host metadata for subset load balancing must be placed under `envoy.lb`.

Setting different value leads to unpredictable behavior because proxies aren't guaranteed to support both paths, and so this protocol does not define what takes precedence.

## Model Server Protocol

This is the protocol between the EPP and the model servers.

### Inference API Protocol

The model server MUST implement OpenAI’s [Completions](https://platform.openai.com/docs/api-reference/completions)
and [Chat](https://platform.openai.com/docs/api-reference/chat) APIs.

### Metrics Reporting

The inference extension scrapes metrics from the model servers to make optimal request scheduling
decisions. The model servers MUST provide the following metrics via a Prometheus endpoint. The exact
metric names don't necessarily need to be the same as the recommended names here, however the
metric types and semantics MUST follow this doc.

Note the requirements here are aligned with the
[model server metrics standardization](https://docs.google.com/document/d/1SpSp1E6moa4HSrJnS4x3NpLuj88sMXr2tbofKlzTZpk)
effort.

The corresponding metrics in vLLM are also shown in the table below, as vLLM is already integrated
into the reference endpoint picker implementation.

| Metric | Type | Description | vLLM metric |
| ----- | ---- | ---- | ---- |
| TotalQueuedRequests         | Gauge     | The current total number of requests in the queue.| `vllm:num_requests_waiting`|
| KVCacheUtilization| Gauge     | The current KV cache utilization in percentage.| `vllm:gpu_cache_usage_perc`|


### LoRA Adapter Serving

Model servers that support dynamic LoRA serving can benefit from the LoRA affinity algorithm. Note
the current algorithm in the reference EPP is highly biased towards vLLM's current dynamic LoRA 
implementation.

The model servers MUST support serving a LoRA adapter specified in the `model` argument of the
request, provided the requested adapter is valid.

The model server MUST expose the following LoRA adapter metrics via the same Prometheus endpoint:

* Metric name implemented in vLLM: `vllm:lora_requests_info` 
* Metric type: Gauge
* Metric value: The last updated timestamp (so the EPP can find the latest).
* Metric labels: 
  * `max_lora`: The maximum number of adapters that can be loaded to GPU memory to serve a batch.
  Requests will be queued if the model server has reached MaxActiveAdapter and canno load the
  requested adapter. Example: `"max_lora": "8"`.
  * `running_lora_adapters`: A comma separated list of adapters that are currently loaded in GPU
    memory and ready to serve requests. Example: `"running_lora_adapters": "adapter1, adapter2"`
