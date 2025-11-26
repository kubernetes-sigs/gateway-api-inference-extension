# Model Server Protocol

This is the protocol between the EPP and the model servers.

## Proposal status
***Partially implemented***

Notes
- With the creation of the [pluggable architecture](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/0683-epp-architecture-proposal) this protocol can, by definition, not be as strict

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


| Metric | Type | Description | vLLM metric | Triton TensorRT-LLM| SGLang |
| ----- | ---- | ------------ | ---- | ---- | ---- |
| TotalQueuedRequests         | Gauge     | The current total number of requests in the queue.| `vllm:num_requests_waiting`| `nv_trt_llm_request_metrics{request_type=waiting}`| `sglang:num_queue_reqs`
| TotalRunningRequests         | Gauge     | The current total number of requests actively being served on the model server.| `vllm:num_requests_running`| `nv_trt_llm_request_metrics{request_type=scheduled}`| `sglang:num_running_reqs`
| KVCacheUtilization| Gauge     | The current KV cache utilization in percentage.| `vllm:kv_cache_usage_perc`| `nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type=fraction}`| `sglang:token_usage`
| [Optional] BlockSize         | Labeled     | The block size in tokens to allocate memory, used by the prefix cache scorer. If this metric is not available, the BlockSize will be derived from the [prefix plugin config](https://gateway-api-inference-extension.sigs.k8s.io/guides/epp-configuration/prefix-aware/#customize-the-prefix-cache-plugin).| name: `vllm:cache_config_info`, label name: `block_size`| | 
| [Optional] NumGPUBlocks| Labeled     | The total number of blocks in the HBM KV cache, used by the prefix cache scorer. If this metric is not available, the NumGPUBlocks will be derived from the [prefix plugin config](https://gateway-api-inference-extension.sigs.k8s.io/guides/epp-configuration/prefix-aware/#customize-the-prefix-cache-plugin).| name: `vllm:cache_config_info`, label name: `num_gpu_blocks`| | 


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
  Requests will be queued if the model server has reached MaxActiveAdapter and cannot load the
  requested adapter. Example: `"max_lora": "8"`.
  * `running_lora_adapters`: A comma separated list of adapters that are currently loaded in GPU
    memory and ready to serve requests. Example: `"running_lora_adapters": "adapter1, adapter2"`
  * `waiting_lora_adapters`: A comma separated list of adapters that are waiting to be served. Example: `"waiting_lora_adapters": "adapter1, adapter2"`

### Prefix Cache Reuse

Starting from [v0.4.0](https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/tag/v0.4.0),
the EPP supports [prefix cache optimized request scheduling](https://gateway-api-inference-extension.sigs.k8s.io/guides/epp-configuration/prefix-aware/).
To benefit from the optimal prefix aware request scheduling, model servers SHOULD support prefix
cache reuse, such as the [vllm automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) feature.
