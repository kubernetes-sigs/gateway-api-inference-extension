

# Supported Model Servers

Any model server that conform to the [model server protocol](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/003-model-server-protocol) are supported by the inference extension.

## Compatible Model Server Versions

|Model Server| Version| Commit| Notes|
|--|--|--|----|
|vLLM V0|v0.6.4 and above| [commit 0ad216f](https://github.com/vllm-project/vllm/commit/0ad216f5750742115c686723bf38698372d483fd)| |
|vLLM V1|v0.8.0 and above| [commit bc32bc7](https://github.com/vllm-project/vllm/commit/bc32bc73aad076849ac88565cff745b01b17d89c)| |
Triton(TensorRT-LLM)| TODO| Pending [PR](https://github.com/triton-inference-server/tensorrtllm_backend/pull/725). |LoRA affinity feature is not available as the required LoRA metrics haven't been implemented in Triton yet.|

## vLLM

vLLM is configured as the default in the [endpoint picker extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/pkg/epp). No further configuration is required.

## Use Triton with TensorRT-LLM Backend

You need to specify the metric names when starting the EPP container. Add the following to the `args` of the [EPP deployment](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/296247b07feed430458b8e0e3f496055a88f5e89/config/manifests/inferencepool.yaml#L48).
```
- -totalQueuedRequestsMetric
- "nv_trt_llm_request_metrics{request_type=waiting}"
- -kvCacheUsagePercentageMetric
- "nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type=fraction}"
```