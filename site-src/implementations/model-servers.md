

# Supported Model Servers

Any model server that conform to the [model server protocol](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/003-model-server-protocol) are supported by the inference extension.

## Compatible Model Server Versions

| Model Server         | Version                                                                                                                | Commit                                                                                                                            | Notes                                                                                                       |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| vLLM V0              | v0.6.4 and above                                                                                                       | [commit 0ad216f](https://github.com/vllm-project/vllm/commit/0ad216f5750742115c686723bf38698372d483fd)                            |                                                                                                             |
| vLLM V1              | v0.8.0 and above                                                                                                       | [commit bc32bc7](https://github.com/vllm-project/vllm/commit/bc32bc73aad076849ac88565cff745b01b17d89c)                            |                                                                                                             |
| Triton(TensorRT-LLM) | [25.03](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-25-03.html#rel-25-03) and above | [commit 15cb989](https://github.com/triton-inference-server/tensorrtllm_backend/commit/15cb989b00523d8e92dce5165b9b9846c047a70d). | LoRA affinity feature is not available as the required LoRA metrics haven't been implemented in Triton yet. |

## vLLM

vLLM is configured as the default in the [endpoint picker extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/pkg/epp). No further configuration is required.

## Triton with TensorRT-LLM Backend

Triton specific metric names need to be specified when starting the EPP.

### Option 1: Use Helm

Use `--set inferencePool.modelServerType=triton-tensorrt-llm` to install the [`inferencepool` via helm](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/42eb5ff1c5af1275df43ac384df0ddf20da95134/config/charts/inferencepool). See the [`inferencepool` helm guide](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/42eb5ff1c5af1275df43ac384df0ddf20da95134/config/charts/inferencepool/README.md) for more details.

### Option 2: Edit EPP deployment yaml

 Add the following to the `args` of the [EPP deployment](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/42eb5ff1c5af1275df43ac384df0ddf20da95134/config/manifests/inferencepool-resources.yaml#L32)
 
 ```
- -totalQueuedRequestsMetric
- "nv_trt_llm_request_metrics{request_type=waiting}"
- -kvCacheUsagePercentageMetric
- "nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type=fraction}"
- -loraInfoMetric
- "" # Set an empty metric to disable LoRA metric scraping as they are not supported by Triton yet.
```