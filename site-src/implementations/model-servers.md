# Supported Model Servers

Any model server that conform to the [model server protocol](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/003-model-server-protocol) are supported by the inference extension.

## Compatible Model Server Versions

| Model Server         | Version                                                                                                                | Commit                                                                                                                            | Notes                                                                                                       |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| vLLM V0              | v0.6.4 and above                                                                                                       | [commit 0ad216f](https://github.com/vllm-project/vllm/commit/0ad216f5750742115c686723bf38698372d483fd)                            |                                                                                                             |
| vLLM V1              | v0.8.0 and above                                                                                                       | [commit bc32bc7](https://github.com/vllm-project/vllm/commit/bc32bc73aad076849ac88565cff745b01b17d89c)                            |                                                                                                             |
| Triton(TensorRT-LLM) | [25.03](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-25-03.html#rel-25-03) and above | [commit 15cb989](https://github.com/triton-inference-server/tensorrtllm_backend/commit/15cb989b00523d8e92dce5165b9b9846c047a70d). | LoRA affinity feature is not available as the required LoRA metrics haven't been implemented in Triton yet. [Feature request](https://github.com/triton-inference-server/server/issues/8181) |
| SGLang               | v0.4.0 and above | [commit 1929c06](https://github.com/sgl-project/sglang/commit/1929c067625089c9c3c04321578f450275f24041) | Set `--enable-metrics` on the model server. LoRA affinity feature is not available as the required LoRA metrics haven't been implemented in SGLang yet.

## vLLM

vLLM is configured as the default in the [endpoint picker extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/pkg/epp). No further configuration is required.

## Triton with TensorRT-LLM Backend

Triton specific metric names need to be specified when starting the EPP.

Use `--set inferencePool.modelServerType=triton-tensorrt-llm` to install the `inferencepool` via helm. See the [`inferencepool` helm guide](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/config/charts/inferencepool/README.md) for more details.

 Add the following to the `flags` in the helm chart as [flags to EPP](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/29ea29028496a638b162ff287c62c0087211bbe5/config/charts/inferencepool/values.yaml#L36)

```
- name=total-queued-requests-metric
  value="nv_trt_llm_request_metrics{request_type=waiting}"
- name=kv-cache-usage-percentage-metric
  value="nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type=fraction}"
- name=lora-info-metric
  value="" # Set an empty metric to disable LoRA metric scraping as they are not supported by Triton yet.
```

## SGLang

 Add the following `flags` while deploying using helm charts in the [EPP deployment](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/29ea29028496a638b162ff287c62c0087211bbe5/config/charts/inferencepool/values.yaml#L36)


```
- name=total-queued-requests-metric
  value="sglang:num_queue_reqs"
- name=kv-cache-usage-percentage-metric
  value="sglang:token_usage"
- name=lora-info-metric
  value="" # Set an empty metric to disable LoRA metric scraping as they are not supported by SGLang yet.
```

## Multi-Engine Support

The Inference Extension supports collecting metrics from multiple inference engines simultaneously within the same `InferencePool`. This is useful for A/B testing or mixed-engine deployments.

By default, EPP includes pre-configured metric mappings for **vLLM** (default) and **SGLang**. You only need to label your Pods with the engine type.

### 1. Label your Pods

Label each deployment with the engine type:

```yaml
# vLLM Deployment
metadata:
  labels:
    inference.networking.k8s.io/engine-type: vllm

# SGLang Deployment
metadata:
  labels:
    inference.networking.k8s.io/engine-type: sglang
```

Pods without the engine label will use the default engine configuration (vLLM).

### 2. Change Default Engine (Optional)

To use SGLang as the default engine instead of vLLM, simply set the `defaultEngine` parameter:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
featureGates:
- dataLayer
plugins:
- name: model-server-protocol-metrics
  type: model-server-protocol-metrics
  parameters:
    defaultEngine: "sglang"  # Pods without engine label will use SGLang metrics
```

### 3. Custom Engine Configuration (Optional)

If you need to customize the metric mappings or add support for other engines (e.g., Triton), provide engine-specific configurations in your `EndpointPickerConfig`. Note that built-in vLLM and SGLang configs are automatically included, so you only need to define them if you want to override the defaults:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
featureGates:
- dataLayer
plugins:
- name: model-server-protocol-metrics
  type: model-server-protocol-metrics
  parameters:
    engineLabelKey: "inference.networking.k8s.io/engine-type"  # Pod label key (optional, this is the default)
    defaultEngine: "vllm"  # Which engine to use for Pods without engine label
    engineConfigs:
    # vllm and sglang are optional - only define them to override defaults
    - name: vllm
      queuedRequestsSpec: "vllm:num_requests_waiting"
      runningRequestsSpec: "vllm:num_requests_running"
      kvUsageSpec: "vllm:kv_cache_usage_perc"
      loraSpec: "vllm:lora_requests_info"
      cacheInfoSpec: "vllm:cache_config_info"
    - name: sglang
      queuedRequestsSpec: "sglang:num_queue_reqs"
      runningRequestsSpec: "sglang:num_running_reqs"
      kvUsageSpec: "sglang:token_usage"
    - name: triton
      queuedRequestsSpec: "nv_trt_llm_request_metrics{request_type=waiting}"
      kvUsageSpec: "nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type=fraction}"
```

**Key points:**
- Use `engineLabelKey` to customize the Pod label key for engine identification (defaults to `inference.networking.k8s.io/engine-type`)
- Use `defaultEngine` to specify which engine is used for Pods without an engine label (defaults to "vllm")
- Built-in vLLM and SGLang configs are automatically included, even when adding custom engines