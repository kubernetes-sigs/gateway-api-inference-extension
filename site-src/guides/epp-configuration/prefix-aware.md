# Prefix Cache Aware Plugin Configuration

The [prefix cache plugin](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/7617439188b410670ed0f1ff805a3b7f9918a75b/pkg/epp/scheduling/framework/plugins/multi/prefix/plugin.go#L63)
takes advantage of the prefix caching (e.g., [vllm APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html))
feature of model servers, and optimizes request scheduling by placing requests sharing the longest
prefixes to the same server as much as possible, while balancing the server load by considering kv-cache
and queue depth.

## Enable the prefix cache plugin

Like any other plugins, the prefix cache aware plugin can be enabled/disabled via the [plugin config file](config-text.md), and is enabled in the [default configuration](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/config/charts/inferencepool/templates/epp-config.yaml).

## Customize the prefix cache plugin

The prefix cache plugin exposes the following advanced configuration parameters:

* `blockSize`: The plugin matches prefixes in the unit of blocks. This is the size
of each block in number of bytes. At runtime, EPP can dynamically fetch this information from the
inference engine metrics, therefore this config is only used when such metric is not available. In
vLLM, the metric name is `vllm:cache_config_info` and the metric label is `block_size`. See the
[model server protocol](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/003-model-server-protocol)
for more details.

    vLLM default block size is 16 tokens. Assume 4 characters per token, the default
    is set to 64 in EPP. The default is recommended unless performance is critical for use cases with
    extremely long inputs.

* `maxPrefixBlocksToMatch`: The maximum number of blocks to find prefix match. The default is
256 (or 256*64=16384 characters, or roughly 4096 tokens). This is useful to tradeoff prefix match accuracy
for performance.

* `lruCapacityPerServer`: Maximum capacity the prefix LRU cache in number of block hashes per server (pod). 
Similar to `blockSize`, EPP can dynamically fetch this from the inference engine metrics endpoints. 
In vLLM, the metric name is `vllm:cache_config_info` and the metric label is `num_gpu_blocks`. See the
[model server protocol](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/003-model-server-protocol)
for more details.

    If such metric is not available, you can follow the guide below on how to estimate this.

        The prefix cache plugin estimates the prefix cache indexes in model server HBMs.  In the perfect
        scenario, EPP has the exact same prefix cache entries per model server as their HBM cache entries. If
        the EPP cache is smaller than HBM cache, a positive EPP cache match is more accurate, but there are more
        false cache misses. If the EPP cache is larger than the HBM cache, then there are more false cache hits.
        Therefore **the EPP prefix cache indexer size should be as close as possible to the HBM cache size.**

        NOTE: EPP builds prefix cache based on characters, while model server maintains prefix cache entries
        in tokens, a conversion between character <-> token is needed.

        Below are the formulas to estimate the EPP prefix indexer size:

        ```
        max_kv_tokens_per_server = (HBM_size - model_size)/ kv_size_per_token
        lru_indexer_capacity_per_server = (max_kv_tokens_per_server * avg_chars_per_token)/prefix_indexer_hash_block_size
        ```

        Let's take an example:

        * Model: llama3 8B
        * Accelerator: Nvidia H100 80GB
        * Num replicas: 3
        * Estimated # characters per token: 4 ([source](https://genai.stackexchange.com/questions/34/how-long-is-a-token))

        ```
        max_kv_tokens_per_server = (80GB - 16GB) / 128KB = 500,000
        # assume avg_chars_per_token = 4, prefix_indexer_hash_block_size = 64 (default)
        # each entry is about 358KB, so the memory footrpint is abut 11 MB per server
        lru_indexer_capacity_per_server = 500,000*4/64 = 31250
        ```
