# Prefix Cache Aware Plugin Configuration

The [prefix cache plugin](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/7617439188b410670ed0f1ff805a3b7f9918a75b/pkg/epp/scheduling/framework/plugins/multi/prefix/plugin.go#L63)
takes advantage of the prefix caching (e.g., [vllm APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html))
feature of model servers, and optimizes request scheduling by placing requests sharing the longest
prefixes to the same server as much as possible, while balancing the server load by considering kv-cache
and queue depth.

## Enable the prefix cache plugin

Currently prefix cache aware plugin is implemented in the V2 scheduler as an experimental feature.
To enable it, set the following environment variables when starting the EndpointPicker(EPP).

```
EXPERIMENTAL_USE_SCHEDULER_V2: true
ENABLE_PREFIX_CACHE_SCHEDULING: true
```

See the [Use Helm section](#helm) to install an inferencepool with the environment variables.


## Customize the prefix cache plugin

The prefix cache plugin exposes the following advanced configuration options via environment variables:

* `PREFIX_CACHE_HASH_BLOCK_SIZE`: The plugin matches prefixes in the unit of blocks. This is the size
of each block in number of bytes. vLLM default block size is 16 tokens. Assume 4 characters per token, the default
is set to 64 in EPP. The default is recommended unless performance is critical for use cases with
extremely long inputs.

* `PREFIX_CACHE_MAX_PREFIX_BLOCKS`: The maximum number of blocks to find prefix match. The default is
128 (or 128*64=8192 characters, or roughly 2048 tokens). This is useful to tradeoff prefix match accuracy
for performance.

* `PREFIX_CACHE_LRU_CAPACITY_PER_SERVER`: Maximum capacity the prefix LRU cache in number of block hashes per server (pod). Below
shows a detailed analysis on how to estimate this.



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

See the [Use Helm section](#helm) to install an inferencepool with the environment variables.


<a id="helm"></a>
## Use Helm

Use the following reference command to install an inferencepool with the prefix
cache plugin environment variable configurations:

```txt
$ helm install vllm-llama3-8b-instruct \
  --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
  --set inferencePool.modelServerType=vllm \
  --set provider.name=[none|gke] \
  --set inferenceExtension.env.EXPERIMENTAL_USE_SCHEDULER_V2=true \
  --set inferenceExtension.env.ENABLE_PREFIX_CACHE_SCHEDULING=true \
  --set inferenceExtension.env.PREFIX_CACHE_LRU_CAPACITY_PER_SERVER=31250 \
  --set inferenceExtension.env.PREFIX_CACHE_MAX_PREFIX_BLOCKS=1024 \
  oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool --version v0
```
