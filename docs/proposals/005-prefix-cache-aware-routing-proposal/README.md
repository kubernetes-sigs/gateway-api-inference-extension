# Prefix Cache Aware Routing

## Background

Prefix caching is a well-known technique in LLM inference to save duplicate tensor computation for prompts with the same prefixes, and is available in many model servers, as well as inference frameworks.

[vLLM](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) has the automatic prefix cache (APC) feature by caching in the accelerator HBM, and uses an LRU cache eviction strategy.

[vLLM production stack](https://github.com/vllm-project/production-stack/issues/59) is exploring a prefix aware router to exploit the APC feature of the vLLM. The WIP [PR](https://github.com/vllm-project/production-stack/issues/59#issuecomment-2677268482) implements two strategies: a HashTrie based matching and a SimHash based consistent hashing. The HashTrie solution is showing better cache hit rate.

[SGLang](https://github.com/sgl-project/sglang/blob/4d2a88bdffe91168dfc73ef7e3bc9100ba96686b/sgl-router/src/router.rs#L61) has a cache aware routing strategy which builds a radix tree based on request history.

[AIBrix](https://aibrix.readthedocs.io/latest/features/distributed-kv-cache.html) uses a distributed prefix cache pool and has a customized vLLM to support loading cache from the pool. At request routing, it has a [Prefix Router](https://github.com/vllm-project/aibrix/blob/6feec99d77c84e371da9c535054c2b8aa8912704/pkg/plugins/gateway/algorithms/prefix_cache.go#L64) that maximizes prefix cache hit on model server HBM. It currently implements a hash based (similar to vLLM) and radix tree based (similar to SGLang) matching strategy.

[KubeAI](https://www.kubeai.org/blog/2025/02/26/llm-load-balancing-at-scale-chwbl/) uses a Consistent Hashing with Bounded Loads (CHWBL)  algorithm which hashes request prefixes up to a configurable length (and therefore will lose some accuracy), and use an "overflow" strategy when the server is hot loaded.

## Goals

Implement a prefix aware routing algorithm on EPP to maximize the cache hit rate on the model servers.

### Non-goals

* Change how model server manages prefix caches, e.g., add DRAM cache support or remote cache support.

## Design Options

1. **Session affinity**

Session affinity is based on client attributes such as IP address. It works well for use cases such as multi-turn conversations, where requests from the same client tend to share the same prefixes. This, of course, highly depends on the nature of the use case.

Pros:

* Easy to implement/understand

Cons:

* Limited use case
* Does not exploit prefix cache between different clients
* Using client IP isn't always reliable, will likely need client to provide "session info" for good affinity

1. **Prefix affinity consistent hashing**

This goes a step beyond the session affinity by using a prefix aware hash function to route requests with similar prefixes to the same or similar servers. A naive hash function can be just taking the hash of the first N characters/tokens of the request, and therefore all requests with the same first N characters/tokens will be routed to the same server. The [vLLM production stack](https://github.com/vllm-project/production-stack/issues/59) is exploring this strategy using simhash, and preliminary experiments showed mixed results. KubeAI uses a simple strategy to only hash request prefix up to a configurable `prefixCharLength`. Its effectiveness is likely highly dependent on the input length distribution.

Pros:

* (Compared to session affinity) Is aware of prefix and not limited to per-client affinity
* Small memory overhead (just need to store the ring of the servers)

Cons:

* Highly depends on the effectiveness of the prefix aware hash function.
* Consistent hashing can be challenging to reason about.
 
1. **Approximate prefix cache on the router**
This builds on the intuition that if `requestA=prefix+XX` was routed to server 1, then routing `requstB=prefix+YY` to the same server will likely hit its prefix cache. Therefore the central router can build an approximate lookup cache of the prefix caches on all the backend servers, by mimicking a similar cache eviction strategy of the model server (e.g., LRU). 

Pros:

* Easy to explain (compared to hashing) and likely more effective than hashing strategy.

Cons:

* Relies on knowledge of the cache eviction strategy of the model server, and may need careful tuning for different environments (e.g., model server with different total kv cache space may have different characteristics of cache eviction).
* Complexity in managing cache state (eviction, memory limit)
* An in memory cache is preferred for high performance. However, that means cache need to be rebuilt for restarts. Moreover, cache hit performance decreases with multiple active EPP replicas
 
1. **Accurate prefix cache on the router**
If the router knows what prefixes are currently cached on each model server replica, it can make the optimal decision. A potential solution is to have the model server (or with a sidecar) report the kv cache indexes to the router.

Pros:

* Best cache hit rate

Cons:

* Requires adding a sidecar and its integration with the model server
* May face scalability concerns with large number of model server replicas and large number of prefix caches


## How does prefix cache affinity routing work with LoRA affinity and load-aware routing

1. Prefix cache needs to be LoRA aware, as different adapters don’t share the same kv cache. Therefore, prefix cache affinity algo should be applied after LoRA affinity, to avoid conflicts. And by doing so, the requests will naturally be colocated by the same LoRA, and further by prefix matching.
1. Only use prefix affinity if the expected gain is high enough. This can be done with a threshold, either on the prefix matching length, or a matching ratio of the entire request, or a combination of both. If the expected gain is below the threshold (e.g, only 10 tokens match), then we ignore prefix affinity and fall back to current load based algo.
1. Prefix affinity needs to be aware of the server load, otherwise we will create hot spots. We can use queue length and k-v cache utilization to understand the server load. This is similar to the [queue depth threshold](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/2a615e981228aa6ffc2a89219c986ac863dde776/pkg/epp/scheduling/scheduler.go#L40) for LoRA affinity.


## Proposal 

Implement an approximate prefix cache lookup on the  EPP.

A request is broken down into N chunks of the same number of characters (we don’t necessarily need to tokenize). For each chunk we will calculate a hash based on the **content of the chunk + hash of the prefix**: `hash(chunk i) = hash(chunk i content + hash(chunk i-1))`. This is very similar to how vLLM does it.

When we route a request `r1` with `N` chunks to a server `s1`, we update the approximate cache lookup table like so:

```
hash(chunk 1): append s1
hash(chunk 2): append s1
…
hash(chunk N): append s1
```

This means all these N chunks are cached on server `s1`.

When the EPP receives a new request `r2`, we calculate its chunk hashes, and look up the table to find a server with longest prefix matching.

Each entry in the table needs a `lastUpdate` time to allow LRU cache eviction.
