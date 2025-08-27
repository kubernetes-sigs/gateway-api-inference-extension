# Prefix Cache Aware Request Scheduling

## Proposal Status
***Implemented***

# Proposal

## Overview

Prefix caching is a well-known technique in LLM inference to save duplicate tensor computation for prompts with the same prefix tokens, and is available in many model servers or model as a service providers. Leveraging prefix caching can significantly boost system performance, especially the time to first token (TTFT). Given that EPP has a global view of requests and model servers in the `InferencePool`, it can schedule requests intelligently to maximize the global prefix cache hit rate.

### Goals

Implement a prefix aware scheduling algorithm on EPP to maximize the cache hit rate on the model servers.

### Non-goals

* Change how model server manages prefix caches, or add any prefix cache APIs.
* Coordinate cache beyond accelerator HBM cache, such as remote caches.

## Terminology

In the gateway-api-inference-extension project, we use the term "request scheduling" to mean the process of estimating the cost of a request and placing it to the best backend server. This is different from "model routing" which oftentimes means picking the right model server endpoint based on cost, availability, etc. However, we acknowledge that various other projects uses the term "routing" or "router" to mean what we call "request scheduling". In this doc, we use "scheduling" when referring to the inference extension, and "routing" or "router" when referring to other projects, respecting the terminology of those projects.

## Existing Solutions

[vLLM](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) has the automatic prefix cache (APC) feature by caching in the accelerator HBM, and uses an LRU cache eviction strategy.

[vLLM production stack](https://github.com/vllm-project/production-stack/issues/59) is exploring a prefix aware router to exploit the APC feature of the vLLM. The WIP [PR](https://github.com/vllm-project/production-stack/issues/59#issuecomment-2677268482) implements two strategies: a HashTrie based matching and a SimHash based consistent hashing. The HashTrie solution is showing better cache hit rate.

[SGLang](https://github.com/sgl-project/sglang/blob/4d2a88bdffe91168dfc73ef7e3bc9100ba96686b/sgl-router/src/router.rs#L61) has a cache aware routing strategy which builds a radix tree based on request history.

[AIBrix](https://aibrix.readthedocs.io/latest/features/distributed-kv-cache.html) uses a distributed prefix cache pool and has a customized vLLM to support loading cache from the pool. At request routing, it has a [Prefix Router](https://github.com/vllm-project/aibrix/blob/6feec99d77c84e371da9c535054c2b8aa8912704/pkg/plugins/gateway/algorithms/prefix_cache.go#L64) that maximizes prefix cache hit on model server HBM. It currently implements a hash based (similar to vLLM) and radix tree based (similar to SGLang) matching strategy.

[KubeAI](https://www.kubeai.org/blog/2025/02/26/llm-load-balancing-at-scale-chwbl/) uses a Consistent Hashing with Bounded Loads (CHWBL)  algorithm which hashes request prefixes up to a configurable length (and therefore will lose some accuracy), and use an "overflow" strategy when the server is hot loaded.

## Design Options

### Session affinity

Session affinity is based on client attributes such as IP address. It works well for use cases such as multi-turn conversations, where requests from the same client tend to share the same prefixes. This, of course, highly depends on the nature of the use case.

Pros:

* Easy to implement/understand

Cons:

* Limited use case
* Does not exploit prefix cache between different clients
* Using client IP isn't always reliable, will likely need client to provide "session info" for good affinity

### Prefix affinity consistent hashing

This goes a step beyond the session affinity by using a prefix aware hash function to schedule requests with similar prefixes to the same or similar servers. A naive hash function can be just taking the hash of the first N characters/tokens of the request, and therefore all requests with the same first N characters/tokens will be scheduled to the same server. The [vLLM production stack](https://github.com/vllm-project/production-stack/issues/59) is exploring this strategy using simhash, and preliminary experiments showed mixed results. KubeAI uses a simple strategy to only hash request prefix up to a configurable `prefixCharLength`. Its effectiveness is likely highly dependent on the input length distribution.

Pros:

* (Compared to session affinity) Is aware of prefix and not limited to per-client affinity
* Small memory overhead (just need to store the ring of the servers)

Cons:

* Highly depends on the effectiveness of the prefix aware hash function.
* Consistent hashing can be challenging to reason about.
 
### Report prefix cache indexes on the EPP

If the EPP knows what prefixes are currently cached on each model server replica, it can make the optimal decision. A potential solution is to have the model server (or with a sidecar) report the kv cache indexes to the EPP.

Pros:

* Best cache hit rate in theory

Cons:

* Requires API changes on the model servers to report the cache indexes.
* Reporting the cache indexes in real time requires non-trivial network bandwidth.

### Approximate prefix index on the EPP

This builds on the intuition that if `requestA=prefix+XX` was scheduled to server 1, then scheduling `requestB=prefix+YY` to the same server will likely hit its prefix cache. Therefore the EPP can build an approximate index table of the prefix caches on all the backend servers, by mimicking a similar cache eviction strategy of the model server (e.g., LRU). 

Pros:

* (Compared to the session affinity strategy) Broader application to most use cases and doesn't require any client integration.
* (Compared to the consistent hashing strategy) Easy to implement and explain and is more effective.

Cons:

* Relies on knowledge of the cache eviction strategy of the model server, and may need careful tuning for different environments (e.g., model server with different total kv cache space may have different characteristics of cache eviction).
* Complexity in managing cache state (eviction, memory limit)
* An in memory cache is preferred for high performance. However, that means cache need to be rebuilt for restarts. Moreover, cache hit performance decreases with multiple active EPP replicas.

## Proposal 

Based on the above discussion, I propose implementing "Approximate prefix cache on the EPP" solution, which has the advantage of fast time to market, automatic prefix cache (without needing client integration), decent performance with the cost of degraded performance when sharded. 

A request is broken down into N chunks of the same number of characters (we don’t necessarily need to tokenize). For each chunk we will calculate a hash based on the **content of the chunk + hash of the prefix**: `hash(chunk i) = hash(chunk i content + hash(chunk i-1))`. This gives us a nice property that if we find a match of a chunk hash, then we know all its prefix chunk hashes match as well. This is very similar to how vLLM does it.

When we schedule a request `r1` with `N` chunks to a server `s1`, we update the approximate cache index table like so:

```
hash(chunk 1): append s1
hash(chunk 2): append s1
…
hash(chunk N): append s1
```

This means all these N chunks are cached on server `s1`.

When the EPP receives a new request `r2`, we calculate its chunk hashes, and look up the table to find a server with longest prefix matching.

<img src="https://docs.google.com/drawings/d/e/2PACX-1vQ9gGbq_vrv46BZpviOUpKCuo_WCo6ANzLoAIP9lo6zrMB9kmVNk4YLKBAoGh3IsZ7mRxDu9pDqukrX/pub?w=1074&amp;h=956">

[Image source](https://docs.google.com/drawings/d/1KL5DKh42Z_XzvcnejUcRymu99_HwW9y8U29IrPzRCss/edit?usp=sharing)


## How does prefix cache affinity work with LoRA affinity and load-aware scheduling

1. Prefix cache needs to be LoRA aware, as different adapters don’t share the same kv cache. Therefore when finding prefix matches, we only match for the same model/adapter.
2. Prefix affinity needs to be aware of the server load and avoid overloading servers. We can calculate a combined weighted score of servers depending on: prefix cache hit ratio,  queue length and k-v cache utilization to achieve a good balance between prefix cache affinity and load balancing. 

## Future work

The main drawback of the proposed solution is the degraded performance when EPP is sharded, as the in memory cache index table loses a global view of all requests. To mitigate this issue, we can consider:

* Establish a "prefix cache index reporting" protocol with model servers, and use a combination of the approximate cache index with reported indexes. This can potentially work better than a solution purely based on reported indexes, as discussed in [`Solution 3`](https://github.com/kubernetes-sigs/gateway-api-inference-extension/discussions/678).
* When scheduling a request with low or no prefix cache in the EPP in memory index table, use the consistent hashing strategy to improve the predictability of two EPPs picking the same server, instead of random picking.