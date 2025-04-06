# Gateway API Inference Extension 

This extension upgrades an [ext-proc](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter)-capable proxy or gateway - such as Envoy Gateway, kGateway, or the GKE Gateway - to become an **inference gateway** - supporting inference platform teams self-hosting large language models on Kubernetes. This integration makes it easy to expose and control access to your local [OpenAI-compatible chat completion endpoints](https://platform.openai.com/docs/api-reference/chat) to other workloads on or off cluster, or to integrate your self-hosted models alongside model-as-a-service providers in a higher level **AI Gateway** like LiteLLM, Solo AI Gateway, or Apigee.

The inference gateway:

* Improves the tail latency and throughput of LLM completion requests against Kubernetes-hosted model servers using an extensible request scheduling alogrithm that is kv-cache and request cost aware, avoiding evictions or queueing as load increases
* Provides [Kubernetes-native declarative APIs](https://gateway-api-inference-extension.sigs.k8s.io/concepts/api-overview/) to route client model names to use-case specific LoRA adapters and control incremental rollout of new adapter versions, A/B traffic splitting, and safe blue-green base model and model server upgrades
* Adds end to end observability around service objective attainment
* Ensures operational guardrails between different client model names, allowing a platform team to safely serve many different GenAI workloads on the same pool of shared foundation model servers for higher utilization and fewer required accelerators

![Architecture Diagram](./docs/inference-gateway-architecture.svg)

It currently requires a version of vLLM that supports the necessary metrics to predict traffic load which is defined in the [model server protocol](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/003-model-server-protocol).  Support for Google's Jetstream, nVidia Triton, text-generation-inference, and SGLang is coming soon.

## Status

This project is [alpha (0.3 release)](https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/tag/v0.3.0).  It should not be used in production yet.

## Getting Started

Follow our [Getting Started Guide](https://gateway-api-inference-extension.sigs.k8s.io/guides/) to get the inference-extension up and running on your cluster!

See our website at https://gateway-api-inference-extension.sigs.k8s.io/ for detailed API documentation on leveraging our Kubernetes-native declarative APIs

## Roadmap

As Inference Gateway builds towards a GA release. We will continue to expand our capabilities, namely:
1. Prefix-cache aware load balancing with interfaces for remote caches 
1. Recommended LoRA adapter pipeline for automated rollout 
1. Fairness and priority between workloads within the same criticality band
1. HPA support for autoscaling on aggregate metrics derived from the load balancer
1. Support for large multi-modal inputs and outputs
1. Support for other GenAI model types (diffusion and other non-completion protocols)
1. Heterogeneous accelerators - serve workloads on multiple types of accelerator using latency and request cost-aware load balancing
1. Disaggregated serving support with independently scaling pools


## End-to-End Tests

Follow this [README](./test/e2e/epp/README.md) to learn more about running the inference-extension end-to-end test suite on your cluster.

## Contributing

Our community meeting is weekly at Thursday 10AM PDT ([Zoom](https://zoom.us/j/9955436256?pwd=Z2FQWU1jeDZkVC9RRTN4TlZyZTBHZz09), [Meeting Notes](https://www.google.com/url?q=https://docs.google.com/document/d/1frfPE5L1sI3737rdQV04IcDGeOcGJj2ItjMg6z2SRH0/edit?usp%3Dsharing&sa=D&source=calendar&usd=2&usg=AOvVaw1pUVy7UN_2PMj8qJJcFm1U)).

We currently utilize the [#wg-serving](https://kubernetes.slack.com/?redir=%2Fmessages%2Fwg-serving) slack channel for communications.

Contributions are readily welcomed, follow the [dev guide](./docs/dev.md) to start contributing!

### Code of conduct

Participation in the Kubernetes community is governed by the [Kubernetes Code of Conduct](code-of-conduct.md).
