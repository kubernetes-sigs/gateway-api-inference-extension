# Introduction

Gateway API Inference Extension is an official Kubernetes project that optimizes self-hosting Generative Models on Kubernetes.

The overall resource model focuses on 2 new inference-focused
[personas](/concepts/roles-and-personas) and corresponding resources that
they are expected to manage:

<!-- Source: https://docs.google.com/presentation/d/11HEYCgFi-aya7FS91JvAfllHiIlvfgcp7qpi_Azjk4E/edit#slide=id.g292839eca6d_1_0 -->
<img src="/images/resource-model.png" alt="Gateway API Inference Extension Resource Model" class="center" width="550" />

## Concepts and Definitions

The following specific terms to this project:

- **Inference Gateway**: A proxy/load-balancer that has been coupled with the
  EndPointer Picker extension. It provides optimized routing and load balancing for
  serving Kubernetes self-hosted generative Artificial Intelligence (AI)
  workloads. It simplifies the deployment, management, and observability of AI
  inference workloads.
- **Inference Scheduler**: An extendable component that makes decisions about which endpoint is optimal (best cost /
  best performance) for an inference request based on `Metrics and Capabilities`
  from [Model Serving](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/003-model-server-protocol/README.md).
- **Metrics and Capabilities**: Data provided by model serving platforms about
  performance, availability and capabilities to optimize routing. Includes
  things like [Prefix Cache](https://docs.vllm.ai/en/stable/design/v1/prefix_caching.html) status or [LoRA Adapters](https://docs.vllm.ai/en/stable/features/lora.html) availability.
- **Endpoint Picker(EPP)**: An implementation of an `Inference Scheduler` with additional Routing, Flow, and Request Control layers to allow for sophisticated routing strategies. Additional info on the architecture of the EPP [here](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/0683-epp-architecture-proposal).
- **Body Based Router(BBR)**: An additional (and optional) implementation of an extension that extracts information from the body portion of the inference request, currently the model name attribute from the body of an OpenAI API request, which can then be used by the gateway to perform model-aware functions such as routing/scheduling. This may be used along with the EPP in order to have a combination of model picking and endpoint picking functionality.

[Inference Gateway]:#concepts-and-definitions

## Key Features 
Gateway API Inference Extension optimizes self-hosting Generative AI Models on Kubernetes.
It provides optimized load-balancing for self-hosted Generative AI Models on Kubernetes.
The project’s goal is to improve and standardize routing to inference workloads across the ecosystem.

This is achieved by leveraging Envoy's [External Processing](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter) to extend any gateway that supports both ext-proc and [Gateway API](https://github.com/kubernetes-sigs/gateway-api) into an [inference gateway](#concepts-and-definitions).
This extension extends popular gateways like Envoy Gateway, kgateway, and GKE Gateway - to become [Inference Gateway](#concepts-and-definitions) -
supporting inference platform teams self-hosting Generative Models (with a current focus on large language models) on Kubernetes.
This integration makes it easy to expose and control access to your local [OpenAI-compatible chat completion endpoints](https://platform.openai.com/docs/api-reference/chat)
to other workloads on or off cluster, or to integrate your self-hosted models alongside model-as-a-service providers
in a higher level **AI Gateways** like [LiteLLM](https://www.litellm.ai/), [Gloo AI Gateway](https://www.solo.io/products/gloo-ai-gateway), or [Apigee](https://cloud.google.com/apigee).


- **Model-aware routing**: Instead of simply routing based on the path of the request, an **[inference gateway]** allows you to route to models based on the model names. This is enabled by support for GenAI Inference API specifications (such as OpenAI API) in the gateway implementations such as in Envoy Proxy. This model-aware routing also extends to Low-Rank Adaptation (LoRA) fine-tuned models.

- **Serving priority**: an **[inference gateway]** allows you to specify the serving priority of your models. For example, you can specify that your models for online inference of chat tasks (which is more latency sensitive) have a higher [*Priority*](/reference/spec/#priority) than a model for latency tolerant tasks such as a summarization. 

- **Model rollouts**:  an **[inference gateway]** allows you to incrementally roll out new model versions by traffic splitting definitions based on the model names. 

- **Extensibility for Inference Services**: an **[inference gateway]** defines extensibility pattern for additional Inference services to create bespoke routing capabilities should out of the box solutions not fit your needs.

- **Customizable Load Balancing for Inference**: an **[inference gateway]** defines a pattern for customizable load balancing and request routing that is optimized for Inference. An **[inference gateway]** provides a reference implementation of model endpoint picking leveraging metrics emitted from the model servers. This endpoint picking mechanism can be used in lieu of traditional load balancing mechanisms. Model Server-aware load balancing ("smart" load balancing as its sometimes referred to in this repo) has been proven to reduce the serving latency and improve utilization of accelerators in your clusters.

By achieving these, the project aims to reduce latency and improve accelerator (GPU) utilization for AI workloads.

## API Resources

Head to our [API overview](/concepts/api-overview/#api-overview) to start exploring our APIs!

## Composable Layers

This project aims to define specifications to enable a compatible ecosystem for
extending the Gateway API with custom endpoint selection algorithms. This
project defines a set of patterns across three distinct layers of components
that are relevant to this project:

### Gateway API Implementations

Gateway API has [more than 25
implementations](https://gateway-api.sigs.k8s.io/implementations/). As this
pattern stabilizes, we expect a wide set of these implementations to support
this project to become an **[inference gateway]**

### Endpoint Picker

As part of this project, we've built the Endpoint Picker. A pluggable & extensible ext-proc deployment that implements [this architecture](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/0683-epp-architecture-proposal).

### Model Server Frameworks

This project will work closely with model server frameworks to establish a
shared standard for interacting with these extensions, particularly focused on
metrics and observability so extensions will be able to make informed routing
decisions. The project is currently focused on integrations with
[vLLM](https://github.com/vllm-project/vllm) and
[Triton](https://github.com/triton-inference-server/server), and will be open to
other integrations as they are requested.

## Request Flow

To illustrate how this all comes together, it may be helpful to walk through a
sample request.

1. The first step involves the Gateway selecting the correct InferencePool
(set of endpoints running a model server framework) or Service to route to. This
logic is based on the existing Gateway and HTTPRoute APIs, and will be familiar
to any Gateway API users or implementers.

2. If the request should be routed to an InferencePool, the Gateway will forward
the request information to the endpoint selection extension for that pool.

3. The inference gateway will fetch metrics from whichever portion of the InferencePool
endpoints can best achieve the configured objectives. Note that this kind of
metrics probing may happen asynchronously, depending on the inference gateway.

4. The inference gateway will instruct the Gateway which endpoint the request should be
routed to.

5. The Gateway will route the request to the desired endpoint.

<img src="/images/request-flow.png" alt="Inference Gateway Request Flow" class="center" />


## Who is working on Gateway API Inference Extension?

This project is being driven by
[WG-Serving](https://github.com/kubernetes/community/tree/master/wg-serving)
[SIG-Network](https://github.com/kubernetes/community/tree/master/sig-network)
to improve and standardize routing to inference workloads in Kubernetes. Check
out the [implementations reference](implementations/gateways.md) to see the latest
projects & products that support this project. If you are interested in
contributing to or building an implementation using Gateway API then don’t
hesitate to [get involved!](/contributing)
