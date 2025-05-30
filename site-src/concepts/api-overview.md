# API Overview

## Background
Gateway API Inference Extension project optimizes self-hosting Generative Models on Kubernetes.
This is achieved by leveraging Envoy's [External Processing](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter) to extend any gateway that supports both ext-proc and [Gateway API](https://github.com/kubernetes-sigs/gateway-api) into an **[inference gateway]**.
This extension upgrades an [ext-proc](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter)
capable proxy or gateway - such as Envoy Gateway, kGateway, or the GKE Gateway - to become an [inference gateway](../index.md#concepts-and-definitions) -
supporting inference platform teams self-hosting Generative Models (with a current focus on large language models) on Kubernetes.
This integration makes it easy to expose and control access to your local [OpenAI-compatible chat completion endpoints](https://platform.openai.com/docs/api-reference/chat)
to other workloads on or off cluster, or to integrate your self-hosted models alongside model-as-a-service providers in a higher level **AI Gateway** like LiteLLM, Solo AI Gateway, or Apigee.

<img src="/images/inference-overview.svg" alt="Overview of API integration" class="center" width="1000" />

## API Resources

### InferencePool

InferencePool represents a set of Inference-focused Pods and an extension that will be used to route to them. Within the broader Gateway API resource model, this resource is considered a "backend". In practice, that means that you'd replace a Kubernetes Service with an InferencePool. This resource has some similarities to Service (a way to select Pods and specify a port), but has some unique capabilities. With InferencePool, you can configure a routing extension as well as inference-specific routing optimizations. For more information on this resource, refer to our [InferencePool documentation](/api-types/inferencepool) or go directly to the [InferencePool spec](/reference/spec/#inferencepool).

### InferenceModel

An InferenceModel represents a model or adapter, and configuration associated with that model. This resource enables you to configure the relative criticality of a model, and allows you to seamlessly translate the requested model name to one or more backend model names. Multiple InferenceModels can be attached to an InferencePool. For more information on this resource, refer to our [InferenceModel documentation](/api-types/inferencemodel) or go directly to the [InferenceModel spec](/reference/spec/#inferencemodel).
