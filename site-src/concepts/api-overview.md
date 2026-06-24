# API Overview

## Background
Gateway API Inference Extension optimizes self-hosting Generative AI Models on Kubernetes. 
It provides optimized load-balancing for self-hosted Generative AI Models on Kubernetes.
The project’s goal is to improve and standardize routing to inference workloads across the ecosystem.

This is achieved by leveraging Envoy's [External Processing Protocol](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter) to extend any gateway that supports both ext-proc and [Gateway API](https://github.com/kubernetes-sigs/gateway-api) into an [inference gateway](../index.md#concepts-and-definitions).
This extension extends popular gateways like Envoy Gateway, kgateway, and GKE Gateway - to become [Inference Gateway](../index.md#concepts-and-definitions) -
supporting inference platform teams self-hosting Generative Models (with a current focus on large language models) on Kubernetes.
This integration makes it easy to expose and control access to your local [OpenAI-compatible chat completion endpoints](https://platform.openai.com/docs/api-reference/chat)
to other workloads on or off cluster, or to integrate your self-hosted models alongside model-as-a-service providers 
in a higher level **AI Gateways** like [LiteLLM](https://www.litellm.ai/), [Gloo AI Gateway](https://www.solo.io/products/gloo-ai-gateway), or [Apigee](https://cloud.google.com/apigee).

## API Resources

Gateway API Inference Extension introduces inference-focused API resources with distinct responsibilities, 
each aligning with a specific user persona in the Generative AI serving workflow.

<img src="/images/inference-overview.svg" alt="Overview of API integration" class="center" width="1000" />

### InferencePool

InferencePool represents a set of Inference-focused Pods and an extension that will be used to route to them. Within the broader Gateway API resource model, this resource is considered a "backend". In practice, that means that you'd replace a Kubernetes Service with an InferencePool. This resource has some similarities to Service (a way to select Pods and specify a port), but has some unique capabilities. With InferencePool, you can configure a routing extension as well as inference-specific routing optimizations. For more information on this resource, refer to our [InferencePool documentation](/api-types/inferencepool) or go directly to the [InferencePool spec](/reference/spec/#inferencepool).

### InferencePoolImport

The `InferencePoolImport` resource is a cluster-local, controller-managed representation of an imported `InferencePool` from another cluster, allowing traffic to be routed to model servers running in a remote cluster. Unlike `InferencePool`, it is not user-authored but populated by the multi-cluster controller. For more information on this resource, refer to our [InferencePoolImport documentation](/api-types/inferencepoolimport) or go directly to the [InferencePoolImport spec](/reference/x-v1a1-spec/#inferencepoolimport).
