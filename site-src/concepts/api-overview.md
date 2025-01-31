# API Overview

## Bakcground
The Gateway API Inference Extension project is an extension of the Kubernetes Gateway API for serving Generative AI models on Kubernetes. Gateway API Inference Extension facilitates standardization of APIs for Kubernetes cluster operators and developers running generative AI inference, while allowing flexibility for underlying gateway implementations (such as Envoy Proxy) to iterate on mechanisms for optimized serving of models. 

<img src="/images/test.svg" alt="Overview of API integration" class="center" width="1000" />

## API Resources

### InferencePool

InferencePool represents a set of Inference-focused Pods and an extension that will be used to route to them. Within the broader Gateway API resource model, this resource is considered a "backend". In practice, that means that you'd replace a Kubernetes Service with an InferencePool. This resource has some similarities to Service (a way to select Pods and specify a port), but has some unique capabilities. With InferenceModel, you can configure a routing extension as well as inference-specific routing optimizations. For more information on this resource, refer to our [InferencePool documentation](/api-types/inferencepool.md) or go directly to the [InferencePool spec](/reference/spec/#inferencepool).

### InferenceModel

An InferenceModel represents a model or adapter, and configuration associated with that model. This resource enables you to configure the relative criticality of a model, and allows you to seamlessly translate the requested model name to one or more backend model names. Multiple InferenceModels can be attached to an InferencePool. For more information on this resource, refer to our [InferenceModel documentation](/api-types/inferencemodel.md) or go directly to the [InferenceModel spec](/reference/spec/#inferencemodel).
