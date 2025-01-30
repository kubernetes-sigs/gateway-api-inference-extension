# API Overview

## Bakcground
The Gateway API Inference Extension project is an extension of the Kubernetes Gateway API for serving Generative AI models on Kubernetes. Gateway API Inference Extension facilitates standardization of APIs for Kubernetes cluster operators and developers running generative AI inference, while allowing flexibility for underlying gateway implementations (such as Envoy Proxy) to iterate on mechanisms for optimized serving of models. 

<img src="/images/inference-overview.png" alt="Overview of API integration" class="center" width="700" />


## Key Features 
Gateway API Inference Extension, along with a reference implementation in Envoy Proxy, provides the following key features: 

- **Model-aware routing**: Instead of simply routing based on the path of the request, Gateway API Inference Extension allows you to route to models based on the model names. This is enabled by support for GenAI Inference API specifications (such as OpenAI API) in the gateway implementations such as in Envoy Proxy. This model-aware routing also extends to Low-Rank Adaptation (LoRA) fine-tuned models.

- **Serving priority**: Gateway API Inference Extension allows you to specify the serving priority of your models. For example, you can specify that your models for online inference of chat tasks (which is more latency sensitive) have a higher priority than a model for latency tolerant tasks such as a summarization. 

- **Model rollouts**:  Gateway API Inference Extension allows you to incrementally roll out new model versions by traffic splitting definitions based on the model names. 

- **Extensibility for Inference Services**: Gateway API Inference Extension defines extensibility pattern for additional Inference services such as AI Safety checks or Semantic Caching. Gateway API Inference Extension.


- **Customizable Load Balancing for Inference**: Gateway API Inference Extension defines a pattern for customizable load balancing and request routing that is optimized for Inference. Gateway API Inference Extension provides a reference implementation of model endpoint picking based on metrics based on Inference Servers, compatible with Envoy Proxy. This endpoint picking mechanism can be used for load balancing based on the Inference Server metrics, instead of traditional mechanisms such as round-robin or random endpoint selection. Load Balancing based on Inference Server metrics can help reduce the serving latency and improve utilization of accelerators in your clusters. 

## API Resources

### InferencePool

InferencePool represents a set of Inference-focused Pods and an extension that will be used to route to them. Within the broader Gateway API resource model, this resource is considered a "backend". In practice, that means that you'd replace a Kubernetes Service with an InferencePool. This resource has some similarities to Service (a way to select Pods and specify a port), but has some unique capabilities. With InferenceModel, you can configure a routing extension as well as inference-specific routing optimizations. For more information on this resource, refer to our [InferencePool documentation](/api-types/inferencepool.md).

### InferenceModel

An InferenceModel represents a model or adapter, and configuration associated with that model. This resource enables you to configure the relative criticality of a model, and allows you to seamlessly translate the requested model name to one or more backend model names. Multiple InferenceModels can be attached to an InferencePool. For more information on this resource, refer to our [InferenceModel documentation](/api-types/inferencemodel.md).
