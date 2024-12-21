# Introduction

Gateway API Inference Extension is an official Kubernetes project focused on
extending [Gateway API](https://gateway-api.sigs.k8s.io/) with inference
specific routing extensions.

The overall resource model focuses on 2 new inference-focused
[personas](/concepts/roles-and-personas) and corresponding resources that
they are expected to manage:

<!-- Source: https://docs.google.com/presentation/d/11HEYCgFi-aya7FS91JvAfllHiIlvfgcp7qpi_Azjk4E/edit#slide=id.g292839eca6d_1_0 -->
<img src="/images/resource-model.png" alt="Gateway API Inference Extension Resource Model" class="center" width="550" />

## API Resources

### InferencePool

### InferenceModel

## Composable Layers

This project aims to develop an ecosystem of implementations that are fully
compatible with each other. There are three distinct layers of components that
are relevant to this project:

### Gateway API Implementations

Gateway API has [more than 25
implementations](https://gateway-api.sigs.k8s.io/implementations/). As this
pattern stabilizes, we expect a wide set of these implementations to support
this project.

### Endpoint Selection Extension

As part of this project, we're building an initial reference extension that is
focused on routing to LoRA workloads. Over time, we hope to see a wide variety
of extensions emerge that follow this pattern and provide a wide range of
choices.

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

1. The first step involves the Gateway selecting the the correct InferencePool
(set of endpoints running a model server framework) or Service to route to. This
logic is based on the existing Gateway and HTTPRoute APIs, and will be familiar
to any Gateway API users or implementers.

2. If the request should be routed to an InferencePool, the Gateway will forward
the request information to the endpoint selection extension for that pool.

3. The extension will fetch metrics from whichever portion of the InferencePool
endpoints can best achieve the configured objectives. Note that this kind of
metrics probing may happen asynchronously, depending on the extension.

4. The extension will instruct the Gateway which endpoint should be routed to.

5. The Gateway will route the request to the desired endpoint.

<img src="/images/request-flow.png" alt="Gateway API Inference Extension Request Flow" class="center" />


## Who is working on Gateway API Inference Extension?

This project is being driven by
[WG-Serving](https://github.com/kubernetes/community/tree/master/wg-serving)
[SIG-Network](https://github.com/kubernetes/community/tree/master/sig-network)
to improve and standardize routing to inference workloads in Kubernetes. Check
out the [implementations reference](implementations.md) to see the latest
projects & products that support this project. If you are interested in
contributing to or building an implementation using Gateway API then don’t
hesitate to [get involved!](/contributing)
