# Implementer's Guide

This guide is intended for developers looking to implement support for the InferencePool custom resources within their Gateway API controller. It outlines how InferencePool fits into the existing resource model, discusses implementation options, explains how to interact with extensions, and provides guidance on testing.

## InferencePool as a Gateway Backend
Before we dive into the implementation, let’s recap how an InferencePool works. 

<img src="/images/inference-overview.svg" alt="Overview of API integration" class="center" width="1000" />

**InferencePool** represents a set of Inference-focused Pods and an extension that will be used to route to them. The InferencePool introduces a new type of backend within the Gateway API resource model. Instead of targeting Services, a Gateway can route traffic to an InferencePool. This InferencePool then becomes responsible for intelligent routing to the underlying model server pods based on the associated InferenceModel configurations. 

Here is an example of how to route traffic to an InferencePool using an HTTPRoute:
```
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: llm-route
spec:
  parentRefs:
  - group: gateway.networking.k8s.io
    kind: Gateway
    name: inference-gateway
  rules:
  - backendRefs:
    - group: inference.networking.k8s.io
      kind: InferencePool
      name: base-model
    matches:
    - path:
        type: PathPrefix
        value: /
```

Note that the `rules.backendRefs` describes which InferencePool should receive the forwarded traffic when the path matches the corresponding path prefix. This is very similar to how we configure a Gateway with an HTTPRoute that directs traffic to a Service (a way to select Pods and specify a port). By using the InferencePool, it provides an abstraction over a set of compute resources (model server pods), and allows the controller to implement specialized routing strategies for these inference workloads.

## Building the Gateway controller 
The general idea of implementing a Gateway controller supporting the InferencePool involves two major steps: 

1. Tracking the endpoints for InferencePool backends 
2. Callout to an extension to make intelligent routing decisions

### Endpoint Tracking
Consider a simple inference pool like this:
```
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: vllm-llama3-8b-instruct
spec:
  targetPorts:
    - number: 8000
  selector:
    app: vllm-llama3-8b-instruct
  extensionRef:
    name: vllm-llama3-8b-instruct-epp
```
There are mainly two options for how to treat the Inference Pool in your controller.

**Option 1: Shadow Service Creation**

If your Gateway controller already handles Service as a backend, you can choose to create a headless Service that mirrors the endpoints defined by the InferencePool, like this:

```
apiVersion: v1
kind: Service
metadata: 
  name: vllm-llama3-8b-instruct-shadow-service
spec:
  ports:
  - port: 54321
    protocol: TCP
    targetPort: 8000
  selector:
    app:  vllm-llama3-8b-instruct
  type: ClusterIP
  clusterIP: None
```

The gateway controller would then treat this shadow service just like any other backend service it routes traffic to. 

This approach likely allows you to leverage existing service discovery, healthcheck infrastructure, and load balancing mechanisms that your controller already supports. However, it does come with the overhead of managing additional Service objects, and hence may affect the overall latency of the reconciliation of the Gateways.

**Option 2: Tracking InferencePool Endpoints Separately**

You can also choose to directly select and monitor the endpoints belonging to the InferencePool. For the simple inference pool example we have above, the controller would use the label `app: vllm-llama3-8b-instruct` to discover the pods matching the criteria, and get their endpoints (i.e. IP and port number). It would then need to monitor these pods for health and availability. 

With this approach, you can tailor the endpoint tracking and routing logic specifically to the characteristics and requirements of your InferencePool.

### Callout Extension

The [Endpoint Picker](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/pkg/epp), or EPP, is a core component of the inference extension. The primary interaction for routing requests is defined between the proxy (e.g., Envoy) and the EPP using the Envoy [external processing service protocol](https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor.proto). See the [Endpoint Picker Protocol specification](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/004-endpoint-picker-protocol) and the [Implementing a Compatible Data Plane section](#implementing-a-compatible-data-plane) section below for more details.

#### How to Callout to EPP

For each HTTP request, the proxy CAN communicate the subset of endpoints the EPP MUST pick from by setting `x-gateway-destination-endpoint-subset` key in the filter metadata field of the ext-proc request. If this key is set, the EPP must select from this endpoint list. If the list is empty or no endpoints are eligible, it should return a 503 error. If the key isn't set, the EPP selects from the endpoints defined by the InferencePool selector.

#### Response from the extension

The EPP communicates the chosen endpoint to the proxy via the `x-gateway-destination-endpoint` HTTP header and the `dynamic_metadata` field of the ext-proc response. Failure to communicate the endpoint using both methods results in a 503 error if no endpoints are ready, or a 429 error if the request should be dropped. The header and metadata values must match. In addition to the chosen endpoint, a single fallback endpoint CAN be set using the key `x-gateway-destination-endpoint-fallback` in the same metadata namespace as one used for `x-gateway-destination-endpoint`.

### Implementing a Compatible Data Plane

To conform with the Inference Extensions API, Gateway data planes must implement the [Endpoint Picker Protocol](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/004-endpoint-picker-protocol).

At a high level, the protocol consists of metadata key/value pairs exchanged between the data plane and extensions containing relevant endpoint selection information:

- From extension to data plane: the metadata contains the selected endpoints.
- From data plane to extension: the metadata contains an optional subset of endpoints that the extension should pick from.

The key requirements for implementing the GIE protocol are as follows:

- Relies on the [ext_proc (External Processing)](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter) protocol as the foundation for exchanging HTTP stream payload and metadata throughout the various HTTP lifecycle events; several key details:
    - ext_proc relies on gRPC (bidirectional streaming) as the transport protocol
    - ext_proc supports several processing modes, including buffered and streaming options for payload exchange
    - ext_proc supports structured metadata passed as part of requests and responses for each processing stage
- The Inference Extension protocol exchanges data between proxy and extension servers as metadata — either via HTTP headers or the structured fields in the ext_proc messages — using well defined names and values:
    - **x-gateway-destination-endpoint**
        - Informs the proxy of the selected (primary) endpoint along with fallback endpoints for retries (if needed).
        - Sent by the extension service to the data plane as [ProcessingResponse](https://github.com/envoyproxy/envoy/blob/v1.34.2/api/envoy/service/ext_proc/v3/external_processor.proto) metadata in response to HTTP request stage events.
    - **x-gateway-destination-endpoint-subset (optional)**
        - Contains the subset of endpoints the extension should pick from.
        - Sent by the data plane to the extension service as [ProcessingRequest](https://github.com/envoyproxy/envoy/blob/v1.34.2/api/envoy/service/ext_proc/v3/external_processor.proto) metadata during HTTP request stage events

#### External Processing Protocol

ext_proc is a mature protocol, implemented by Envoy to support communication with external processing services. It has gained adoption across several types of use cases:

- [Google Cloud Load Balancer and CDN Service Extensions](https://cloud.google.com/service-extensions/docs/overview)
    - Supports generic “service callouts” not restricted to genAI serving or AI use cases; e.g., mutation of cache keys for caching.
- [Alibaba Cloud](https://www.alibabacloud.com/help/en/asm/user-guide/use-envoy-external-processing-for-custom-processing-of-requests)
- GenAI serving
    - [AIBrix](https://aibrix.readthedocs.io/latest/features/gateway-plugins.html)
        - Enables inference optimized routing for the Gateway in Bytedance’s genAI inference infrastructure.
    - [Envoy AI Gateway](https://aigateway.envoyproxy.io/docs/concepts/architecture/data-plane)
        - Enables AI model based routing, request transformations and upstream authn.
- [Atlassian Guard](https://www.atlassian.com/software/guard)

Supporting this broad range of extension capabilities (including for inference, as evidenced above) requires hooks into all HTTP stream (i.e., request and response) lifecycle events as well as the corresponding headers, trailers and payload. This is the core value proposition for ext_proc, along with configurable options (such as for buffering and streaming modes) that enable its use across a variety of deployment scenarios and networking topologies.

#### Open Source Implementations

Several implementations can be used as references:

- A fully featured [reference implementation](https://github.com/envoyproxy/envoy/tree/main/source/extensions/filters/http/ext_proc) (C++) can be found in the Envoy GitHub repository.
- A second implementation (Rust, non-Envoy) is available in [agentgateway](https://github.com/agentgateway/agentgateway/blob/v0.7.2/crates/agentgateway/src/http/ext_proc.rs).

#### Portable Implementation

A portable WASM module implementing ext_proc can be developed, leveraging the [Proxy-Wasm ABI](https://github.com/proxy-wasm/spec) that is now supported by hosts such as Envoy, NGINX, Apache Traffic Server and others. This enables a common implementation to be shared, until native support is implemented or as a long term solution depending on each host’s needs.

A challenge to this option is that Proxy-Wasm becomes a dependency and may need to evolve in conjunction with ext_proc. With that said, this is very unlikely to be a problem in practice, given the breadth of Proxy-Wasm’s ABI and the use cases in scope of the ext_proc protocol.

An example of a similar approach is Kuadrant’s [WASM Shim](https://github.com/Kuadrant/wasm-shim/tree/main), which implements the protocols required by External Authorization and Rate Limiting Service APIs as a WASM module.

## Testing Tips

Here are some tips for testing your controller end-to-end:

- **Focus on Key Scenarios**: Add common scenarios like creating, updating, and deleting InferencePool resources, as well as different routing rules that target InferencePool backends.
- **Verify Routing Behaviors**: Design more complex routing scenarios and verify that requests are correctly routed to the appropriate model server pods within the InferencePool.
- **Test Error Handling**: Verify that the controller correctly handles scenarios like unsupported model names or resource constraints (if priority-based shedding is implemented). Test with state transitions (such as constant requests while Pods behind EPP are being replaced and Pods behind InferencePool are being replaced) to ensure that the system is resilient to failures and can automatically recover by redirecting traffic to healthy Pods.
- **Using Reference EPP Implementation + Echoserver**: You can use the [reference EPP implementation](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/pkg/epp) for testing your controller end-to-end. Instead of a full-fledged model server, a simple mock server (like the [echoserver](https://github.com/kubernetes-sigs/ingress-controller-conformance/tree/master/images/echoserver)) can be very useful for verifying routing to ensure the correct pod received the request. 
- **Performance Test**: Run end-to-end [benchmarks](https://gateway-api-inference-extension.sigs.k8s.io/performance/benchmark/) to make sure that your inference gateway can achieve the latency target that is desired.

### Conformance Tests

See [Conformance Test Setup and Execution](https://gateway-api-inference-extension.sigs.k8s.io/guides/conformance-tests).
