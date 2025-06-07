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
    - group: inference.networking.x-k8s.io
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
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferencePool
metadata:
  name: vllm-llama3-8b-instruct
spec:
  targetPortNumber: 8000
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

The [Endpoint Picker](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/pkg/epp), or EPP, is a core component of the inference extension. The primary interaction for routing requests is defined between the proxy (e.g., Envoy) and the EPP using the Envoy [external processing service protocol](https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor.proto). See the [Endpoint Picker Protocol](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/004-endpoint-picker-protocol) for more information.

#### How to Callout to EPP

For each HTTP request, the proxy CAN communicate the subset of endpoints the EPP MUST pick from by setting `x-gateway-destination-endpoint-subset` key in the filter metadata field of the ext-proc request. If this key is set, the EPP must select from this endpoint list. If the list is empty or no endpoints are eligible, it should return a 503 error. If the key isn't set, the EPP selects from the endpoints defined by the InferencePool selector.

#### Response from the extension

The EPP communicates the chosen endpoint to the proxy via the `x-gateway-destination-endpoint` HTTP header and the `dynamic_metadata` field of the ext-proc response. Failure to communicate the endpoint using both methods results in a 503 error if no endpoints are ready, or a 429 error if the request should be dropped. The header and metadata values must match. In addition to the chosen endpoint, a single fallback endpoint CAN be set using the key `x-gateway-destination-endpoint-fallback` in the same metadata namespace as one used for `x-gateway-destination-endpoint`.

## Testing Tips

Here are some tips for testing your controller end-to-end:

- **Focus on Key Scenarios**: Add common scenarios like creating, updating, and deleting InferencePool resources, as well as different routing rules that target InferencePool backends.
- **Verify Routing Behaviors**: Design more complex routing scenarios and verify that requests are correctly routed to the appropriate model server pods within the InferencePool based on the InferenceModel configuration.
- **Test Error Handling**: Verify that the controller correctly handles scenarios like unsupported model names or resource constraints (if criticality-based shedding is implemented). Test with state transitions (such as constant requests while Pods behind EPP are being replaced and Pods behind InferencePool are being replaced) to ensure that the system is resilient to failures and can automatically recover by redirecting traffic to healthy Pods.
- **Using Reference EPP Implementation + Echoserver**: You can use the [reference EPP implementation](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/pkg/epp) for testing your controller end-to-end. Instead of a full-fledged model server, a simple mock server (like the [echoserver](https://github.com/kubernetes-sigs/ingress-controller-conformance/tree/master/images/echoserver)) can be very useful for verifying routing to ensure the correct pod received the request. 
- **Performance Test**: Run end-to-end [benchmarks](https://gateway-api-inference-extension.sigs.k8s.io/performance/benchmark/) to make sure that your inference gateway can achieve the latency target that is desired.

### Conformance Tests

See [Conformance Test Setup and Execution](https://gateway-api-inference-extension.sigs.k8s.io/guides/conformance-tests).
