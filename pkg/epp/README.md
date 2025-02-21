# The EndPoint Picker (EPP)
This package provides the reference implementation for the Endpoint Picker (EPP). It implements the [extension protocol](../../docs/proposals/003-endpoint-picker-protocol), enabling a proxy or gateway to request endpoint hints from an extension. As it is implemented now, an EPP instance handles a single `InferencePool` (and so for each `InferencePool`, one must create a dedicated EPP deployment).


The Endpoint Picker performs the following core functions:

- Endpoint Selection
  - The EPP determines the appropriate Pod endpoint for the load balancer (LB) to route requests.
  - It selects from the pool of ready Pods designated by the assigned InferencePool.
  - Endpoint selection is contingent on the request's ModelName matching an `InferenceModel` that references the `InferencePool`.
  - Requests with unmatched ModelName values trigger an error response to the proxy.
  - The endpoint selection algorithm is detailed below.
- Traffic Splitting and ModelName Rewriting
  - The EPP facilitates controlled rollouts of new adapter versions by implementing traffic splitting between adapters within the same `InferencePool`, as defined by the `InferenceModel`.
  - EPP rewrites the model name in the request to the target model name as defined on the `InferenceModel` object.
- Observability
  - The EPP generates metrics to enhance observability.
  - It reports InferenceModel-level metrics, further broken down by target model.
  - Detailed information regarding metrics can be found on the [website](https://gateway-api-inference-extension.sigs.k8s.io/guides/metrics/).

## The scheduling algorithm 
The scheduling package implements request scheduling algorithms for load balancing requests across backend pods in an inference gateway. The scheduler ensures efficient resource utilization while maintaining low latency and prioritizing critical requests. It applies a series of filters based on metrics and heuristics to select the best pod for a given request. The following flow chart summarizes the current scheduling algorithm

# Flowchart
<img src="../../docs/scheduler-flowchart.png" alt="Scheduling Algorithm" width="400" />
