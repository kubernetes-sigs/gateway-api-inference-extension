# Flow Control Layer User Guide

The Flow Control layer in the Endpoint Picker (EPP) is a pool defense and multi-tenancy mechanism. It protects your pool of model servers from overload while enforcing strict priority and tenant fairness.

## The Elevator Pitch

By shifting intelligent queuing into the EPP, the Flow Control layer provides three critical benefits:

1. **Fairness & Prioritization:** Prevents "noisy neighbors" from monopolizing GPUs. High-priority requests strictly bypass lower-priority traffic, and resources are shared equitably among tenants from the same priority during spikes.
2. **Resource Efficiency & Late Binding:** Prevents "scheduling regret." By keeping endpoint queues small and holding requests centrally, the EPP delays routing decisions until the last possible moment. This late-binding approach allows the EPP to dispatch requests to the most optimal candidate that is not currently saturated (e.g., the endpoint with the highest Prefix Cache affinity), rather than locking it prematurely into a suboptimal endpoint's local queue.
3. **Operational Stability:** Smooths out traffic spikes, hides endpoint cold-start times from clients (by queuing requests instead of dropping them), and provides clearer signals for pool autoscaling.
4. **Centralized Backpressure & Observability:** Model servers protect their hardware during spikes by queuing requests locally. However, requests trapped inside isolated, local queues cannot be dynamically re-routed or preempted by higher-priority traffic, and they do not automatically expose tenant-specific backpressure metrics. By actively monitoring backend saturation signals, the EPP intercepts this backpressure at the proxy layer. Buffering excess load centrally empowers the gateway to enforce global priority, maintain fairness, make optimal late-binding routing decisions, and surface tenant-specific queue metrics that individual downstream endpoints cannot.

## Why Flow Control? (The LLM Queuing Problem)

Model servers are highly optimized for a single, local goal: maximizing GPU throughput by building efficient continuous batches. However, they are entirely unaware of global, business-level objectives like tenant fairness or SLAs.

Without Flow Control, a sudden burst of traffic piles up directly inside the model server's internal queues. This creates severe issues:

* **The Noisy Neighbor:** A single user sending massive, complex prompts can monopolize the endpoint's queue and KV-cache, starving all other tenants.
* **Scheduling Regret (Premature Routing):** Once a request is queued inside a specific model server, the EPP cannot move it. Dispatching early locks the request to a suboptimal routing candidate, preventing it from utilizing an endpoint that might soon free up and already hold the user's system prompt in its physical KV-cache.
* **Priority Inversion:** Model servers batch based on sequence lengths or arrival times, not business priority. A critical real-time chat request might get stuck behind a massive offline batch job.
* **Resource Asymmetry:** In traditional web services, connection count (RPS) is a reliable proxy for load. In LLM inference, resource consumption is predominantly driven by token counts (the total length of the sequence being processed) and the unpredictability of the autoregressive decode loop. A single request with a massive input context (such as a RAG prompt) or an unbounded max generation limit can consume vastly more KV cache capacity and GPU compute than hundreds of short chat requests. Standard API gateways rate-limit by RPS, which blindly allows heavy requests to saturate the available endpoint memory and fill the model server's local buffer. The EPP Flow Control layer is designed to govern this physical capacity (KV cache, queue saturation metrics), not just HTTP connection counts.

**The Solution:** The Flow Control layer solves these issues by **shifting queuing to the gateway** instead of the model servers. By holding excess load inside the EPP's policy-aware queues, the EPP buffers these excess requests. It only dispatches requests to the model servers when they actually have the capacity to process them.

## Core Concepts

Traffic is organized into **Flows**. When a request arrives, the EPP assigns it a `FlowKey` consisting of two parts:

1. **Fairness ID:** An identifier extracted from the `x-gateway-inference-fairness-id` HTTP header (e.g., a tenant ID, a user tier, or an API key). If absent, it defaults to a global bucket.
2. **Priority:** An integer value derived from the [`InferenceObjective`](../concepts/priority-and-capacity.md) Kubernetes resource targeting the pool. Negative values are permissible and explicitly define background/low-priority traffic.

### Priority (Strict Ordering)
Priority provides a hard guarantee for service order. The Flow Controller will **always** dispatch all buffered requests from higher-priority queues before servicing any requests from lower-priority queues. Unlike the default admission mode (when Flow Control is disabled), negative-priority requests are not immediately rejected upon saturation but are held in their own dynamically provisioned queues until dispatched, until they expire, or until configured [capacity limits](epp-configuration/config-text.md#priority-band-configuration) are exceeded. This means operators can control load shedding by strictly limiting the capacity applied to lower-priority levels.

### Fairness (Equitable Sharing)
Fairness policies determine how to share resources between different flows that exist *within the same Priority level*.
Crucially, the Flow Control layer is **work-conserving**. It will not artificially throttle requests if the GPUs have spare capacity. Fairness policies only activate when the system is under contention, ensuring each competing tenant gets an equitable share of the dispatch opportunities.

### The User Experience: TTFT vs. TPOT
When a pool is under heavy load, operators must choose how the system degrades. Without Flow Control, endpoints accept all requests and attempt to timeslice the GPU. This results in a fast Time-To-First-Token (TTFT), but a degraded Time-Per-Output-Token (TPOT) as the GPU thrashes between contexts, despite the continuous batching engine's best efforts to prevent total stalling.

By enabling the Flow Control layer, you make the explicit choice to protect TPOT at the expense of queue time. During a spike, users will wait in the EPP queue.

**The Late Binding Mechanism:** While requests wait in the queue, the EPP uses this delay to its advantage via Late Binding. By delaying the scheduling decision until the last possible moment, the EPP boosts the efficacy of its pluggable high-affinity scorers. Rather than scheduling prematurely to a suboptimal backend, the EPP can dynamically schedule the request to the endpoint that scores the highest across its criteria. While the Flow Control layer intentionally trades mean TTFT for a protected TPOT, late-binding is designed to reduce latency variance (tail latency). By matching the request to an optimal high-affinity backend, the GPU can skip unnecessary compute and deliver a more consistent, predictable user experience.

## How It Works: The Dispatch Loop

1. **Enqueue & Wait:** Incoming requests are sorted into queues based on their `FlowKey`. The request's goroutine blocks, waiting for a signal.
2. **Policy Evaluation:** Background workers continuously evaluate the queues, selecting the "next best" request based on Strict Priority, followed by the intra-Priority Fairness policy.
3. **The Saturation Check (The Gatekeeper):** Before dispatching the selected request, the EPP queries the **Saturation Detector**. The detector evaluates the aggregate health of the pool (e.g., checking overall KV-cache utilization and local queue depth).
    * **If the pool has capacity:** The request is dispatched immediately.
    * **If the pool is saturated:** The dispatch cycle halts. This enforces **Strict Priority Preservation** at the EPP, holding the highest-priority request safely in memory until the pool recovers. Lower-priority requests are intentionally held back to guarantee they do not steal the exact GPU cycles the high-priority request is waiting for.

## Configuration Guide

The Flow Control layer leverages the standard [EndpointPickerConfig](epp-configuration/config-text.md#flow-control-configuration) schema for configuration.

### 1. Enabling the Layer
Enable the Flow Control layer by adding the "flowControl" FeatureGate to your `EndpointPickerConfig`:

```yaml
apiVersion: config.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
featureGates:
  - "flowControl"
# ...
```

### 2. Tuning the [Saturation Detector](./epp-configuration/config-text.md#saturation-detector-plugins) (The "Healthy Buffer")

The effectiveness of the Flow Control layer depends entirely on the Saturation Detector. Your goal is to tune these settings to maintain a small, "healthy buffer" of requests on the model servers—enough to form efficient continuous batches, but no more.

To customize these thresholds, define the plugin parameters in the global `plugins` list and reference that plugin in your `saturationDetector` block. When tuning the default `utilization-detector`, consider the following:

* **`queueDepthThreshold` (Default: `5`)**: Controls the allowed size of the pending queue *on the model servers themselves*.
  * **For Maximum Throughput:** Set a small, non-zero value (e.g., a fraction of your max batch size) to ensure the GPU is never starved for work.
  * **For Maximum Fairness/Control:** Set this to `1`. This forces nearly all queuing to happen centrally inside the EPP, ensuring maximum control over priority, fairness, and late-binding routing.
* **`kvCacheUtilThreshold` (Default: `0.8`)**: The maximum KV-cache memory utilization allowed before the pool is considered saturated and EPP queueing engages.

**Example:**
```yaml
plugins:
- type: utilization-detector
  parameters:
    queueDepthThreshold: 2 # Controls allowed size of queue on model servers (Default: 5)
    kvCacheUtilThreshold: 0.85 # Max KV-cache utilization allowed before saturation (Default: 0.8)
saturationDetector:
  pluginRef: utilization-detector
```

#### Alternative: Concurrency Detector Tuning

If you are using the alternative `concurrency-detector`, your buffer is dictated by the maximum active request limits rather than hardware telemetry.

We recommend setting `maxConcurrency` to 110% of your model server's active batch capacity to provide a 10% local queue buffer. This allows the continuous batching engine to perform optimally, providing it with a steady stream of requests to pull into the next batch without creating massive local queues. For example, if your model server's max active batch size is `100`, set `maxConcurrency` to `110`.

**Example:**
```yaml
plugins:
- type: concurrency-detector
  parameters:
    concurrencyMode: "requests"
    maxConcurrency: 110 # 100 active batch size + 10 buffered requests
saturationDetector:
  pluginRef: concurrency-detector
```

### 3. [Priority Bands and Capacity Config](epp-configuration/config-text.md#priority-band-configuration)

Use the `EndpointPickerConfig.flowControl` configuration block to define your dynamic priority bands and global capacity constraints.

```yaml
flowControl:
  # maxBytes limits the aggregate HTTP payload size of all pending requests held in
  # the EPP's memory. (Note: This bounds proxy memory footprint, not GPU VRAM or Tokens).
  # Supports both plain integers (bytes) and Kubernetes Quantity format (e.g., 10Gi, 512Mi).
  maxBytes: 1000000000 # 1GB total HTTP payload capacity limit
  defaultRequestTTL: 30s # Fallback TTL if client doesn't specify one
  priorityBands:
    - priority: 100
      maxBytes: 500000000 # 500MB HTTP payload limit for Priority 100
      # Default: "global-strict-fairness-policy"
      fairnessPolicyRef: "global-strict-fairness-policy"
      # Default: "fcfs-ordering-policy"
      orderingPolicyRef: "fcfs-ordering-policy"
```

## Autoscaling: KEDA and Scale-to-Zero

Autoscaling LLM backends presents unique challenges. Standard hardware metrics like CPU or GPU utilization reflect physical activity, but they fail to quantify unfulfilled user demand. Because LLM resource consumption is highly non-linear, a GPU operating at 100% compute utilization might be processing a single massive prompt or perfectly multiplexing a hundred smaller ones. This makes it impossible for standard autoscalers to calculate exactly how many additional replicas are required to handle waiting users.

By shifting the queue to the EPP Gateway extension, the Flow Control Queue Depth becomes the definitive "True Demand" metric. Binding your custom metric autoscalers (such as the Kubernetes HPA or external scale-to-zero controllers like KEDA) to the EPP's Flow Control metrics (specifically the `inference_extension_flow_control_queue_size` or `inference_extension_flow_control_request_queue_duration_seconds` metrics detailed in the Observability guide) allows the cluster to scale out based on the exact volume of traffic waiting to be served, completely independent of the endpoints' current hardware states.

Furthermore, because the EPP safely holds incoming HTTP connections in memory, you can confidently implement Scale-to-Zero architectures. The EPP will queue requests while cold-booting the first endpoint, seamlessly dispatching the traffic the moment the new model server comes online without dropping client connections.

## Observability & Next Steps

The Flow Control layer exposes Prometheus metrics to provide insights into load management and policy enforcement. Key metrics include queue lengths, dispatch rates, rejection counts, and latencies, all broken down by flow and priority. (See the [Flow Control Metrics](metrics-and-observability.md#flow-control-metrics) section in the main Observability guide).
