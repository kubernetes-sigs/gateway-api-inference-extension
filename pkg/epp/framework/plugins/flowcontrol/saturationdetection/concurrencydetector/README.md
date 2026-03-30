# Concurrency Detector Plugin

Synchronous saturation detection and scheduling filter mechanism based on active in-flight request accounting.

It is registered as type `concurrency-detector` and runs as a saturation detector and scheduling filter.

## What it does

This plugin uses a two-tier approach to manage average pool load and protect individual endpoints.

### Role in Flow Control (The Gatekeeper)
The detector implements the `SaturationDetector` interface to provide a utilization gradient, allowing the Flow Controller to apply proportional backpressure.

    PoolSaturation = Aggregate Inflight Load / Aggregate Pool Capacity

In token mode, both numerator and denominator are evaluated in tokens: the aggregate inflight token count divided by the sum of all endpoints' MaxTokenConcurrency.

**Heterogeneous Deployments:** Because this detector calculates saturation globally as a single aggregate fraction, it utilizes an aggregate queueing model. In deployments with heterogeneous compute (e.g., mixing H100 and L4 nodes), this heavily biases the pool saturation metric toward the state of the larger nodes. Contrast this with the Utilization Detector, which evaluates saturation as an unweighted average of individual endpoint scores.

### Role in Scheduling (The Traffic Shaper)
The detector implements the `Filter` interface to protect individual endpoints. It removes endpoints from candidate lists if their local inflight count exceeds the safety limit:

    EndpointLimit = Capacity * (1 + Headroom)

This approach allows the Flow Controller to manage average pool load, while the Scheduler retains the flexibility to burst above ideal targets (the "Headroom") to satisfy affinity or scoring objectives.

## Inputs consumed

The plugin internally tracks active concurrency by hooking into the request lifecycle (`PreRequest` and `ResponseBody`).
- **Requests Mode**: Maintains atomic counters of active requests per endpoint.
- **Tokens Mode**: Uses a `TokenEstimator` to estimate tokens from the incoming request and tracks the aggregate inflight tokens per endpoint.

## Configuration

The plugin accepts JSON parameters decoding to the following fields:

- `concurrencyMode` (`string`): Evaluation mode. Valid values are `"requests"` or `"tokens"`. (Default: `"requests"`)
- `maxConcurrency` (`int64`): Maximum requests in flight. Serves as the "ideal" request capacity for a single endpoint. Must be > 0. (Default: `100`)
- `maxTokenConcurrency` (`int64`): Maximum tokens in flight. The "tokens" mode equivalent of `maxConcurrency`. Must be > 0. (Default: `1000000`)
- `headroom` (`float64`): Allowed burst capacity above the ideal threshold, expressed as a fraction (e.g., `0.2` for 20%). Must be >= 0.0. (Default: `0.0`)

## Trade-offs

Unlike the Utilization Detector, this approach reacts instantaneously to new requests, preventing sudden bursts from overwhelming an endpoint before telemetry updates. However, it suffers from two critical flaws:

1. **Open-Loop Divergence**: The detector operates as an open-loop controller, completely blind to actual hardware telemetry. While the internal counters are mathematically zero-sum and do not leak, consistent under- or over-estimations of token lengths will cause the view of pool saturation to systematically drift from the physical reality of the GPU workload.
2. **KV Cache Blindness**: Because the detector cannot observe true engine memory pressure, it is highly vulnerable to continuous-batching edge cases. If actual output generations exceed static estimates, the underlying KV cache will silently fill up. This forces the inference engine to preempt active requests and swap KV blocks to CPU memory, causing severe latency degradation (TPOT spikes) that remains completely invisible to this detector.
