# Latency SLO Admitter (`latency-slo-admitter`)

Rejects sheddable requests when no endpoint can meet latency SLO constraints.

## Interface

AdmissionPlugin

## Behavior

Non-sheddable requests (priority >= 0) always bypass admission.

For sheddable requests (priority < 0), the plugin admits if **any** of these conditions
are met:

| Condition | What it checks |
|-----------|---------------|
| **hasValid** | At least one endpoint has valid predictions meeting SLOs |
| **hasIdle** | At least one endpoint has zero dispatched requests |
| **hasCold** | At least one endpoint has <2% KV cache (predictions unreliable) |

If none are met, the request is rejected.

## Config

None. The plugin reads prediction validity directly from `LatencyPredictionInfo` endpoint
attributes set by the `predicted-latency-producer` plugin. No `StreamingMode` config needed
because the predictor already neutralizes TPOT for non-streaming mode and prefill endpoints.

## Dependencies

- Requires `predicted-latency-producer` to run first (in PrepareRequestData) to populate
  `LatencyPredictionInfo` attributes on endpoints.
- Reads `DispatchedRequestCount` from the same attributes for idle detection.
- Reads `KVCacheUsagePercent` from endpoint metrics for cold detection.
