# SLO Deadline Ordering Policy

It is registered as type `slo-deadline-ordering-policy` and runs as an ordering policy.

The SLO Deadline ordering policy selects requests based on a deadline derived from a Service Level Objective (SLO) specified in the request headers.

## Why Choose This Policy?

- **Header-Driven SLOs:** Ideal for systems where clients or upstream proxies specify latency targets (e.g., Time-To-First-Token) dynamically per request.
- **Dynamic Prioritization:** Allows prioritizing urgent requests (with tight deadlines) over less urgent ones on the fly.
- **Maximizes Goodput:** By prioritizing requests closest to their SLO deadline, it helps maximize the number of requests that successfully meet their latency targets.
- **Graceful Degradation:** Requests without the SLO header are still processed but are yielded to requests that have explicit SLO targets.

## What It Does

The policy computes a deadline for each request as:
`Deadline = ReceivedTimestamp + x-slo-ttft-ms`

- **`ReceivedTimestamp`**: The time the request was received by the gateway.
- **`x-slo-ttft-ms`**: A request header specifying the target Time-To-First-Token in milliseconds.

## Inputs consumed

This policy inspects the following attributes of the request:
- **ReceivedTimestamp**: The time the request was received by the gateway.
- **`x-slo-ttft-ms` Header**: A header specifying the target Time-To-First-Token in milliseconds.

Requests are prioritized as follows:
1. Requests with **earlier absolute deadlines** are dispatched first.
2. If two requests have the same deadline, or if both lack a valid deadline, they are ordered by `ReceivedTimestamp` (FCFS).
3. Requests without a valid `x-slo-ttft-ms` header (missing, empty, or invalid integer) are assigned a far-future deadline, placing them behind all SLO-bound requests.

## Behavior and Queue Pairing

This policy **requires** specific queue capabilities to function correctly.

- **Required Capability:** `CapabilityPriorityConfigurable` (e.g., a heap-based priority queue).
- This policy cannot be paired with a simple FIFO list queue because it must maintain items in a sorted order based on dynamic deadlines.

## Configuration

This policy does not require any custom parameters in the flow control configuration itself, but it relies on the presence of the `x-slo-ttft-ms` header in incoming requests.

```yaml
orderingPolicyRef: slo-deadline-ordering-policy
```

## Trade-offs

- **Header Dependency:** Requires clients or upstream components to correctly set the `x-slo-ttft-ms` header.
- **Starvation Risk:** Non-SLO requests or requests with very loose SLOs may be starved if there is a constant influx of tight-SLO requests.
- **Computational Overhead:** Similar to EDF, maintaining a priority heap incurs higher CPU overhead ($O(\log n)$) than a simple FIFO list.

## Related Documentation

*   [Ordering Overview](../README.md)
*   [Flow Control User Guide](../../../../../../../site-src/guides/flow-control.md)
