# Earliest Deadline First (EDF) Ordering Policy

It is registered as type `edf-ordering-policy` and runs as an ordering policy.

The Earliest Deadline First (EDF) ordering policy selects requests based on their absolute deadline, prioritizing requests that are closest to expiring.

## Why Choose This Policy?

- **SLO Awareness:** Best suited for workloads with strict latency Service Level Objectives (SLOs) or Time-To-Live (TTL) requirements.
- **Minimizes Timeout Rates:** By prioritizing requests that are about to expire, it helps maximize the number of requests completed before their deadline.
- **Graceful Degradation:** Requests without a deadline are still processed but are yielded to time-sensitive requests.

## What It Does

The EDF policy computes an absolute deadline for each request as:
`Deadline = EnqueueTime + EffectiveTTL`

- Requests with **earlier absolute deadlines** are prioritized.
- If two requests have the same deadline, FCFS (First-Come, First-Served) is used as a tie-breaker based on logical enqueue time.
- Requests without a valid TTL (i.e., `EffectiveTTL <= 0`) are treated as having no deadline and are assigned a far-future deadline, ensuring they are scheduled after all time-bound requests.

## Inputs consumed

This policy inspects the following attributes of the request item:
- **Logical Enqueue Time**: Used as the arrival timestamp and as a tie-breaker.
- **Effective TTL**: Used to calculate the absolute deadline.

## Behavior and Queue Pairing

Unlike FCFS, this policy **requires** specific queue capabilities to function correctly.

- **Required Capability:** `CapabilityPriorityConfigurable` (e.g., a heap-based priority queue).
- This policy cannot be paired with a simple FIFO list queue because it must maintain items in a sorted order that changes as new items arrive with different deadlines.

## Configuration

This policy does not require any custom parameters.

```yaml
orderingPolicyRef: edf-ordering-policy
```

## Trade-offs

- **Starvation Risk:** If there is a continuous stream of requests with tight deadlines, requests with loose or no deadlines may be starved and never processed.
- **Computational Overhead:** Maintaining a priority heap incurs higher CPU overhead ($O(\log n)$ for insertions and deletions) compared to a simple FIFO list ($O(1)$).

## Related Documentation

*   [Ordering Overview](../README.md)
*   [Flow Control User Guide](../../../../../../../site-src/guides/flow-control.md)
