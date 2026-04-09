# Global Strict Fairness Policy

The Global Strict fairness policy is a greedy strategy that operates across all flows within a Priority Band. By ignoring flow boundaries and selecting the flow with the "best" item according to the configured Ordering Policy, this policy effectively transforms the multi-queue Priority Band into a single logical queue.

It is registered as type `global-strict-fairness-policy` and runs as a fairness policy.

## Why choose this policy?

While this policy does not provide fairness between tenants, it is the ideal choice when:
*   **Global Ordering is Paramount**: You want to strictly enforce arrival order (FCFS) or deadline order (EDF) across *all* requests in a priority band, regardless of which tenant sent them.
*   **No Multi-Tenancy Contention**: You do not need to protect against "noisy neighbors" within a priority level, or you handle isolation at a different layer.
*   **Maximum Performance**: It has the lowest overhead as it is stateless and does not need to manage cycles or cursors across flows.

## What it does

*   **Fairness**: None. A single flow with high volume can starve others.
*   **Ordering**: Delegated. It relies on the `OrderingPolicy` associated with the queues to determine which item is "best". It does not enforce FIFO by itself.
*   **Stateless**: This policy does not maintain any mutable state across dispatch attempts.

## Unit of Fairness

**None**. This policy does not attempt to provide fairness.

## Inputs consumed

This policy inspects the state of the queues and their items:
*   **Queues**: Accesses all active queues in the band via `PriorityBandAccessor`.
*   **Head Items**: Peeks at the head item of each queue to find the globally "best" candidate.
*   **Ordering Policy**: Invokes the `OrderingPolicy.Less` method to compare items across flows.

## Configuration

This policy does not require any additional configuration parameters beyond its type registration.

```yaml
fairnessPolicyRef: global-strict-fairness-policy
```

## Requirements

All flows in the band **MUST** use compatible `OrderingPolicy` types (i.e., identical score types). If incompatible policies are detected, the policy will return an error.

## Trade-offs

*   **Starvation Risk**: High. Flushes traffic based solely on item ordering, ignoring tenant/flow isolation.
*   **Performance**: High efficiency due to statelessness and lack of cycle management.

## Related Documentation

*   [Fairness Overview](../README.md)
*   [Flow Control User Guide](../../../../../../../site-src/guides/flow-control.md)
