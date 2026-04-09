# First-Come, First-Served (FCFS) Ordering Policy

It is registered as type `fcfs-ordering-policy` and runs as an ordering policy.

The First-Come, First-Served (FCFS) ordering policy selects requests based on their arrival order at the Flow Control layer.

## Why Choose This Policy?

- **Simplicity and Intuitive Behavior:** Requests are processed in the order they arrive, making the system behavior easy to understand and predict.
- **No Special Inputs Required:** Unlike policies that rely on deadlines or priorities, FCFS works without any additional request metadata or headers.
- **Good for Homogeneous Workloads:** When all requests have similar processing costs and importance, FCFS is often the fairest approach.

## What It Does

The FCFS policy compares requests by their **Logical Enqueue Time**—the timestamp when the request first arrived at the `controller.FlowController`. The request with the earliest timestamp is rendered "less" than (i.e., higher priority than) requests that arrived later.

## Inputs consumed

This policy inspects the following attributes of the request item:
- **Logical Enqueue Time**: Used as the arrival timestamp to determine order.

## Behavior and Queue Pairing

The behavioral guarantees of this policy are critically dependent on the capabilities of the `SafeQueue` it is paired with in the configuration.

1. **Strict FCFS (High Accuracy, Lower Throughput):**
   - Paired with a `CapabilityPriorityConfigurable` queue (e.g., a heap-based queue).
   - Requests are strictly ordered by their logical enqueue time.
2. **Approximate FCFS (High Throughput, Lower Accuracy):**
   - Paired with a `CapabilityFIFO` queue (e.g., a list-based queue).
   - Requests are ordered by their *physical* arrival time at the specific queue shard.
   - This is the **default** behavior because it avoids the overhead of maintaining a min-heap across shards, yielding better performance for high-throughput workloads.

## Configuration

This policy does not require any custom parameters.

```yaml
orderingPolicyRef: fcfs-ordering-policy
```

## Trade-offs

- **Head-of-Line Blocking:** A large request at the front of the queue can delay smaller, more urgent requests behind it.
- **No Urgency Awareness:** It does not consider deadlines or request priority; a request that has already timed out on the client side might still be processed if it was the oldest in the queue.

## Related Documentation

*   [Ordering Overview](../README.md)
*   [Flow Control User Guide](../../../../../../../site-src/guides/flow-control.md)
