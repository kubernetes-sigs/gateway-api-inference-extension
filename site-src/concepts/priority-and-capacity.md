# Priority and Capacity

The `InferenceObjective` API resource allows users to define the priority of their inference requests. This priority heavily influences how the EPP handles requests under load, with behavior changing significantly depending on whether the [Flow Control layer](../guides/flow-control.md) is enabled.

## Defining Priority

Priority is an integer value defined in the `InferenceObjective.spec.priority` field.

- **Higher values** indicate higher priority.
- **Negative values** are allowed and designate requests as "sheddable" (safe to drop under heavy load).
- If a request does not have an associated `InferenceObjective` or the priority field is unset, it defaults to `0`.

## Admission Control Behaviors

The Endpoint Picker (EPP) employs a **Saturation Detector** to assess the load on the InferencePool (monitoring aggregate metrics like KV cache utilization and local queue depth).

How Priority interacts with Capacity depends on your EPP configuration:

### 1. With Flow Control Enabled
*Recommended Approach.* Enable Flow Control by adding the `flowControl` feature gate to your `EndpointPickerConfig`. Details are covered in the [Flow Control Layer User Guide](../guides/flow-control.md).

* **Intelligent Queuing:** All requests, including those with negative priority, are admitted into the Flow Control layer's internal queues, subject to dynamic priority bands and global HTTP payload capacity limits. If an inbound request would exceed these configured payload capacities, it is immediately rejected (HTTP 503). This enables efficient load shedding by limiting the aggregate backlog size before it impacts the proxy's memory.
* **Strict Priority Dispatch:** The Flow Control layer dispatches requests from its queues with strict priority. Higher-priority requests are *always* dispatched before lower-priority ones.
* **Flow Control Backpressure:** When the pool is saturated, the EPP enforces **Strict Priority Preservation**. The dispatch cycle is paused, holding the highest-priority request safely in memory until the pool recovers. Lower-priority requests are intentionally held back to guarantee they do not steal the exact GPU cycles the high-priority request is waiting for.
* **Fairness:** Among requests of the same priority, fairness policies ensure equitable access to resources during contention.

### 2. With Flow Control Disabled (Legacy Admission)
*Default Behavior.*

* **Load Shedding:** If the pool is saturated, requests with **negative priority** are immediately rejected (HTTP 503). Requests with a priority >= 0 are *not* shed by the EPP.
* **No EPP-Level Queuing:** All non-sheddable requests are passed directly to the Scheduling layer. Any necessary queuing happens directly on the model server backends, which are generally unaware of global priority.
* **Limited Priority Enforcement:** Priority is only used for the binary sheddable/non-sheddable decision. There is no EPP-level mechanism to enforce finer-grained priority ordering or fairness. All non-sheddable requests are handled strictly first-come, first-served (FCFS).
