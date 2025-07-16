# Flow Controller Intra-Flow Dispatch Policy Plugins

This directory contains concrete implementations of the [`framework.IntraFlowDispatchPolicy`](../../../policies.go)
interface. These policies are responsible for **temporal scheduling**: determining the order in which requests are
selected for dispatch *from within a single flow's queue*.

## Overview

The `controller.FlowController` uses a two-tier policy system to manage requests. `framework.IntraFlowDispatchPolicy`
plugins represent the first tier, making tactical decisions about the ordering of requests *within* a single logical
flow (e.g., for a specific model or tenant).

This contrasts with the `framework.InterFlowDispatchPolicy`, which is responsible for deciding *which flow's queue*
gets the next opportunity to dispatch a request. The `framework.IntraFlowDispatchPolicy` only operates *after* the
inter-flow policy has selected a specific queue.

Key responsibilities and characteristics of a `framework.IntraFlowDispatchPolicy`:

1.  **Request Selection (`SelectItem`)**: The primary method, `SelectItem(queue framework.FlowQueueAccessor)`, inspects
    the given flow's queue (via a read-only accessor) and decides which item, if any, should be dispatched next from
    *that specific queue*.

2.  **Priority Definition (`ItemComparator`)**:
    - This policy type is unique because it defines the nature of priority for items *within its specific managed
      queue*. It makes this logic explicit by vending a [`framework.ItemComparator`](../../../policies.go).
    - The vended comparator defines the "less than" relationship between two items and exposes a `ScoreType()` string
      (e.g., `"enqueue_time_ns_asc"`, `"slo_deadline_urgency"`) that gives a semantic meaning to the comparison.

3.  **Queue Compatibility (`RequiredQueueCapabilities`)**: The policy specifies the capabilities its associated
    [`framework.SafeQueue`](../../../queue.go) must support for it to function correctly. For example, a simple FCFS
    policy would require `framework.CapabilityFIFO`, while a more complex, priority-based policy would require
    `framework.CapabilityPriorityConfigurable`. The `contracts.FlowRegistry` uses this information to pair policies with
    compatible queues.

The `framework.IntraFlowDispatchPolicy` allows for fine-grained control over how individual requests within a single flow are
serviced, enabling strategies like basic FCFS or more advanced schemes based on SLOs or deadlines.

## Contributing a New `framework.IntraFlowDispatchPolicy` Implementation

To contribute a new dispatch policy implementation, follow these steps:

1.  **Define Your Implementation**
    - Create a new Go package in a subdirectory (e.g., `mycustompolicy/`).
    - Implement the `framework.IntraFlowDispatchPolicy` interface.
    - Ensure all methods are goroutine-safe if your policy maintains any internal state.

2.  **Register Your Policy**
    - In an `init()` function within your policy's Go file, call [`MustRegisterPolicy()`](./factory.go) with a
      unique name and a constructor function that matches the `PolicyConstructor` signature.

3.  **Add to the Functional Test**
    - Add a blank import for your new package to [`functional_test.go`](./functional_test.go). Your policy will then
      be automatically included in the functional test suite, which validates the basic
      `framework.IntraFlowDispatchPolicy` contract (e.g., correct initialization, handling of nil/empty queues).

4.  **Add Policy-Specific Tests**
    - The functional test suite only validates the universal contract. You MUST add a separate `_test.go` file within
      your package to test the specific logic of your policy.
    - For example, your tests should validate that your `Comparator()` works as expected and that `SelectItem()`
      correctly implements your desired selection logic for a non-empty queue.

5.  **Documentation**
    - Add a package-level GoDoc comment to your new policy's Go file, explaining its behavior and any trade-offs.