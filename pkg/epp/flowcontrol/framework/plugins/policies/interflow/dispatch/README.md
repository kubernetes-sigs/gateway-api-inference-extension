# Flow Controller Inter-Flow Dispatch Policy Plugins

This directory contains concrete implementations of the [`framework.InterFlowDispatchPolicy`](../../../policies.go)
interface. These policies are responsible for determining *which flow's queue* gets the next opportunity to dispatch a
request to the scheduler.

## Overview

The `controller.FlowController` uses a two-tier policy system to manage requests. `framework.InterFlowDispatchPolicy`
plugins represent the second tier, making strategic decisions about fairness *between* different logical flows (e.g.,
between different tenants or models).

This contrasts with the `framework.IntraFlowDispatchPolicy`, which is responsible for **temporal scheduling**: deciding
the order of requests *within* a single flow's queue after it has been selected by the inter-flow policy.

Key responsibilities and characteristics of a `framework.InterFlowDispatchPolicy`:

1.  **Queue Selection (`SelectQueue`)**: The primary method, `SelectQueue(band framework.PriorityBandAccessor)`,
    inspects a set of queues within a single priority level (a "band") and decides which queue, if any, should be
    selected to dispatch a request from next.

2.  **Fairness Across Flows**: The core purpose of this policy is to enforce a fairness doctrine across multiple
    competing flows. This could be simple round-robin, or more complex weighted fairness schemes.

3.  **Stateless vs. Stateful**: Policies can be stateless (like `besthead`, which makes a decision based only on the
    current state of the queues) or stateful (like `roundrobin`, which needs to remember which queue it selected last).
    Any state must be managed in a goroutine-safe manner.

The `framework.InterFlowDispatchPolicy` is critical for multi-tenancy and preventing any single high-traffic flow from
starving all others.

## Contributing a New `framework.InterFlowDispatchPolicy` Implementation

To contribute a new dispatch policy implementation, follow these steps:

1.  **Define Your Implementation**
    - Create a new Go package in a subdirectory (e.g., `mycustompolicy/`).
    - Implement the `framework.InterFlowDispatchPolicy` interface.
    - Ensure all methods are goroutine-safe if your policy maintains any internal state.

2.  **Register Your Policy**
    - In an `init()` function within your policy's Go file, call [`MustRegisterPolicy()`](./factory.go) with a unique
      name and a constructor function that matches the `PolicyConstructor` signature.

3.  **Add to the Functional Test**
    - Add a blank import for your new package to [`functional_test.go`](./functional_test.go). Your policy will then
      be automatically included in the functional test suite, which validates the basic
      `framework.InterFlowDispatchPolicy` contract (e.g., correct initialization, handling of nil/empty bands).

4.  **Add Policy-Specific Tests**
    - The functional test suite only validates the universal contract. You MUST add a separate `_test.go` file within
      your package to test the specific logic of your policy.
    - For example, your tests should validate that `SelectQueue()` correctly implements your desired selection logic
      (e.g., that round-robin correctly cycles through queues).

5.  **Documentation**
    - Add a package-level GoDoc comment to your new policy's Go file, explaining its behavior and any trade-offs.