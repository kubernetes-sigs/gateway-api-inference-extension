# Fairness Policies

Fairness policies determine which active flow within a Priority Band is selected for the next dispatch opportunity. They manage contention between tenants or models sharing the same priority level.

This corresponds to **Tier 2** of the strict 3-Tier Dispatch Hierarchy in the Flow Control system:

1.  **Priority (Band Selection)**: Selects the highest-priority Band that has pending work.
2.  **Fairness (Flow Selection)**: **[Implemented by this package]** Determines which Flow within that band gets the next dispatch opportunity.
3.  **Ordering (Item Selection)**: Determines which Request from that specific flow's queue is dispatched.

## Architecture: The Flyweight Pattern

Fairness policies must often maintain state (e.g., Round Robin cursors) for each Priority Band they govern. To support this efficiently without creating a new plugin instance for every Band, this package uses the Flyweight pattern:

1.  **The Plugin Instance** is a Singleton containing only immutable configuration.
2.  **The State** is created via `NewState()` and stored on the Band.
3.  **The Logic** (`Pick` method) accepts the State as an argument during execution.

## Available Implementations

*   **[Round Robin](./roundrobin/README.md)** (`round-robin-fairness-policy`): Cycles through active flows one by one to guarantee no single flow can starve others.
*   **[Global Strict](./globalstrict/README.md)** (`global-strict-fairness-policy`): A greedy strategy that ignores flow boundaries and picks the absolute "best" request globally.

## Conformance Testing

This directory contains a functional test suite (`functional_test.go`) that verifies `FairnessPolicy` implementations against the expected contract.

To run the conformance tests:
```bash
go test ./pkg/epp/framework/plugins/flowcontrol/fairness/...
```

To include a new fairness policy in these tests, add its factory to the local `policies` map inside `TestFairnessPolicyConformance` within `functional_test.go`.
