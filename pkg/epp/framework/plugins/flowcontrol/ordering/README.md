# Ordering Policies

Ordering policies determine which request from a specific flow's queue is selected for dispatch when that flow is picked by a Fairness Policy.

This corresponds to **Tier 3** of the strict 3-Tier Dispatch Hierarchy in the Flow Control system:

1.  **Priority (Band Selection)**: Selects the highest-priority Band that has pending work.
2.  **Fairness (Flow Selection)**: Determines which Flow within that band gets the next dispatch opportunity.
3.  **Ordering (Item Selection)**: **[Implemented by this package]** Determines which Request from that specific flow's queue is dispatched.

## Available Implementations

*   **[First-Come, First-Served (FCFS)](./fcfs/README.md)** (`fcfs-ordering-policy`): Selects requests based on their arrival order. This is the default policy.
*   **[Earliest Deadline First (EDF)](./edf/README.md)** (`edf-ordering-policy`): Selects requests based on their absolute deadline, derived from TTL.
*   **[SLO Deadline](./slodeadline/README.md)** (`slo-deadline-ordering-policy`): Selects requests based on a deadline derived from an SLO header (e.g., target TTFT).

## Conformance Testing

This directory contains a functional test suite (`functional_test.go`) that verifies `OrderingPolicy` implementations against the expected contract.

To run the conformance tests:
```bash
go test ./pkg/epp/framework/plugins/flowcontrol/ordering/...
```

To include a new ordering policy in these tests, add its factory to the local `policies` map inside `TestOrderingPolicyConformance` within `functional_test.go`.

## Related Documentation

*   [Flow Control User Guide](../../../../../../site-src/guides/flow-control.md)
*   [Fairness Overview](../fairness/README.md)
