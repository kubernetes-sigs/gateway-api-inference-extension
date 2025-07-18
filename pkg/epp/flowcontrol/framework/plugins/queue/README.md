# Flow Controller Queue Plugins

This directory contains concrete implementations of the [`framework.SafeQueue`](../../queue.go) interface. This contract
defines core, self-contained queue data structures used by the `controller.FlowController`.

## Overview

The `controller.FlowController` manages requests by organizing them into queues. Each logical "flow" within a given
priority band has its own `contracts.ManagedQueue` instance, which wraps a `framework.SafeQueue`. This design allows the
`controller.FlowController` to apply policies at both the inter-flow (across different flows) and intra-flow (within a
single flow's queue) levels.

The `framework.SafeQueue` interface abstracts the underlying data structure and its ordering logic. This pluggable
design allows for:

- **Different Queuing Disciplines**: A basic FIFO queue ([`listqueue`](./listqueue/)) is provided, but other disciplines
  like priority queues ([`maxminheap`](./maxminheap/)) can be used for more complex ordering requirements.
- **Specialized Capabilities**: Policies can declare `RequiredQueueCapabilities()` (e.g., `framework.CapabilityFIFO` or
  `framework.CapabilityPriorityConfigurable`). The `contracts.FlowRegistry` pairs the policy with a queue that provides
  the necessary capabilities.
- **Performance Optimization**: Different queue implementations offer varying performance characteristics, which can be
  compared using the centralized benchmark suite to select the best fit for a given workload.

## Contributing a New `framework.SafeQueue` Implementation

To contribute a new queue implementation, follow these steps:

1.  **Define Your Implementation**
    - Create a new Go package in a subdirectory (e.g., `mycustomqueue/`).
    - Implement the `framework.SafeQueue` and `types.QueueItemHandle` interfaces.
    - Ensure all methods of `framework.SafeQueue` are goroutine-safe, typically by using a `sync.Mutex` or
      `sync.RWMutex`.
    - If your queue declares `framework.CapabilityPriorityConfigurable`, it MUST use the
      [`framework.ItemComparator`](../../policies.go) passed to its constructor for all internal ordering logic.

2. **Register Your Queue**
    - In an `init()` function within your queue's Go file, call [`queue.MustRegisterQueue()`](./factory.go) with a
      unique name and a constructor function that matches the `queue.QueueConstructor` signature.

3.  **Add to the Functional Test**
    - Add a blank import for your new package to [`functional_test.go`](./functional_test.go). Your queue will then be
      automatically included in the functional test suite, which validates the `framework.SafeQueue` contract.

4.  **Documentation**
    - Add GoDoc comments to your new queue type, explaining its behavior, capabilities, and any trade-offs.

5.  **Benchmarking**
    - You do not need to write custom benchmarks. The centralized suite in [`benchmark_test.go`](./benchmark_test.go)
      automatically includes any new queue implementation after it is registered. This ensures all queues are compared
      fairly under the same conditions.

## Benchmarking Strategy and Results

A centralized benchmark suite runs against all registered `framework.SafeQueue` implementations to provide a consistent
performance comparison. To run the benchmarks, use the following command:

```sh
go test -bench=. -benchmem ./pkg/epp/flowcontrol/framework/plugins/queue/...
```

### Benchmark Scenarios

The suite includes the following scenarios:

- **`AddRemove`**: Measures throughput of tightly coupled `Add` and `Remove` operations under high parallelism. This
  tests the raw overhead of the data structure and its locking mechanism for simple, transactional workloads.
- **`AddPeekRemove`**: Measures performance of a sequential `Add` -> `PeekHead` -> `Remove` loop. This simulates a
  common consumer pattern where a single worker inspects an item before processing it.
- **`BulkAddThenBulkRemove`**: Tests performance of adding a large batch of items and then removing them all. This can
  reveal how the data structure's performance changes as it grows and shrinks under load.
- **`HighContention`**: Simulates a realistic workload with multiple concurrent producers (adding items) and consumers
  (peeking and removing items) operating on the same queue.

### Latest Results

*Last Updated: Commit `35a9d6c`*
*(CPU: AMD EPYC 7B12)*

| Benchmark                   | Implementation | Iterations | ns/op   | B/op  | allocs/op |
| --------------------------- | -------------- | ---------- | ------- | ----- | --------- |
| **AddRemove**               | `ListQueue`    | 1,906,153  | 595.5   | 224   | 5         |
|                             | `MaxMinHeap`   | 1,763,473  | 668.9   | 184   | 4         |
| **AddPeekRemove**           | `ListQueue`    | 3,547,653  | 298.5   | 224   | 5         |
|                             | `MaxMinHeap`   | 1,986,780  | 751.5   | 184   | 4         |
| **AddPeekTailRemove**       | `ListQueue`    | 3,732,302  | 303.3   | 224   | 5         |
|                             | `MaxMinHeap`   | 2,006,383  | 551.6   | 184   | 4         |
| **BulkAddThenBulkRemove**   | `ListQueue`    | 24,046     | 47,240  | 24800 | 698       |
|                             | `MaxMinHeap`   | 9,410      | 110,929 | 20786 | 597       |
| **HighContention**          | `ListQueue`    | 21,283,537 | 47.53   | 11    | 0         |
|                             | `MaxMinHeap`   | 16,953,121 | 74.09   | 4     | 0         |

### Interpretation of Results

The benchmark results highlight the trade-offs between the different queue implementations based on their underlying
data structures:

- **`ListQueue`**: As a linked list, it excels in scenarios involving frequent additions or removals from either end of
  the queue (`AddPeekRemove`, `AddPeekTailRemove`), which are O(1) operations. The `HighContention` benchmark shows that
  its simple, low-overhead operations are also extremely performant for consumer throughput even under heavy concurrent
  load.
- **`MaxMinHeap`**: As a slice-based heap, it has a lower allocation overhead per operation, making it efficient for
  high-throughput `AddRemove` cycles. Peeking and removing items involves maintaining the heap property, which has an
  O(log n) cost, making individual peek operations slower than `ListQueue`.

**Choosing a Queue:**

The data suggests the following guidance:
- For simple **FIFO** workloads where the primary operations are consuming from the head, `ListQueue` is a strong and
  simple choice.
- For workloads requiring **priority-based ordering** or those that are sensitive to allocation overhead under high
  contention, `MaxMinHeap` is likely the more suitable option.

These benchmarks provide a baseline for performance. The best choice for a specific use case will depend on the expected
workload patterns.
