# Flow Control Core Types

This package defines the fundamental data structures, interfaces, and errors that form the vocabulary of the Flow
Control system. It establishes the core concepts of the request lifecycle and its final, reportable outcomes.

## Request Lifecycle Interfaces

A request's journey through the Flow Controller is represented by a series of interfaces that define its state as it
moves through the system:

1.  **`FlowControlRequest`**: The initial, "raw" contract for an incoming request. It carries the essential data
    provided by the client, such as its `FlowID` and `ByteSize`.
2.  **`QueueItemAccessor`**: The internal, enriched, and read-only view of a request once it has been accepted by the
    controller. This interface is the primary means by which policy plugins inspect items.
3.  **`QueueItemHandle`**: An opaque, queue-specific handle to a queued item. The controller uses this handle to perform
    targeted operations, such as removing a specific item, without needing to know the queue's internal implementation
    details.

## Final State Reporting: Outcomes and Errors

The final state of every request is reported using a combination of a `QueueOutcome` enum and a corresponding `error`.
This provides a clear, machine-inspectable way to understand the result.

* **`QueueOutcome`**: A concise enum summarizing the final result (e.g., `QueueOutcomeDispatched`,
  `QueueOutcomeRejectedCapacity`, `QueueOutcomeEvictedDisplaced`). This is ideal for metrics.

* **Errors**: For any non-dispatch outcome, a specific sentinel error is returned. These are nested to provide detailed
  context:
    * `ErrRejected`: The parent error for any request rejected *before* being enqueued.
    * `ErrEvicted`: The parent error for any request removed *after* being enqueued for reasons other than dispatch.

Callers of `FlowController.EnqueueAndWait()` can first use `errors.Is()` to check for the general class of failure
(`ErrRejected` or `ErrEvicted`), and then unwrap the error to find the specific cause (e.g., `ErrQueueAtCapacity`).
