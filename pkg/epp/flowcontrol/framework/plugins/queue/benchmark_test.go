/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package queue_test

import (
	"fmt"
	"sync"
	"testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

var benchmarkFlowKey = types.FlowKey{ID: "benchmark-flow"}

// BenchmarkQueues runs a series of benchmarks against all registered queue implementations.
func BenchmarkQueues(b *testing.B) {
	for queueName, constructor := range queue.RegisteredQueues {
		b.Run(string(queueName), func(b *testing.B) {
			// All queue implementations must support the default enqueue time comparator.
			q, err := constructor(enqueueTimeComparator)
			if err != nil {
				b.Fatalf("Failed to construct queue '%s': %v", queueName, err)
			}

			b.Run("AddRemove", func(b *testing.B) {
				benchmarkAddRemove(b, q)
			})

			b.Run("AddPeekRemove", func(b *testing.B) {
				benchmarkAddPeekRemove(b, q)
			})

			b.Run("AddPeekTailRemove", func(b *testing.B) {
				benchmarkAddPeekTailRemove(b, q)
			})

			b.Run("BulkAddThenBulkRemove", func(b *testing.B) {
				benchmarkBulkAddThenBulkRemove(b, q)
			})

			b.Run("HighContention", func(b *testing.B) {
				benchmarkHighContention(b, q)
			})
		})
	}
}

// benchmarkAddRemove measures the throughput of tightly coupled Add and Remove operations in parallel. This is a good
// measure of the base overhead of the queue's data structure and locking mechanism.
func benchmarkAddRemove(b *testing.B, q framework.SafeQueue) {
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			item := mocks.NewMockQueueItemAccessor(1, "item", benchmarkFlowKey)
			err := q.Add(item)
			if err != nil {
				b.Fatalf("Add failed: %v", err)
			}
			_, err = q.Remove(item.Handle())
			if err != nil {
				b.Fatalf("Remove failed: %v", err)
			}
		}
	})
}

// benchmarkAddPeekRemove measures the throughput of a serial Add, PeekHead, and Remove sequence. This simulates a
// common consumer pattern where a single worker peeks at an item before deciding to process and remove it.
func benchmarkAddPeekRemove(b *testing.B, q framework.SafeQueue) {
	// Pre-add one item so PeekHead doesn't fail on the first iteration.
	initialItem := mocks.NewMockQueueItemAccessor(1, "initial", benchmarkFlowKey)
	if err := q.Add(initialItem); err != nil {
		b.Fatalf("Failed to add initial item: %v", err)
	}

	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		item := mocks.NewMockQueueItemAccessor(1, "item", benchmarkFlowKey)
		err := q.Add(item)
		if err != nil {
			b.Fatalf("Add failed: %v", err)
		}

		peeked, err := q.PeekHead()
		if err != nil {
			// In a concurrent benchmark, this could happen if the queue becomes empty.
			// In a serial one, it's a fatal error.
			b.Fatalf("PeekHead failed: %v", err)
		}

		_, err = q.Remove(peeked.Handle())
		if err != nil {
			b.Fatalf("Remove failed: %v", err)
		}
	}
}

// benchmarkBulkAddThenBulkRemove measures performance of filling the queue up with a batch of items and then draining
// it. This can reveal performance characteristics related to how the data structure grows and shrinks.
func benchmarkBulkAddThenBulkRemove(b *testing.B, q framework.SafeQueue) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Add a batch of items
		items := make([]types.QueueItemAccessor, 100)
		for j := range items {
			item := mocks.NewMockQueueItemAccessor(1, fmt.Sprintf("bulk-%d-%d", i, j), benchmarkFlowKey)
			items[j] = item
			if err := q.Add(item); err != nil {
				b.Fatalf("Add failed: %v", err)
			}
		}

		// Remove the same number of items
		for range items {
			peeked, err := q.PeekHead()
			if err != nil {
				b.Fatalf("PeekHead failed: %v", err)
			}
			if _, err := q.Remove(peeked.Handle()); err != nil {
				b.Fatalf("Remove failed: %v", err)
			}
		}
	}
}

// benchmarkAddPeekTailRemove measures the throughput of a serial Add, PeekTail, and Remove sequence. This is useful for
// understanding the performance of accessing the lowest-priority item.
func benchmarkAddPeekTailRemove(b *testing.B, q framework.SafeQueue) {
	// Pre-add one item so PeekTail doesn't fail on the first iteration.
	initialItem := mocks.NewMockQueueItemAccessor(1, "initial", benchmarkFlowKey)
	if err := q.Add(initialItem); err != nil {
		b.Fatalf("Failed to add initial item: %v", err)
	}

	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		item := mocks.NewMockQueueItemAccessor(1, "item", benchmarkFlowKey)
		err := q.Add(item)
		if err != nil {
			b.Fatalf("Add failed: %v", err)
		}

		peeked, err := q.PeekTail()
		if err != nil {
			b.Fatalf("PeekTail failed: %v", err)
		}

		_, err = q.Remove(peeked.Handle())
		if err != nil {
			b.Fatalf("Remove failed: %v", err)
		}
	}
}

// benchmarkHighContention simulates a more realistic workload with multiple producers and consumers operating on the
// queue concurrently.
func benchmarkHighContention(b *testing.B, q framework.SafeQueue) {
	// Pre-fill the queue to ensure consumers have work to do immediately.
	for i := range 1000 {
		item := mocks.NewMockQueueItemAccessor(1, fmt.Sprintf("prefill-%d", i), benchmarkFlowKey)
		if err := q.Add(item); err != nil {
			b.Fatalf("Failed to pre-fill queue: %v", err)
		}
	}

	stopCh := make(chan struct{})
	var wgProducers sync.WaitGroup

	// Start producer goroutines to run in the background.
	for range 4 {
		wgProducers.Add(1)
		go func() {
			defer wgProducers.Done()
			for {
				select {
				case <-stopCh:
					return
				default:
					item := mocks.NewMockQueueItemAccessor(1, "item", benchmarkFlowKey)
					_ = q.Add(item)
				}
			}
		}()
	}

	b.ReportAllocs()
	b.ResetTimer()

	// Consumers drive the benchmark.
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			peeked, err := q.PeekHead()
			if err == nil {
				_, _ = q.Remove(peeked.Handle())
			}
		}
	})

	b.StopTimer()
	close(stopCh) // Signal producers to stop.
	wgProducers.Wait()
}
