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

package registry

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	typesmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

// --- Test Harness and Mocks ---

// mqTestHarness holds all components for testing a `managedQueue`.
type mqTestHarness struct {
	t          *testing.T
	mq         *managedQueue
	propagator *mockStatsPropagator
	mockPolicy *frameworkmocks.MockIntraFlowDispatchPolicy
}

// newMockedMqHarness creates a harness that uses a mocked underlying queue.
// This is ideal for isolating and unit testing the decorator logic of `managedQueue`.
func newMockedMqHarness(t *testing.T, queue *frameworkmocks.MockSafeQueue, key types.FlowKey) *mqTestHarness {
	t.Helper()
	return newMqHarness(t, queue, key, false)
}

// newRealMqHarness creates a harness that uses a real "ListQueue" implementation.
// This is essential for integration and concurrency tests.
func newRealMqHarness(t *testing.T, key types.FlowKey) *mqTestHarness {
	t.Helper()
	q, err := queue.NewQueueFromName(queue.ListQueueName, nil)
	require.NoError(t, err, "Test setup: creating a real ListQueue implementation should not fail")
	return newMqHarness(t, q, key, false)
}

// newMqHarness is the base constructor for the test harness.
func newMqHarness(t *testing.T, queue framework.SafeQueue, key types.FlowKey, isDraining bool) *mqTestHarness {
	t.Helper()

	propagator := &mockStatsPropagator{}
	mockPolicy := &frameworkmocks.MockIntraFlowDispatchPolicy{
		ComparatorV: &frameworkmocks.MockItemComparator{},
	}

	isDrainingFunc := func() bool { return isDraining }
	mq := newManagedQueue(queue, mockPolicy, key, logr.Discard(), propagator.propagate, isDrainingFunc)
	require.NotNil(t, mq, "Test setup: newManagedQueue must return a valid instance")

	return &mqTestHarness{
		t:          t,
		mq:         mq,
		propagator: propagator,
		mockPolicy: mockPolicy,
	}
}

// setupWithItems pre-populates the queue and resets the mock propagator for focused testing.
func (h *mqTestHarness) setupWithItems(items ...types.QueueItemAccessor) {
	h.t.Helper()
	for _, item := range items {
		err := h.mq.Add(item)
		require.NoError(h.t, err, "Harness setup: failed to add initial item to the queue")
	}
	h.propagator.reset()
}

// mockStatsPropagator captures stat changes from the system under test.
type mockStatsPropagator struct {
	lenDelta      atomic.Int64
	byteSizeDelta atomic.Int64
}

func (p *mockStatsPropagator) propagate(_ int, lenDelta, byteSizeDelta int64) {
	p.lenDelta.Add(lenDelta)
	p.byteSizeDelta.Add(byteSizeDelta)
}

func (p *mockStatsPropagator) reset() {
	p.lenDelta.Store(0)
	p.byteSizeDelta.Store(0)
}

// --- Unit Tests ---

func TestManagedQueue_InitialState(t *testing.T) {
	t.Parallel()
	h := newMockedMqHarness(t, &frameworkmocks.MockSafeQueue{}, types.FlowKey{ID: "flow", Priority: 1})
	assert.Zero(t, h.mq.Len(), "A newly initialized queue must have a length of 0")
	assert.Zero(t, h.mq.ByteSize(), "A newly initialized queue must have a byte size of 0")
}

func TestManagedQueue_Add(t *testing.T) {
	t.Parallel()
	flowKey := types.FlowKey{ID: "flow", Priority: 1}

	testCases := []struct {
		name                  string
		setupMock             func(q *frameworkmocks.MockSafeQueue)
		isDraining            bool
		expectErr             bool
		expectErrIs           error // Optional
		expectedLenDelta      int64
		expectedByteSizeDelta int64
	}{
		{
			name: "ShouldSucceed_AndIncrementStats",
			setupMock: func(q *frameworkmocks.MockSafeQueue) {
				q.AddFunc = func(types.QueueItemAccessor) {}
			},
			isDraining:            false,
			expectErr:             false,
			expectedLenDelta:      1,
			expectedByteSizeDelta: 100,
		},
		{
			name:                  "ShouldFail_AndNotChangeStats_WhenQueueIsDraining",
			setupMock:             func(q *frameworkmocks.MockSafeQueue) {},
			isDraining:            true,
			expectErr:             true,
			expectErrIs:           contracts.ErrShardDraining,
			expectedLenDelta:      0,
			expectedByteSizeDelta: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			q := &frameworkmocks.MockSafeQueue{}
			h := newMqHarness(t, q, flowKey, tc.isDraining)
			item := typesmocks.NewMockQueueItemAccessor(100, "req", flowKey)
			if tc.setupMock != nil {
				tc.setupMock(q)
			}

			err := h.mq.Add(item)

			if tc.expectErr {
				require.Error(t, err, "Add operation must fail when the underlying queue returns an error")
				if tc.expectErrIs != nil {
					assert.ErrorIs(t, err, tc.expectErrIs, "The returned error was not of the expected type")
				}
			} else {
				require.NoError(t, err, "Add operation must succeed when the underlying queue accepts the item")
			}
			assert.Equal(t, tc.expectedLenDelta, h.propagator.lenDelta.Load(),
				"The propagated length delta must exactly match the change in queue size")
			assert.Equal(t, tc.expectedByteSizeDelta, h.propagator.byteSizeDelta.Load(),
				"The propagated byte size delta must exactly match the change in queue size")
		})
	}
}

func TestManagedQueue_Remove(t *testing.T) {
	t.Parallel()
	flowKey := types.FlowKey{ID: "flow", Priority: 1}

	testCases := []struct {
		name                  string
		setupMock             func(q *frameworkmocks.MockSafeQueue, item types.QueueItemAccessor)
		expectErr             bool
		expectedLenDelta      int64
		expectedByteSizeDelta int64
	}{
		{
			name: "ShouldSucceed_AndDecrementStats",
			setupMock: func(q *frameworkmocks.MockSafeQueue, item types.QueueItemAccessor) {
				q.RemoveFunc = func(_ types.QueueItemHandle) (types.QueueItemAccessor, error) {
					return item, nil
				}
			},
			expectErr:             false,
			expectedLenDelta:      -1,
			expectedByteSizeDelta: -100,
		},
		{
			name: "ShouldFail_AndNotChangeStats_WhenUnderlyingQueueFails",
			setupMock: func(q *frameworkmocks.MockSafeQueue, item types.QueueItemAccessor) {
				q.RemoveFunc = func(_ types.QueueItemHandle) (types.QueueItemAccessor, error) {
					return nil, errors.New("remove failed")
				}
			},
			expectErr:             true,
			expectedLenDelta:      0,
			expectedByteSizeDelta: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			q := &frameworkmocks.MockSafeQueue{}
			h := newMockedMqHarness(t, q, flowKey)
			item := typesmocks.NewMockQueueItemAccessor(100, "req", flowKey)
			h.setupWithItems(item)
			tc.setupMock(q, item)

			_, err := h.mq.Remove(item.Handle())

			if tc.expectErr {
				require.Error(t, err,
					"Remove operation must fail when the underlying queue returns an error (e.g., item not found)")
			} else {
				require.NoError(t, err, "Remove operation must succeed when the underlying queue successfully removes the item")
			}
			assert.Equal(t, tc.expectedLenDelta, h.propagator.lenDelta.Load(),
				"The propagated length delta must exactly match the change in queue size")
			assert.Equal(t, tc.expectedByteSizeDelta, h.propagator.byteSizeDelta.Load(),
				"The propagated byte size delta must exactly match the change in queue size")
		})
	}
}

func TestManagedQueue_Cleanup(t *testing.T) {
	t.Parallel()
	flowKey := types.FlowKey{ID: "flow", Priority: 1}

	testCases := []struct {
		name                  string
		setupMock             func(q *frameworkmocks.MockSafeQueue, items []types.QueueItemAccessor)
		expectedLenDelta      int64
		expectedByteSizeDelta int64
	}{
		{
			name: "ShouldSucceed_AndDecrementStats_WhenItemsRemoved",
			setupMock: func(q *frameworkmocks.MockSafeQueue, items []types.QueueItemAccessor) {
				q.CleanupFunc = func(_ framework.PredicateFunc) []types.QueueItemAccessor {
					return items
				}
			},
			expectedLenDelta:      -2,
			expectedByteSizeDelta: -125,
		},
		{
			name: "ShouldSucceed_AndNotChangeStats_WhenNoItemsRemoved",
			setupMock: func(q *frameworkmocks.MockSafeQueue, items []types.QueueItemAccessor) {
				q.CleanupFunc = func(_ framework.PredicateFunc) []types.QueueItemAccessor {
					return nil // Simulate no items matching predicate.
				}
			},
			expectedLenDelta:      0,
			expectedByteSizeDelta: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			q := &frameworkmocks.MockSafeQueue{}
			h := newMockedMqHarness(t, q, flowKey)
			items := []types.QueueItemAccessor{
				typesmocks.NewMockQueueItemAccessor(50, "req", flowKey),
				typesmocks.NewMockQueueItemAccessor(75, "req", flowKey),
			}
			h.setupWithItems(items...)
			tc.setupMock(q, items)
			h.mq.Cleanup(func(_ types.QueueItemAccessor) bool { return true })
			assert.Equal(t, tc.expectedLenDelta, h.propagator.lenDelta.Load(),
				"The propagated length delta must exactly match the total number of items removed during cleanup")
			assert.Equal(t, tc.expectedByteSizeDelta, h.propagator.byteSizeDelta.Load(),
				"The propagated byte size delta must exactly match the total size of items removed during cleanup")
		})
	}
}

func TestManagedQueue_Drain(t *testing.T) {
	t.Parallel()
	flowKey := types.FlowKey{ID: "flow", Priority: 1}

	testCases := []struct {
		name                  string
		setupMock             func(q *frameworkmocks.MockSafeQueue, items []types.QueueItemAccessor)
		expectedLenDelta      int64
		expectedByteSizeDelta int64
	}{
		{
			name: "ShouldSucceed_AndDecrementStats",
			setupMock: func(q *frameworkmocks.MockSafeQueue, items []types.QueueItemAccessor) {
				q.DrainFunc = func() []types.QueueItemAccessor {
					return items
				}
			},
			expectedLenDelta:      -2,
			expectedByteSizeDelta: -125,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			q := &frameworkmocks.MockSafeQueue{}
			h := newMockedMqHarness(t, q, flowKey)
			items := []types.QueueItemAccessor{
				typesmocks.NewMockQueueItemAccessor(50, "req", flowKey),
				typesmocks.NewMockQueueItemAccessor(75, "req", flowKey),
			}
			h.setupWithItems(items...)
			tc.setupMock(q, items)

			h.mq.Drain()
			assert.Equal(t, tc.expectedLenDelta, h.propagator.lenDelta.Load(),
				"The propagated length delta must exactly match the total number of items drained")
			assert.Equal(t, tc.expectedByteSizeDelta, h.propagator.byteSizeDelta.Load(),
				"The propagated byte size delta must exactly match the total size of items drained")
		})
	}
}

func TestManagedQueue_FlowQueueAccessor(t *testing.T) {
	t.Parallel()

	t.Run("ProxiesCalls", func(t *testing.T) {
		t.Parallel()
		flowKey := types.FlowKey{ID: "flow", Priority: 1}
		q := &frameworkmocks.MockSafeQueue{}
		harness := newMockedMqHarness(t, q, flowKey)
		item := typesmocks.NewMockQueueItemAccessor(100, "req-1", flowKey)
		q.PeekHeadV = item
		q.PeekTailV = item
		q.NameV = "MockQueue"
		q.CapabilitiesV = []framework.QueueCapability{framework.CapabilityFIFO}
		require.NoError(t, harness.mq.Add(item), "Test setup: Adding an item must succeed")

		accessor := harness.mq.FlowQueueAccessor()
		require.NotNil(t, accessor, "FlowQueueAccessor must return a non-nil instance (guaranteed by contract)")

		assert.Equal(t, harness.mq.queue.Name(), accessor.Name(), "Accessor Name() must proxy the underlying queue's name")
		assert.Equal(t, harness.mq.queue.Capabilities(), accessor.Capabilities(),
			"Accessor Capabilities() must proxy the underlying queue's capabilities")
		assert.Equal(t, harness.mq.Len(), accessor.Len(), "Accessor Len() must reflect the managed queue's current length")
		assert.Equal(t, harness.mq.ByteSize(), accessor.ByteSize(),
			"Accessor ByteSize() must reflect the managed queue's current byte size")
		assert.Equal(t, flowKey, accessor.FlowKey(), "Accessor FlowKey() must return the correct identifier for the flow")
		assert.Equal(t, harness.mockPolicy.Comparator(), accessor.Comparator(),
			"Accessor Comparator() must return the comparator provided by the configured intra-flow policy")

		peekedHead := accessor.PeekHead()
		assert.Same(t, item, peekedHead, "Accessor PeekHead() must return the exact item instance at the head")

		peekedTail := accessor.PeekTail()
		assert.Same(t, item, peekedTail, "Accessor PeekTail() must return the exact item instance at the tail")
	})

	t.Run("EmptyQueue", func(t *testing.T) {
		t.Parallel()
		flowKey := types.FlowKey{ID: "flow", Priority: 1}
		q := &frameworkmocks.MockSafeQueue{}
		q.PeekHeadV = nil
		harness := newMockedMqHarness(t, q, flowKey)
		accessor := harness.mq.FlowQueueAccessor()
		assert.Nil(t, accessor.PeekHead(), "Accessor PeekHead() should return an nil on an empty queue")
	})
}

// --- Concurrency Test ---

// TestManagedQueue_Concurrency_StatsIntegrity validates that under high contention, the final propagated statistics
// are consistent. It spins up multiple goroutines that concurrently and rapidly add and remove items to stress-test
// the mutex protecting write operations and the atomic propagation of statistics.
func TestManagedQueue_Concurrency_StatsIntegrity(t *testing.T) {
	t.Parallel()
	const (
		numWorkers   = 10
		opsPerWorker = 500
		itemByteSize = 10
	)

	flowKey := types.FlowKey{ID: "flow", Priority: 1}
	h := newRealMqHarness(t, flowKey)
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for range numWorkers {
		go func() {
			defer wg.Done()
			for range opsPerWorker {
				item := typesmocks.NewMockQueueItemAccessor(uint64(itemByteSize), "req", flowKey)
				require.NoError(t, h.mq.Add(item), "Concurrent Add operation must succeed without errors or races")
				// In this chaos test, `Remove` may fail if another goroutine removes the item first. This is expected.
				_, _ = h.mq.Remove(item.Handle())
			}
		}()
	}
	wg.Wait()

	// After all operations, the queue should ideally be empty, but we drain any remaining items to get a definitive final
	// state.
	h.mq.Drain()
	assert.Zero(t, h.mq.Len(), "Final queue length must be zero after draining all remaining items")
	assert.Zero(t, h.mq.ByteSize(), "Final queue byte size must be zero after draining all remaining items")
	assert.Equal(t, int64(0), h.propagator.lenDelta.Load(),
		"The net length delta propagated across all operations must be exactly zero")
	assert.Equal(t, int64(0), h.propagator.byteSizeDelta.Load(),
		"The net byte size delta propagated across all operations must be exactly zero")
}

// --- Invariant Test ---

func TestManagedQueue_InvariantPanics_OnUnderflow(t *testing.T) {
	t.Parallel()
	flowKey := types.FlowKey{ID: "flow", Priority: 1}
	item := typesmocks.NewMockQueueItemAccessor(100, "req", flowKey)
	q := &frameworkmocks.MockSafeQueue{}
	q.AddFunc = func(types.QueueItemAccessor) {}
	q.RemoveFunc = func(types.QueueItemHandle) (types.QueueItemAccessor, error) {
		return item, nil
	}
	h := newMockedMqHarness(t, q, flowKey)

	require.NoError(t, h.mq.Add(item), "Test setup: Initial Add must succeed")
	_, err := h.mq.Remove(item.Handle())
	require.NoError(t, err, "Test setup: First Remove must succeed")

	// This remove call should cause the stats to go negative.
	assert.Panics(t,
		func() {
			// Mock the underlying queue to succeed on the second remove, even though it's logically inconsistent.
			// This isolates the panic to the `managedQueue`'s decorator logic.
			_, _ = h.mq.Remove(item.Handle())
		},
		"Attempting to remove an item that results in negative statistics must trigger an invariant violation panic",
	)
}
