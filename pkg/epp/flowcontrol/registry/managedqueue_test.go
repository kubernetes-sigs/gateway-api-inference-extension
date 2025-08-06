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
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue/listqueue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	typesmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

// --- Test Harness and Mocks ---

// mqTestHarness holds all components for testing a `managedQueue`.
type mqTestHarness struct {
	t              *testing.T
	mq             *managedQueue
	propagator     *mockStatsPropagator
	signalRecorder *mockQueueStateSignalRecorder
	mockPolicy     *frameworkmocks.MockIntraFlowDispatchPolicy
}

// newMockedMqHarness creates a harness that uses a mocked underlying queue.
// This is ideal for isolating and unit testing the decorator logic of `managedQueue`.
func newMockedMqHarness(t *testing.T, queue *frameworkmocks.MockSafeQueue, key types.FlowKey) *mqTestHarness {
	t.Helper()
	return newMqHarness(t, queue, key)
}

// newRealMqHarness creates a harness that uses a real "ListQueue" implementation.
// This is essential for integration and concurrency tests.
func newRealMqHarness(t *testing.T, key types.FlowKey) *mqTestHarness {
	t.Helper()
	q, err := queue.NewQueueFromName(listqueue.ListQueueName, nil)
	require.NoError(t, err, "Test setup: creating a real ListQueue should not fail")
	return newMqHarness(t, q, key)
}

func newMqHarness(t *testing.T, queue framework.SafeQueue, key types.FlowKey) *mqTestHarness {
	t.Helper()

	propagator := &mockStatsPropagator{}
	signalRec := newMockQueueStateSignalRecorder()
	mockPolicy := &frameworkmocks.MockIntraFlowDispatchPolicy{
		ComparatorV: &frameworkmocks.MockItemComparator{},
	}

	callbacks := managedQueueCallbacks{
		propagateStatsDelta: propagator.propagate,
		signalQueueState:    signalRec.signal,
	}
	mq := newManagedQueue(queue, mockPolicy, key, logr.Discard(), callbacks)
	require.NotNil(t, mq, "Test setup: newManagedQueue should not return nil")

	return &mqTestHarness{
		t:              t,
		mq:             mq,
		propagator:     propagator,
		signalRecorder: signalRec,
		mockPolicy:     mockPolicy,
	}
}

func (h *mqTestHarness) setupWithItems(items ...types.QueueItemAccessor) {
	h.t.Helper()
	for _, item := range items {
		err := h.mq.Add(item)
		require.NoError(h.t, err, "Harness setup: failed to add item")
	}
	// Reset counters after the setup phase is complete.
	h.propagator.reset()
}

// mockStatsPropagator captures stat changes from the system under test.
type mockStatsPropagator struct {
	lenDelta      atomic.Int64
	byteSizeDelta atomic.Int64
}

func (p *mockStatsPropagator) propagate(_ uint, lenDelta, byteSizeDelta int64) {
	p.lenDelta.Add(lenDelta)
	p.byteSizeDelta.Add(byteSizeDelta)
}

func (p *mockStatsPropagator) reset() {
	p.lenDelta.Store(0)
	p.byteSizeDelta.Store(0)
}

// mockQueueStateSignalRecorder captures queue state signals from the system under test.
type mockQueueStateSignalRecorder struct {
	mu      sync.Mutex
	signals []queueStateSignal
}

func newMockQueueStateSignalRecorder() *mockQueueStateSignalRecorder {
	return &mockQueueStateSignalRecorder{signals: make([]queueStateSignal, 0)}
}

func (r *mockQueueStateSignalRecorder) signal(_ types.FlowKey, signal queueStateSignal) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.signals = append(r.signals, signal)
}

func (r *mockQueueStateSignalRecorder) getSignals() []queueStateSignal {
	r.mu.Lock()
	defer r.mu.Unlock()
	signalsCopy := make([]queueStateSignal, len(r.signals))
	copy(signalsCopy, r.signals)
	return signalsCopy
}

// --- Unit Tests ---

func TestManagedQueue_InitialState(t *testing.T) {
	t.Parallel()
	h := newMockedMqHarness(t, &frameworkmocks.MockSafeQueue{}, types.FlowKey{ID: "flow", Priority: 1})
	assert.Zero(t, h.mq.Len(), "A new queue should have a length of 0")
	assert.Zero(t, h.mq.ByteSize(), "A new queue should have a byte size of 0")
}

func TestManagedQueue_Add(t *testing.T) {
	t.Parallel()
	flowKey := types.FlowKey{ID: "flow", Priority: 1}

	testCases := []struct {
		name                  string
		setupMock             func(q *frameworkmocks.MockSafeQueue)
		expectErr             bool
		expectedLenDelta      int64
		expectedByteSizeDelta int64
	}{
		{
			name: "ShouldSucceed_AndIncrementStats",
			setupMock: func(q *frameworkmocks.MockSafeQueue) {
				q.AddFunc = func(types.QueueItemAccessor) error { return nil }
			},
			expectErr:             false,
			expectedLenDelta:      1,
			expectedByteSizeDelta: 100,
		},
		{
			name: "ShouldFail_AndNotChangeStats_WhenUnderlyingQueueFails",
			setupMock: func(q *frameworkmocks.MockSafeQueue) {
				q.AddFunc = func(types.QueueItemAccessor) error { return errors.New("add failed") }
			},
			expectErr:             true,
			expectedLenDelta:      0,
			expectedByteSizeDelta: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			item := typesmocks.NewMockQueueItemAccessor(100, "req", flowKey)
			q := &frameworkmocks.MockSafeQueue{}
			tc.setupMock(q)
			h := newMockedMqHarness(t, q, flowKey)

			err := h.mq.Add(item)

			if tc.expectErr {
				require.Error(t, err, "Add should have returned an error")
			} else {
				require.NoError(t, err, "Add should not have returned an error")
			}
			assert.Equal(t, tc.expectedLenDelta, h.propagator.lenDelta.Load(),
				"Propagated length delta should be correct")
			assert.Equal(t, tc.expectedByteSizeDelta, h.propagator.byteSizeDelta.Load(),
				"Propagated byte size delta should be correct")
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
				require.Error(t, err, "Remove should have returned an error")
			} else {
				require.NoError(t, err, "Remove should not have returned an error")
			}
			assert.Equal(t, tc.expectedLenDelta, h.propagator.lenDelta.Load(), "Propagated length delta should be correct")
			assert.Equal(t, tc.expectedByteSizeDelta, h.propagator.byteSizeDelta.Load(),
				"Propagated byte size delta should be correct")
		})
	}
}

func TestManagedQueue_Cleanup(t *testing.T) {
	t.Parallel()
	flowKey := types.FlowKey{ID: "flow", Priority: 1}

	testCases := []struct {
		name                  string
		setupMock             func(q *frameworkmocks.MockSafeQueue, items []types.QueueItemAccessor)
		expectErr             bool
		expectedLenDelta      int64
		expectedByteSizeDelta int64
	}{
		{
			name: "ShouldSucceed_AndDecrementStats",
			setupMock: func(q *frameworkmocks.MockSafeQueue, items []types.QueueItemAccessor) {
				q.CleanupFunc = func(_ framework.PredicateFunc) ([]types.QueueItemAccessor, error) {
					return items, nil
				}
			},
			expectErr:             false,
			expectedLenDelta:      -2,
			expectedByteSizeDelta: -125,
		},
		{
			name: "ShouldSucceed_AndNotChangeStats_WhenNoItemsRemoved",
			setupMock: func(q *frameworkmocks.MockSafeQueue, items []types.QueueItemAccessor) {
				q.CleanupFunc = func(_ framework.PredicateFunc) ([]types.QueueItemAccessor, error) {
					return nil, nil // Simulate no items matching predicate.
				}
			},
			expectErr:             false,
			expectedLenDelta:      0,
			expectedByteSizeDelta: 0,
		},
		{
			name: "ShouldFail_AndNotChangeStats_WhenUnderlyingQueueFails",
			setupMock: func(q *frameworkmocks.MockSafeQueue, items []types.QueueItemAccessor) {
				q.CleanupFunc = func(_ framework.PredicateFunc) ([]types.QueueItemAccessor, error) {
					return nil, errors.New("cleanup failed")
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
			items := []types.QueueItemAccessor{
				typesmocks.NewMockQueueItemAccessor(50, "req", flowKey),
				typesmocks.NewMockQueueItemAccessor(75, "req", flowKey),
			}
			h.setupWithItems(items...)
			tc.setupMock(q, items)

			_, err := h.mq.Cleanup(func(_ types.QueueItemAccessor) bool { return true })

			if tc.expectErr {
				require.Error(t, err, "Cleanup should have returned an error")
			} else {
				require.NoError(t, err, "Cleanup should not have returned an error")
			}
			assert.Equal(t, tc.expectedLenDelta, h.propagator.lenDelta.Load(), "Propagated length delta should be correct")
			assert.Equal(t, tc.expectedByteSizeDelta, h.propagator.byteSizeDelta.Load(),
				"Propagated byte size delta should be correct")
		})
	}
}

func TestManagedQueue_Drain(t *testing.T) {
	t.Parallel()
	flowKey := types.FlowKey{ID: "flow", Priority: 1}

	testCases := []struct {
		name                  string
		setupMock             func(q *frameworkmocks.MockSafeQueue, items []types.QueueItemAccessor)
		expectErr             bool
		expectedLenDelta      int64
		expectedByteSizeDelta int64
	}{
		{
			name: "ShouldSucceed_AndDecrementStats",
			setupMock: func(q *frameworkmocks.MockSafeQueue, items []types.QueueItemAccessor) {
				q.DrainFunc = func() ([]types.QueueItemAccessor, error) {
					return items, nil
				}
			},
			expectErr:             false,
			expectedLenDelta:      -2,
			expectedByteSizeDelta: -125,
		},
		{
			name: "ShouldFail_AndNotChangeStats_WhenUnderlyingQueueFails",
			setupMock: func(q *frameworkmocks.MockSafeQueue, items []types.QueueItemAccessor) {
				q.DrainFunc = func() ([]types.QueueItemAccessor, error) {
					return nil, errors.New("drain failed")
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
			items := []types.QueueItemAccessor{
				typesmocks.NewMockQueueItemAccessor(50, "req", flowKey),
				typesmocks.NewMockQueueItemAccessor(75, "req", flowKey),
			}
			h.setupWithItems(items...)
			tc.setupMock(q, items)

			_, err := h.mq.Drain()

			if tc.expectErr {
				require.Error(t, err, "Drain should have returned an error")
			} else {
				require.NoError(t, err, "Drain should not have returned an error")
			}
			assert.Equal(t, tc.expectedLenDelta, h.propagator.lenDelta.Load(), "Propagated length delta should be correct")
			assert.Equal(t, tc.expectedByteSizeDelta, h.propagator.byteSizeDelta.Load(),
				"Propagated byte size delta should be correct")
		})
	}
}

func TestManagedQueue_PanicOnUnderflow(t *testing.T) {
	t.Parallel()
	flowKey := types.FlowKey{ID: "flow", Priority: 1}
	item := typesmocks.NewMockQueueItemAccessor(100, "req", flowKey)
	q := &frameworkmocks.MockSafeQueue{}
	q.AddFunc = func(types.QueueItemAccessor) error { return nil }
	q.RemoveFunc = func(types.QueueItemHandle) (types.QueueItemAccessor, error) {
		return item, nil
	}
	h := newMockedMqHarness(t, q, flowKey)

	// Add and then successfully remove the item.
	require.NoError(t, h.mq.Add(item), "Test setup: Add should succeed")
	_, err := h.mq.Remove(item.Handle())
	require.NoError(t, err, "Test setup: First Remove should succeed")
	require.Zero(t, h.mq.Len(), "Test setup: Queue should be empty")

	// Attempting to remove the same item again should cause a panic.
	assert.PanicsWithValue(t,
		fmt.Sprintf("invariant violation: managedQueue length for flow %s became negative (-1)", flowKey),
		func() { _, _ = h.mq.Remove(item.Handle()) },
		"A second removal of the same item should trigger a panic on length underflow",
	)
}

func TestManagedQueue_Signaling(t *testing.T) {
	t.Parallel()
	flowKey := types.FlowKey{ID: "flow", Priority: 1}
	h := newRealMqHarness(t, flowKey)
	item1 := typesmocks.NewMockQueueItemAccessor(100, "r1", flowKey)
	item2 := typesmocks.NewMockQueueItemAccessor(50, "r2", flowKey)

	// 1. Initial state: Empty
	assert.Empty(t, h.signalRecorder.getSignals(), "No signals should be present on a new queue")

	// 2. Transition: Empty -> NonEmpty
	require.NoError(t, h.mq.Add(item1), "Adding an item should not fail")
	assert.Equal(t, []queueStateSignal{queueStateSignalBecameNonEmpty}, h.signalRecorder.getSignals(),
		"Should signal BecameNonEmpty on first add")

	// 3. Steady state: NonEmpty -> NonEmpty
	require.NoError(t, h.mq.Add(item2), "Adding a second item should not fail")
	assert.Equal(t, []queueStateSignal{queueStateSignalBecameNonEmpty}, h.signalRecorder.getSignals(),
		"No new signal should be sent when adding to a non-empty queue")

	// 4. Steady state: NonEmpty -> NonEmpty
	_, err := h.mq.Remove(item1.Handle())
	require.NoError(t, err, "Removing an item should not fail")
	assert.Equal(t, []queueStateSignal{queueStateSignalBecameNonEmpty}, h.signalRecorder.getSignals(),
		"No new signal should be sent when removing from a multi-item queue")

	// 5. Transition: NonEmpty -> Empty
	_, err = h.mq.Remove(item2.Handle())
	require.NoError(t, err, "Removing the last item should not fail")
	expectedSignalSequence := []queueStateSignal{queueStateSignalBecameNonEmpty, queueStateSignalBecameEmpty}
	assert.Equal(t, expectedSignalSequence, h.signalRecorder.getSignals(),
		"Should signal BecameEmpty on removal of the last item")
}

func TestManagedQueue_FlowQueueAccessor_ProxiesCalls(t *testing.T) {
	t.Parallel()
	flowKey := types.FlowKey{ID: "flow", Priority: 1}
	q := &frameworkmocks.MockSafeQueue{}
	harness := newMockedMqHarness(t, q, flowKey)
	item := typesmocks.NewMockQueueItemAccessor(100, "req-1", flowKey)
	q.PeekHeadV = item
	q.PeekTailV = item
	q.NameV = "MockQueue"
	q.CapabilitiesV = []framework.QueueCapability{framework.CapabilityFIFO}
	require.NoError(t, harness.mq.Add(item), "Test setup: Adding an item should not fail")

	accessor := harness.mq.FlowQueueAccessor()
	require.NotNil(t, accessor, "FlowQueueAccessor should not be nil")

	assert.Equal(t, harness.mq.Name(), accessor.Name(), "Accessor Name() should match managed queue")
	assert.Equal(t, harness.mq.Capabilities(), accessor.Capabilities(),
		"Accessor Capabilities() should match managed queue")
	assert.Equal(t, harness.mq.Len(), accessor.Len(), "Accessor Len() should match managed queue")
	assert.Equal(t, harness.mq.ByteSize(), accessor.ByteSize(), "Accessor ByteSize() should match managed queue")
	assert.Equal(t, flowKey, accessor.FlowKey(), "Accessor FlowKey() should match managed queue")
	assert.Equal(t, harness.mockPolicy.Comparator(), accessor.Comparator(),
		"Accessor Comparator() should match the one from the policy")
	assert.Equal(t, harness.mockPolicy.Comparator(), harness.mq.Comparator(),
		"ManagedQueue Comparator() should also match the one from the policy")

	peekedHead, err := accessor.PeekHead()
	require.NoError(t, err, "Accessor PeekHead() should not return an error")
	assert.Same(t, item, peekedHead, "Accessor PeekHead() should return the correct item instance")

	peekedTail, err := accessor.PeekTail()
	require.NoError(t, err, "Accessor PeekTail() should not return an error")
	assert.Same(t, item, peekedTail, "Accessor PeekTail() should return the correct item instance")
}

// --- Concurrency Tests ---

// TestManagedQueue_Concurrency_SignalingRace targets the race condition of a queue flapping between empty and non-empty
// states.
// It ensures the `BecameEmpty` and `BecameNonEmpty` signals are sent correctly in strict alternation, without
// duplicates or missed signals.
func TestManagedQueue_Concurrency_SignalingRace(t *testing.T) {
	t.Parallel()

	const ops = 1000
	flowKey := types.FlowKey{ID: "flow", Priority: 1}
	h := newRealMqHarness(t, flowKey)

	var wg sync.WaitGroup
	wg.Add(2)

	// Goroutine 1: Adder - continuously adds items.
	go func() {
		defer wg.Done()
		for range ops {
			_ = h.mq.Add(typesmocks.NewMockQueueItemAccessor(10, "req", flowKey))
		}
	}()

	// Goroutine 2: Remover - continuously drains the queue.
	go func() {
		defer wg.Done()
		for range ops {
			// Drain is used to remove all items present in a single atomic operation.
			_, _ = h.mq.Drain()
		}
	}()

	wg.Wait()

	// Verification: The critical part of this test is to analyze the sequence of signals.
	signals := h.signalRecorder.getSignals()
	require.NotEmpty(t, signals, "At least some signals should have been generated")

	// The sequence must be a strict alternation of `BecameNonEmpty` and `BecameEmpty` signals.
	// There should never be two of the same signal in a row.
	for i := 0; i < len(signals)-1; i++ {
		assert.NotEqual(t, signals[i], signals[i+1], "Signals at index %d and %d must not be duplicates", i, i+1)
	}

	// The first signal must be `BecameNonEmpty`.
	assert.Equal(t, queueStateSignalBecameNonEmpty, signals[0], "The first signal must be BecameNonEmpty")
}

// TestManagedQueue_Concurrency_ItemIntegrity validates that under high concurrency, the queue does not lose or
// duplicate items and that the final propagated statistics are consistent with the operations performed.
func TestManagedQueue_Concurrency_ItemIntegrity(t *testing.T) {
	t.Parallel()

	const numGoRoutines = 10
	const opsPerGoRoutine = 500
	const itemByteSize = 10

	flowKey := types.FlowKey{ID: "flow", Priority: 1}
	h := newRealMqHarness(t, flowKey)
	var wg sync.WaitGroup
	wg.Add(numGoRoutines)

	// Each goroutine will attempt to perform a mix of `Add` and `Remove` operations.
	for range numGoRoutines {
		go func() {
			defer wg.Done()
			for range opsPerGoRoutine {
				// Add an item.
				item := typesmocks.NewMockQueueItemAccessor(uint64(itemByteSize), "req", flowKey)
				require.NoError(t, h.mq.Add(item), "Concurrent Add should not fail")

				// Immediately try to remove it. This creates high contention on the queue's internal state.
				_, _ = h.mq.Remove(item.Handle())
			}
		}()
	}

	wg.Wait()

	// After all operations, the queue should ideally be empty, but some removals might have failed if another goroutine
	// got to it first. We drain any remaining items to get a final count.
	_, err := h.mq.Drain()
	require.NoError(t, err, "Final drain should not fail")

	// Final State Verification
	assert.Zero(t, h.mq.Len(), "Queue length must be zero after final drain")
	assert.Zero(t, h.mq.ByteSize(), "Queue byte size must be zero after final drain")

	// Statistical Integrity Verification
	// The total number of propagated additions must equal the number of propagated removals.
	lenDelta := h.propagator.lenDelta.Load()
	byteSizeDelta := h.propagator.byteSizeDelta.Load()

	assert.Equal(t, int64(0), lenDelta,
		"Net length delta propagated must be zero, proving every add was matched by a remove delta")
	assert.Equal(t, int64(0), byteSizeDelta, "Net byte size delta propagated must be zero")
}
