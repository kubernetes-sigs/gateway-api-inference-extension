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
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue/listqueue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	typesmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

// mockStatsPropagator is a mock implementation of `propagateStatsDeltaFunc` for capturing stat changes.
// It uses atomics to be safe for concurrent use in stress tests.
type mockStatsPropagator struct {
	lenDelta      atomic.Int64
	byteSizeDelta atomic.Int64
	invocations   atomic.Int64
}

func (p *mockStatsPropagator) propagate(_ uint, lenDelta, byteSizeDelta int64) {
	p.lenDelta.Add(lenDelta)
	p.byteSizeDelta.Add(byteSizeDelta)
	p.invocations.Add(1)
}

func (p *mockStatsPropagator) getStats() (lenDelta, byteSizeDelta int64, count int) {
	return p.lenDelta.Load(), p.byteSizeDelta.Load(), int(p.invocations.Load())
}

// mockManagedQueueSignalRecorder is a thread-safe helper for recording queue state signals.
type mockManagedQueueSignalRecorder struct {
	mu      sync.Mutex
	signals []queueStateSignal
}

// newMockManagedQueueSignalRecorder initializes the signals slice to be non-nil, preventing assertion failures when
// comparing a nil slice with an empty one.
func newMockManagedQueueSignalRecorder() *mockManagedQueueSignalRecorder {
	return &mockManagedQueueSignalRecorder{
		signals: make([]queueStateSignal, 0),
	}
}

func (r *mockManagedQueueSignalRecorder) signal(_ types.FlowSpecification, signal queueStateSignal) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.signals = append(r.signals, signal)
}

func (r *mockManagedQueueSignalRecorder) getSignals() []queueStateSignal {
	r.mu.Lock()
	defer r.mu.Unlock()
	// Return a copy to prevent data races if the caller iterates over the slice while new signals are concurrently added
	// by the system under test.
	signalsCopy := make([]queueStateSignal, len(r.signals))
	copy(signalsCopy, r.signals)
	return signalsCopy
}

func (r *mockManagedQueueSignalRecorder) clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.signals = make([]queueStateSignal, 0)
}

// mqTestHarness holds all components for testing a `managedQueue`.
type mqTestHarness struct {
	t              *testing.T
	mq             *managedQueue
	mockQueue      *frameworkmocks.MockSafeQueue
	mockPolicy     *frameworkmocks.MockIntraFlowDispatchPolicy
	propagator     *mockStatsPropagator
	signalRecorder *mockManagedQueueSignalRecorder
	flowSpec       types.FlowSpecification
}

// newMqTestHarness creates a new test harness.
// The `useRealQueue` flag allows swapping between a mocked `framework.SafeQueue` and a real one for different test
// scenarios.
func newMqTestHarness(t *testing.T, useRealQueue bool) *mqTestHarness {
	t.Helper()

	propagator := &mockStatsPropagator{}
	signalRec := newMockManagedQueueSignalRecorder()
	flowSpec := types.FlowSpecification{ID: "test-flow", Priority: 1}
	mockPolicy := &frameworkmocks.MockIntraFlowDispatchPolicy{
		ComparatorV: &frameworkmocks.MockItemComparator{},
	}

	var q framework.SafeQueue
	var mockQueue *frameworkmocks.MockSafeQueue

	if useRealQueue {
		// Use a real queue implementation for concurrency tests or when behavior is complex.
		realQueue, err := queue.NewQueueFromName(listqueue.ListQueueName, nil)
		require.NoError(t, err, "Test setup: creating a real listqueue should not fail")
		q = realQueue
	} else {
		// Use a mock queue for unit tests to isolate the `managedQueue`'s logic.
		mockQueue = &frameworkmocks.MockSafeQueue{}
		q = mockQueue
	}

	callbacks := managedQueueCallbacks{
		propagateStatsDelta: propagator.propagate,
		signalQueueState:    signalRec.signal,
	}
	mq := newManagedQueue(q, mockPolicy, flowSpec, logr.Discard(), callbacks)
	require.NotNil(t, mq, "Test setup: newManagedQueue should not return nil")

	return &mqTestHarness{
		t:              t,
		mq:             mq,
		mockQueue:      mockQueue,
		mockPolicy:     mockPolicy,
		propagator:     propagator,
		signalRecorder: signalRec,
		flowSpec:       flowSpec,
	}
}

// addItem is a test helper to add an item to the managed queue.
func (h *mqTestHarness) addItem(size uint64) types.QueueItemAccessor {
	h.t.Helper()
	item := typesmocks.NewMockQueueItemAccessor(size, "req", h.flowSpec.ID)
	require.NoError(h.t, h.mq.Add(item), "addItem helper should successfully add item to queue")
	return item
}

// removeItem is a test helper to remove an item from the managed queue.
func (h *mqTestHarness) removeItem(item types.QueueItemAccessor) {
	h.t.Helper()
	_, err := h.mq.Remove(item.Handle())
	require.NoError(h.t, err, "removeItem helper should successfully remove item from queue")
}

// assertSignals checks that the recorded signals match the expected sequence.
func (h *mqTestHarness) assertSignals(expected ...queueStateSignal) {
	h.t.Helper()
	// Ensure nil expected slice is treated as empty for consistent assertions.
	if expected == nil {
		expected = make([]queueStateSignal, 0)
	}
	assert.Equal(h.t, expected, h.signalRecorder.getSignals(), "The sequence of emitted GC signals should be correct")
}

// assertStatus verifies the queue's lifecycle status.
func (h *mqTestHarness) assertStatus(expected componentStatus, msgAndArgs ...interface{}) {
	h.t.Helper()
	assert.Equal(h.t, expected, componentStatus(h.mq.status.Load()), msgAndArgs...)
}

func TestManagedQueue(t *testing.T) {
	t.Parallel()

	t.Run("New_InitialState", func(t *testing.T) {
		t.Parallel()
		harness := newMqTestHarness(t, false)

		assert.Zero(t, harness.mq.Len(), "A new managedQueue should have a length of 0")
		assert.Zero(t, harness.mq.ByteSize(), "A new managedQueue should have a byte size of 0")
		harness.assertStatus(componentStatusActive, "A new queue should be in the Active state")
	})

	t.Run("Add_Scenarios", func(t *testing.T) {
		t.Parallel()

		testCases := []struct {
			name                  string
			setup                 func(h *mqTestHarness)
			itemByteSize          uint64
			expectErr             bool
			errIs                 error
			expectedLen           int
			expectedByteSize      uint64
			expectedLenDelta      int64
			expectedByteSizeDelta int64
			expectPropagateCall   bool
		}{
			{
				name: "WhenQueueIsActive_ShouldSucceed",
				setup: func(h *mqTestHarness) {
					h.mockQueue.AddFunc = func(item types.QueueItemAccessor) error { return nil }
				},
				itemByteSize:          100,
				expectErr:             false,
				expectedLen:           1,
				expectedByteSize:      100,
				expectedLenDelta:      1,
				expectedByteSizeDelta: 100,
				expectPropagateCall:   true,
			},
			{
				name: "WhenUnderlyingQueueFails_ShouldFail",
				setup: func(h *mqTestHarness) {
					h.mockQueue.AddFunc = func(item types.QueueItemAccessor) error { return errors.New("queue full") }
				},
				itemByteSize:          100,
				expectErr:             true,
				expectedLen:           0,
				expectedByteSize:      0,
				expectedLenDelta:      0,
				expectedByteSizeDelta: 0,
				expectPropagateCall:   false,
			},
			{
				name: "WhenQueueIsDraining_ShouldFail",
				setup: func(h *mqTestHarness) {
					h.mq.status.Store(int32(componentStatusDraining))
				},
				itemByteSize:          100,
				expectErr:             true,
				errIs:                 contracts.ErrFlowInstanceNotFound,
				expectedLen:           0,
				expectedByteSize:      0,
				expectedLenDelta:      0,
				expectedByteSizeDelta: 0,
				expectPropagateCall:   false,
			},
			{
				name: "WhenQueueIsDrained_ShouldFail",
				setup: func(h *mqTestHarness) {
					h.mq.status.Store(int32(componentStatusDrained))
				},
				itemByteSize:          100,
				expectErr:             true,
				errIs:                 contracts.ErrFlowInstanceNotFound,
				expectedLen:           0,
				expectedByteSize:      0,
				expectedLenDelta:      0,
				expectedByteSizeDelta: 0,
				expectPropagateCall:   false,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				harness := newMqTestHarness(t, false)

				if tc.setup != nil {
					tc.setup(harness)
				}

				item := typesmocks.NewMockQueueItemAccessor(tc.itemByteSize, "req-1", "test-flow")
				err := harness.mq.Add(item)

				if tc.expectErr {
					require.Error(t, err, "Add should have returned an error")
					if tc.errIs != nil {
						assert.ErrorIs(t, err, tc.errIs, "Error should wrap the expected sentinel error")
					}
				} else {
					require.NoError(t, err, "Add should not have returned an error")
				}

				assert.Equal(t, tc.expectedLen, harness.mq.Len(), "Final length should be as expected")
				assert.Equal(t, tc.expectedByteSize, harness.mq.ByteSize(), "Final byte size should be as expected")

				lenDelta, byteSizeDelta, count := harness.propagator.getStats()
				assert.Equal(t, tc.expectedLenDelta, lenDelta, "Propagator length delta should be as expected")
				assert.Equal(t, tc.expectedByteSizeDelta, byteSizeDelta, "Propagator byte size delta should be as expected")
				if tc.expectPropagateCall {
					assert.Equal(t, 1, count, "Propagator should have been called exactly once")
				} else {
					assert.Zero(t, count, "Propagator should not have been called")
				}
			})
		}
	})

	t.Run("Remove_Scenarios", func(t *testing.T) {
		t.Parallel()

		item := typesmocks.NewMockQueueItemAccessor(100, "req-1", "test-flow")

		testCases := []struct {
			name                  string
			setupMock             func(q *frameworkmocks.MockSafeQueue)
			expectErr             bool
			expectedLen           int
			expectedByteSize      uint64
			expectedPropagatorOps int
		}{
			{
				name: "WhenRemoveSucceeds_ShouldDecrementStats",
				setupMock: func(q *frameworkmocks.MockSafeQueue) {
					q.RemoveFunc = func(handle types.QueueItemHandle) (types.QueueItemAccessor, error) {
						return item, nil
					}
				},
				expectErr:             false,
				expectedLen:           0,
				expectedByteSize:      0,
				expectedPropagatorOps: 2, // 1 for add, 1 for remove
			},
			{
				name: "WhenRemoveFails_ShouldNotChangeStats",
				setupMock: func(q *frameworkmocks.MockSafeQueue) {
					q.RemoveFunc = func(handle types.QueueItemHandle) (types.QueueItemAccessor, error) {
						return nil, errors.New("item not found")
					}
				},
				expectErr:             true,
				expectedLen:           1,
				expectedByteSize:      100,
				expectedPropagatorOps: 1, // Only the initial add
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				harness := newMqTestHarness(t, false)

				// Common setup: add one item to the queue.
				harness.mockQueue.AddFunc = func(item types.QueueItemAccessor) error { return nil }
				require.NoError(t, harness.mq.Add(item), "Test setup: Adding an item should not fail")

				// Apply test-case specific mock behavior.
				tc.setupMock(harness.mockQueue)

				_, err := harness.mq.Remove(item.Handle())

				if tc.expectErr {
					require.Error(t, err, "Remove should have returned an error")
				} else {
					require.NoError(t, err, "Remove should not have returned an error")
				}

				assert.Equal(t, tc.expectedLen, harness.mq.Len(), "Final length should be as expected")
				assert.Equal(t, tc.expectedByteSize, harness.mq.ByteSize(), "Final byte size should be as expected")

				_, _, count := harness.propagator.getStats()
				assert.Equal(t, tc.expectedPropagatorOps, count,
					"Propagator should have been called the expected number of times")
			})
		}
	})

	t.Run("Cleanup_Scenarios", func(t *testing.T) {
		t.Parallel()
		item1 := typesmocks.NewMockQueueItemAccessor(10, "req-1", "test-flow")
		item2 := typesmocks.NewMockQueueItemAccessor(20, "req-2", "test-flow")
		item3 := typesmocks.NewMockQueueItemAccessor(30, "req-3", "test-flow")

		testCases := []struct {
			name                  string
			itemsToAdd            []types.QueueItemAccessor
			setupMock             func(q *frameworkmocks.MockSafeQueue)
			expectErr             bool
			expectedFinalLen      int
			expectedFinalByteSize uint64
			expectedPropagatorOps int
		}{
			{
				name:       "WhenCleanupSucceeds_ShouldDecrementStats",
				itemsToAdd: []types.QueueItemAccessor{item1, item2, item3},
				setupMock: func(q *frameworkmocks.MockSafeQueue) {
					q.CleanupFunc = func(p framework.PredicateFunc) ([]types.QueueItemAccessor, error) {
						// Simulate removing one item (item2)
						return []types.QueueItemAccessor{item2}, nil
					}
				},
				expectErr:             false,
				expectedFinalLen:      2,
				expectedFinalByteSize: 40, // 10 + 30
				expectedPropagatorOps: 4,  // 3 adds + 1 cleanup
			},
			{
				name:       "WhenCleanupFails_ShouldNotChangeStats",
				itemsToAdd: []types.QueueItemAccessor{item1},
				setupMock: func(q *frameworkmocks.MockSafeQueue) {
					q.CleanupFunc = func(p framework.PredicateFunc) ([]types.QueueItemAccessor, error) {
						return nil, errors.New("internal error")
					}
				},
				expectErr:             true,
				expectedFinalLen:      1,
				expectedFinalByteSize: 10,
				expectedPropagatorOps: 1, // Only the initial add
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				harness := newMqTestHarness(t, false)

				// Common setup: add the specified initial items.
				harness.mockQueue.AddFunc = func(item types.QueueItemAccessor) error { return nil }
				for _, item := range tc.itemsToAdd {
					require.NoError(t, harness.mq.Add(item), "Test setup: adding initial items should not fail")
				}

				// Apply test-case specific mock behavior.
				tc.setupMock(harness.mockQueue)

				_, err := harness.mq.Cleanup(func(i types.QueueItemAccessor) bool { return true })

				if tc.expectErr {
					require.Error(t, err, "Cleanup should have returned an error")
				} else {
					require.NoError(t, err, "Cleanup should not have returned an error")
				}

				assert.Equal(t, tc.expectedFinalLen, harness.mq.Len(), "Final length should be as expected")
				assert.Equal(t, tc.expectedFinalByteSize, harness.mq.ByteSize(), "Final byte size should be as expected")
				_, _, count := harness.propagator.getStats()
				assert.Equal(t, tc.expectedPropagatorOps, count,
					"Propagator should have been called the expected number of times")
			})
		}
	})

	t.Run("Drain_Scenarios", func(t *testing.T) {
		t.Parallel()
		item1 := typesmocks.NewMockQueueItemAccessor(10, "req-1", "test-flow")
		item2 := typesmocks.NewMockQueueItemAccessor(20, "req-2", "test-flow")

		testCases := []struct {
			name                  string
			itemsToAdd            []types.QueueItemAccessor
			setupMock             func(q *frameworkmocks.MockSafeQueue)
			expectErr             bool
			expectedFinalLen      int
			expectedFinalByteSize uint64
			expectedPropagatorOps int
		}{
			{
				name:       "WhenDrainSucceeds_ShouldDecrementStats",
				itemsToAdd: []types.QueueItemAccessor{item1, item2},
				setupMock: func(q *frameworkmocks.MockSafeQueue) {
					q.DrainFunc = func() ([]types.QueueItemAccessor, error) {
						return []types.QueueItemAccessor{item1, item2}, nil
					}
				},
				expectErr:             false,
				expectedFinalLen:      0,
				expectedFinalByteSize: 0,
				expectedPropagatorOps: 3, // 2 adds + 1 drain
			},
			{
				name:       "WhenDrainFails_ShouldNotChangeStats",
				itemsToAdd: []types.QueueItemAccessor{item1},
				setupMock: func(q *frameworkmocks.MockSafeQueue) {
					q.DrainFunc = func() ([]types.QueueItemAccessor, error) {
						return nil, errors.New("internal error")
					}
				},
				expectErr:             true,
				expectedFinalLen:      1,
				expectedFinalByteSize: 10,
				expectedPropagatorOps: 1, // Only the initial add
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				harness := newMqTestHarness(t, false)

				// Common setup: add the specified initial items.
				harness.mockQueue.AddFunc = func(item types.QueueItemAccessor) error { return nil }
				for _, item := range tc.itemsToAdd {
					require.NoError(t, harness.mq.Add(item), "Test setup: adding initial items should not fail")
				}

				// Apply test-case specific mock behavior.
				tc.setupMock(harness.mockQueue)

				_, err := harness.mq.Drain()

				if tc.expectErr {
					require.Error(t, err, "Drain should have returned an error")
				} else {
					require.NoError(t, err, "Drain should not have returned an error")
				}

				assert.Equal(t, tc.expectedFinalLen, harness.mq.Len(), "Final length should be as expected")
				assert.Equal(t, tc.expectedFinalByteSize, harness.mq.ByteSize(), "Final byte size should be as expected")
				_, _, count := harness.propagator.getStats()
				assert.Equal(t, tc.expectedPropagatorOps, count,
					"Propagator should have been called the expected number of times")
			})
		}
	})

	t.Run("FlowQueueAccessor_ProxiesCalls", func(t *testing.T) {
		t.Parallel()
		harness := newMqTestHarness(t, false)
		item := typesmocks.NewMockQueueItemAccessor(100, "req-1", "test-flow")

		harness.mockQueue.PeekHeadV = item
		harness.mockQueue.PeekTailV = item
		harness.mockQueue.NameV = "MockQueue"
		harness.mockQueue.CapabilitiesV = []framework.QueueCapability{framework.CapabilityFIFO}
		harness.mockQueue.AddFunc = func(item types.QueueItemAccessor) error { return nil }

		require.NoError(t, harness.mq.Add(item), "Test setup: Adding an item should not fail")

		accessor := harness.mq.FlowQueueAccessor()
		require.NotNil(t, accessor, "FlowQueueAccessor should not be nil")

		assert.Equal(t, harness.mq.Name(), accessor.Name(), "Accessor Name() should match managed queue")
		assert.Equal(t, harness.mq.Capabilities(), accessor.Capabilities(),
			"Accessor Capabilities() should match managed queue")
		assert.Equal(t, harness.mq.Len(), accessor.Len(), "Accessor Len() should match managed queue")
		assert.Equal(t, harness.mq.ByteSize(), accessor.ByteSize(), "Accessor ByteSize() should match managed queue")
		assert.Equal(t, harness.flowSpec, accessor.FlowSpec(), "Accessor FlowSpec() should match managed queue")
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
	})

	t.Run("StateTransitions_EmitGCSignals", func(t *testing.T) {
		t.Parallel()

		type step struct {
			action          string // "add", "remove", "markDraining"
			itemSize        uint64
			expectedSignals []queueStateSignal
			expectedStatus  componentStatus
		}

		testCases := []struct {
			name           string
			steps          []step
			expectedLen    int
			expectedBytes  uint64
			expectedStatus componentStatus
		}{
			{
				name: "WhenAddingToEmptyActiveQueue_ShouldSignalBecameNonEmpty",
				steps: []step{
					{
						action:          "add",
						itemSize:        100,
						expectedSignals: []queueStateSignal{queueStateSignalBecameNonEmpty},
						expectedStatus:  componentStatusActive,
					},
				},
				expectedLen:    1,
				expectedBytes:  100,
				expectedStatus: componentStatusActive,
			},
			{
				name: "WhenAddingToNonEmptyActiveQueue_ShouldNotSignal",
				steps: []step{
					{
						action:          "add",
						itemSize:        100,
						expectedSignals: []queueStateSignal{queueStateSignalBecameNonEmpty},
						expectedStatus:  componentStatusActive,
					},
					{
						action:          "add",
						itemSize:        50,
						expectedSignals: nil,
						expectedStatus:  componentStatusActive,
					},
				},
				expectedLen:    2,
				expectedBytes:  150,
				expectedStatus: componentStatusActive,
			},
			{
				name: "WhenRemovingLastItemFromActiveQueue_ShouldSignalBecameEmpty",
				steps: []step{
					{
						action:          "add",
						itemSize:        100,
						expectedSignals: []queueStateSignal{queueStateSignalBecameNonEmpty},
						expectedStatus:  componentStatusActive,
					},
					{
						action:          "remove",
						expectedSignals: []queueStateSignal{queueStateSignalBecameEmpty},
						expectedStatus:  componentStatusActive,
					},
				},
				expectedLen:    0,
				expectedBytes:  0,
				expectedStatus: componentStatusActive,
			},
			{
				name: "WhenRemovingFromMultiItemActiveQueue_ShouldNotSignal",
				steps: []step{
					{
						action:          "add",
						itemSize:        100,
						expectedSignals: []queueStateSignal{queueStateSignalBecameNonEmpty},
						expectedStatus:  componentStatusActive,
					},
					{
						action:          "add",
						itemSize:        50,
						expectedSignals: nil,
						expectedStatus:  componentStatusActive,
					},
					{
						action:          "remove",
						expectedSignals: nil,
						expectedStatus:  componentStatusActive,
					},
				},
				expectedLen:    1,
				expectedBytes:  100,
				expectedStatus: componentStatusActive,
			},
			{
				name: "WhenRemovingLastItemFromDrainingQueue_ShouldSignalBecameDrained",
				steps: []step{
					{
						action:          "add",
						itemSize:        100,
						expectedSignals: []queueStateSignal{queueStateSignalBecameNonEmpty},
						expectedStatus:  componentStatusActive,
					},
					{
						action:          "markDraining",
						expectedSignals: nil,
						expectedStatus:  componentStatusDraining,
					},
					{
						action:          "remove",
						expectedSignals: []queueStateSignal{queueStateSignalBecameDrained},
						expectedStatus:  componentStatusDrained,
					},
				},
				expectedLen:    0,
				expectedBytes:  0,
				expectedStatus: componentStatusDrained,
			},
			{
				name: "WhenMarkingEmptyQueueAsDraining_ShouldSignalBecameDrained",
				steps: []step{
					{
						action:          "markDraining",
						expectedSignals: []queueStateSignal{queueStateSignalBecameDrained},
						expectedStatus:  componentStatusDrained,
					},
				},
				expectedLen:    0,
				expectedBytes:  0,
				expectedStatus: componentStatusDrained,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				harness := newMqTestHarness(t, true)
				var lastItem types.QueueItemAccessor

				for i, step := range tc.steps {
					harness.signalRecorder.clear()
					switch step.action {
					case "add":
						lastItem = harness.addItem(step.itemSize)
					case "remove":
						require.NotNil(t, lastItem, "Test setup error: cannot remove without a prior add")
						harness.removeItem(lastItem)
					case "markDraining":
						harness.mq.markAsDraining()
					default:
						t.Fatalf("Unknown test step action: %s", step.action)
					}

					harness.assertSignals(step.expectedSignals...)
					harness.assertStatus(step.expectedStatus, "Step %d (%s): status mismatch", i, step.action)
				}

				assert.Equal(t, tc.expectedLen, harness.mq.Len(), "Final queue length mismatch")
				assert.Equal(t, tc.expectedBytes, harness.mq.ByteSize(), "Final queue byte size mismatch")
				harness.assertStatus(tc.expectedStatus, "Final queue status mismatch")
			})
		}
	})

	t.Run("Lifecycle", func(t *testing.T) {
		t.Parallel()

		t.Run("StateTransitions", func(t *testing.T) {
			t.Parallel()

			t.Run("NewQueue_ShouldBeActive", func(t *testing.T) {
				t.Parallel()
				harness := newMqTestHarness(t, true)
				harness.assertStatus(componentStatusActive)
			})

			t.Run("MarkAsDraining_OnEmptyQueue_ShouldTransitionToDrained", func(t *testing.T) {
				t.Parallel()
				harness := newMqTestHarness(t, true)
				harness.mq.markAsDraining()
				harness.assertStatus(componentStatusDrained,
					"An empty queue should immediately become Drained when marked as such")
			})

			t.Run("Reactivate_OnDrainedQueue_ShouldTransitionToActive", func(t *testing.T) {
				t.Parallel()
				harness := newMqTestHarness(t, true)
				harness.mq.status.Store(int32(componentStatusDrained)) // Setup initial state
				harness.mq.reactivate()
				harness.assertStatus(componentStatusActive, "A Drained queue should become Active after reactivation")
			})

			t.Run("MarkAsDraining_OnNonEmptyQueue_ShouldTransitionToDraining", func(t *testing.T) {
				t.Parallel()
				harness := newMqTestHarness(t, true)
				harness.addItem(10)
				harness.mq.markAsDraining()
				harness.assertStatus(componentStatusDraining,
					"A non-empty queue should stay Draining, not become Drained, until empty")
			})

			t.Run("Reactivate_OnDrainingQueue_ShouldTransitionToActive", func(t *testing.T) {
				t.Parallel()
				harness := newMqTestHarness(t, true)
				harness.addItem(10)
				harness.mq.markAsDraining()
				require.Equal(t, componentStatusDraining, componentStatus(harness.mq.status.Load()),
					"Test setup: queue should be in Draining state before reactivation")

				harness.mq.reactivate()

				harness.assertStatus(componentStatusActive, "A Draining queue should become Active after reactivation")
				assert.Equal(t, 1, harness.mq.Len(), "Queue should still contain its item after reactivation")
			})
		})

		// DrainingRace specifically targets the race condition between marking a queue as draining and the queue
		// concurrently becoming empty. This test ensures that the state machine correctly and atomically transitions to
		// Drained, sending the signal exactly once.
		t.Run("DrainingRace", func(t *testing.T) {
			t.Parallel()
			harness := newMqTestHarness(t, true)
			item := harness.addItem(10)

			var wg sync.WaitGroup
			wg.Add(2)

			// Goroutine 1: Vigorously attempts to mark the queue as draining.
			go func() {
				defer wg.Done()
				harness.mq.markAsDraining()
			}()

			// Goroutine 2: Vigorously attempts to remove the single item.
			go func() {
				defer wg.Done()
				harness.removeItem(item)
			}()

			wg.Wait()

			// Verification:
			// The core assertion is that no matter which operation "won" the race, the final state is deterministically
			// Drained.
			// This proves the atomic CAS operations in the state machine are correct.
			harness.assertStatus(componentStatusDrained, "Final state must be Drained regardless of the race outcome")

			// We also verify that the correct signal was sent exactly once. The exact sequence of signals can vary depending
			// on the race, but the final Drained signal is guaranteed.
			signals := harness.signalRecorder.getSignals()
			assert.Contains(t, signals, queueStateSignalBecameDrained, "The BecameDrained signal must be sent")
			count := 0
			for _, s := range signals {
				if s == queueStateSignalBecameDrained {
					count++
				}
			}
			assert.Equal(t, 1, count, "The BecameDrained signal must be sent exactly once")
		})

		// ActiveFlappingRace targets the race condition of a queue rapidly transitioning between empty and non-empty
		// states. This test ensures the `BecameEmpty` and `BecameNonEmpty` signals are sent correctly in strict
		// alternation, without duplicates or missed signals.
		t.Run("ActiveFlappingRace", func(t *testing.T) {
			t.Parallel()

			numGoroutines := 4
			opsPerGoroutine := 100
			if testing.Short() {
				t.Log("Running in -short mode, reducing workload.")
				numGoroutines = 2
				opsPerGoroutine = 50
			}
			const itemByteSize = 10

			harness := newMqTestHarness(t, true)
			// This channel safely passes items from producer goroutines to consumer goroutines.
			itemsToProcess := make(chan types.QueueItemAccessor, numGoroutines*opsPerGoroutine)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			var wg sync.WaitGroup

			// Start Adder goroutines.
			wg.Add(numGoroutines)
			for i := 0; i < numGoroutines; i++ {
				go func() {
					defer wg.Done()
					for j := 0; j < opsPerGoroutine; j++ {
						select {
						case <-ctx.Done():
							return
						default:
							item := typesmocks.NewMockQueueItemAccessor(itemByteSize, "req", "flow")
							if err := harness.mq.Add(item); err == nil {
								itemsToProcess <- item
							}
						}
					}
				}()
			}

			// Start Remover goroutines.
			wg.Add(numGoroutines)
			for i := 0; i < numGoroutines; i++ {
				go func() {
					defer wg.Done()
					for {
						select {
						case <-ctx.Done():
							return
						case item := <-itemsToProcess:
							_, _ = harness.mq.Remove(item.Handle())
						}
					}
				}()
			}

			wg.Wait()

			// Verification:
			// The critical part of this test is to analyze the sequence of signals.
			signals := harness.signalRecorder.getSignals()
			require.NotEmpty(t, signals, "At least some signals should have been generated")

			// The sequence must be a strict alternation of NonEmpty and Empty signals.
			// There should never be two of the same signal in a row.
			for i := 0; i < len(signals)-1; i++ {
				assert.NotEqual(t, signals[i], signals[i+1], "Signals at index %d and %d must not be duplicates", i, i+1)
			}

			// Depending on the timing, the sequence can start with NonEmpty and end with either.
			if signals[0] != queueStateSignalBecameNonEmpty {
				assert.Fail(t, "The first signal must be BecameNonEmpty")
			}
		})
	})

	t.Run("Concurrency_StressTest", func(t *testing.T) {
		t.Parallel()
		harness := newMqTestHarness(t, true)

		numGoroutines := 20
		opsPerGoroutine := 200
		initialItems := 500
		if testing.Short() {
			t.Log("Running in -short mode, reducing workload.")
			numGoroutines = 4
			opsPerGoroutine = 50
			initialItems = 100
		}

		const itemByteSize = 10

		var wg sync.WaitGroup
		var successfulAdds, successfulRemoves atomic.Int64

		// This channel safely passes item handles from producer goroutines to consumer goroutines.
		handles := make(chan types.QueueItemHandle, initialItems+(numGoroutines*opsPerGoroutine))

		// Pre-fill the queue to ensure removals can happen from the start.
		for range initialItems {
			item := typesmocks.NewMockQueueItemAccessor(uint64(itemByteSize), "initial", "flow")
			require.NoError(t, harness.mq.Add(item), "Test setup: pre-filling queue should not fail")
			handles <- item.Handle()
		}
		// Reset the propagator to only measure the concurrent phase and final drain.
		harness.propagator = &mockStatsPropagator{}
		harness.mq.parentCallbacks.propagateStatsDelta = harness.propagator.propagate

		wg.Add(numGoroutines)
		for i := range numGoroutines {
			go func(routineID int) {
				defer wg.Done()
				for j := range opsPerGoroutine {
					// Alternate between adding and removing items.
					if (routineID+j)%2 == 0 {
						item := typesmocks.NewMockQueueItemAccessor(uint64(itemByteSize), "req", "flow")
						if err := harness.mq.Add(item); err == nil {
							successfulAdds.Add(1)
							handles <- item.Handle()
						}
					} else {
						select {
						case handle := <-handles:
							if _, err := harness.mq.Remove(handle); err == nil {
								successfulRemoves.Add(1)
							}
						default:
							// No handle available, skip this removal attempt.
						}
					}
				}
			}(i)
		}
		wg.Wait()

		drainedItems, err := harness.mq.Drain()
		require.NoError(t, err, "Draining the queue at the end should not fail")

		finalItemCount := len(drainedItems)

		// Core correctness check: The final number of items in the queue must exactly match the initial number, plus all
		// successful concurrent additions, minus all successful concurrent removals.
		// This proves that no items were lost or duplicated during concurrent operations.
		expectedFinalItemCount := initialItems + int(successfulAdds.Load()) - int(successfulRemoves.Load())
		assert.Equal(t, expectedFinalItemCount, finalItemCount,
			"Final item count must match initial + adds - removes, proving no item loss")

		// After a successful drain, the managed queue's own counters must be zero.
		assert.Zero(t, harness.mq.Len(), "Managed queue length must be zero after drain")
		assert.Zero(t, harness.mq.ByteSize(), "Managed queue byte size must be zero after drain")

		// End-to-end statistics check: The net change recorded by the stats propagator must match the net effect of all
		// operations (concurrent phase + final drain).
		// This validates the atomic stats propagation logic across multiple phases.
		netLenChangeDuringConcurrentPhase := successfulAdds.Load() - successfulRemoves.Load()
		netByteSizeChangeDuringConcurrentPhase := netLenChangeDuringConcurrentPhase * itemByteSize

		// The final drain operation also propagates a negative delta.
		lenDeltaFromDrain := -int64(finalItemCount)
		byteSizeDeltaFromDrain := -int64(uint64(finalItemCount) * itemByteSize)

		lenDelta, byteSizeDelta, _ := harness.propagator.getStats()
		expectedLenDelta := netLenChangeDuringConcurrentPhase + lenDeltaFromDrain
		expectedByteSizeDelta := netByteSizeChangeDuringConcurrentPhase + byteSizeDeltaFromDrain

		assert.Equal(t, expectedLenDelta, lenDelta,
			"Net length delta in propagator must match the net change from all operations")
		assert.Equal(t, expectedByteSizeDelta, byteSizeDelta,
			"Net byte size delta in propagator must match the net change from all operations")
	})
}
