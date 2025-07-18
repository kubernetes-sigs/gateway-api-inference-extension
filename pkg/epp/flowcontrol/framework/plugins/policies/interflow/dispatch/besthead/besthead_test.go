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

package besthead

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	typesmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

const (
	flow1 = "flow1"
	flow2 = "flow2"
)

// enqueueTimeComparatorFunc is a test utility. Lower enqueue time is better.
func enqueueTimeComparatorFunc(a, b types.QueueItemAccessor) bool {
	return a.EnqueueTime().Before(b.EnqueueTime())
}

const commonScoreType = "enqueue_time_ns_asc"

func newTestComparator() *frameworkmocks.MockItemComparator {
	return &frameworkmocks.MockItemComparator{
		ScoreTypeV: commonScoreType,
		FuncV:      enqueueTimeComparatorFunc,
	}
}

func TestBestHead_Name(t *testing.T) {
	t.Parallel()
	policy := newBestHead()
	assert.Equal(t, BestHeadPolicyName, policy.Name(), "Name should match the policy's constant")
}

func TestBestHead_SelectQueue(t *testing.T) {
	t.Parallel()
	policy := newBestHead()

	t.Run("SelectQueue_Logic", func(t *testing.T) {
		t.Parallel()

		t.Run("BasicSelection_TwoQueues", func(t *testing.T) {
			t.Parallel()
			now := time.Now()
			itemBetter := typesmocks.NewMockQueueItemAccessor(10, "itemBetter", flow1)
			itemBetter.EnqueueTimeV = now.Add(-10 * time.Second) // Earlier enqueue time = better
			queue1 := &frameworkmocks.MockFlowQueueAccessor{
				LenV:        1,
				PeekHeadV:   itemBetter,
				FlowSpecV:   types.FlowSpecification{ID: flow1},
				ComparatorV: newTestComparator(),
			}

			itemWorse := typesmocks.NewMockQueueItemAccessor(20, "itemWorse", flow2)
			itemWorse.EnqueueTimeV = now.Add(-5 * time.Second) // Later enqueue time = worse
			queue2 := &frameworkmocks.MockFlowQueueAccessor{
				LenV:        1,
				PeekHeadV:   itemWorse,
				FlowSpecV:   types.FlowSpecification{ID: flow2},
				ComparatorV: newTestComparator(),
			}

			mockBand := &frameworkmocks.MockPriorityBandAccessor{
				FlowIDsV: []string{flow1, flow2},
				QueueFuncV: func(id string) framework.FlowQueueAccessor {
					if id == flow1 {
						return queue1
					}
					if id == flow2 {
						return queue2
					}
					return nil
				},
			}

			selected, err := policy.SelectQueue(mockBand)
			require.NoError(t, err, "SelectQueue should not error with valid inputs")
			require.NotNil(t, selected, "SelectQueue should select a queue")
			assert.Equal(t, flow1, selected.FlowSpec().ID, "Should select queue1 with the better item")
		})

		t.Run("IgnoresEmptyQueues", func(t *testing.T) {
			t.Parallel()
			now := time.Now()
			itemBetter := typesmocks.NewMockQueueItemAccessor(10, "itemBetter", flow1)
			itemBetter.EnqueueTimeV = now.Add(-10 * time.Second)
			queue1 := &frameworkmocks.MockFlowQueueAccessor{ // Non-empty
				LenV:        1,
				PeekHeadV:   itemBetter,
				FlowSpecV:   types.FlowSpecification{ID: flow1},
				ComparatorV: newTestComparator(),
			}
			queueEmpty := &frameworkmocks.MockFlowQueueAccessor{ // Empty
				LenV:         0,
				PeekHeadErrV: framework.ErrQueueEmpty,
				FlowSpecV:    types.FlowSpecification{ID: "flowEmpty"},
				ComparatorV:  newTestComparator(),
			}
			itemWorse := typesmocks.NewMockQueueItemAccessor(20, "itemWorse", flow2)
			itemWorse.EnqueueTimeV = now.Add(-5 * time.Second)
			queue2 := &frameworkmocks.MockFlowQueueAccessor{ // Non-empty
				LenV:        1,
				PeekHeadV:   itemWorse,
				FlowSpecV:   types.FlowSpecification{ID: flow2},
				ComparatorV: newTestComparator(),
			}

			mockBand := &frameworkmocks.MockPriorityBandAccessor{
				FlowIDsV: []string{"flowEmpty", flow1, flow2},
				QueueFuncV: func(id string) framework.FlowQueueAccessor {
					switch id {
					case flow1:
						return queue1
					case "flowEmpty":
						return queueEmpty
					case flow2:
						return queue2
					}
					return nil
				},
			}
			selected, err := policy.SelectQueue(mockBand)
			require.NoError(t, err, "SelectQueue should not error when some queues are empty")
			require.NotNil(t, selected, "SelectQueue should select a queue from the non-empty set")
			assert.Equal(t, flow1, selected.FlowSpec().ID, "Should select queue1, ignoring empty queue")
		})

		t.Run("SingleNonEmptyQueue", func(t *testing.T) {
			t.Parallel()
			item := typesmocks.NewMockQueueItemAccessor(10, "item", flow1)
			queue1 := &frameworkmocks.MockFlowQueueAccessor{
				LenV:        1,
				PeekHeadV:   item,
				FlowSpecV:   types.FlowSpecification{ID: flow1},
				ComparatorV: newTestComparator(),
			}
			mockBand := &frameworkmocks.MockPriorityBandAccessor{
				FlowIDsV: []string{flow1},
				QueueFuncV: func(id string) framework.FlowQueueAccessor {
					if id == flow1 {
						return queue1
					}
					return nil
				},
			}
			selected, err := policy.SelectQueue(mockBand)
			require.NoError(t, err, "SelectQueue should not error for a single valid queue")
			require.NotNil(t, selected, "SelectQueue should select the only available queue")
			assert.Equal(t, flow1, selected.FlowSpec().ID, "Should select the only non-empty queue")
		})
	})

	t.Run("SelectQueue_ComparatorCompatibility", func(t *testing.T) {
		t.Parallel()
		queue1 := &frameworkmocks.MockFlowQueueAccessor{
			LenV:      1,
			PeekHeadV: typesmocks.NewMockQueueItemAccessor(10, "item1", flow1),
			FlowSpecV: types.FlowSpecification{ID: flow1},
			ComparatorV: &frameworkmocks.MockItemComparator{
				ScoreTypeV: "typeA",
				FuncV:      enqueueTimeComparatorFunc,
			},
		}
		queue2 := &frameworkmocks.MockFlowQueueAccessor{
			LenV:      1,
			PeekHeadV: typesmocks.NewMockQueueItemAccessor(20, "item2", flow2),
			FlowSpecV: types.FlowSpecification{ID: flow2},
			ComparatorV: &frameworkmocks.MockItemComparator{
				ScoreTypeV: "typeB", // Different ScoreType
				FuncV:      enqueueTimeComparatorFunc,
			},
		}
		mockBand := &frameworkmocks.MockPriorityBandAccessor{
			FlowIDsV: []string{flow1, flow2},
			QueueFuncV: func(id string) framework.FlowQueueAccessor {
				if id == flow1 {
					return queue1
				}
				if id == flow2 {
					return queue2
				}
				return nil
			},
		}
		selected, err := policy.SelectQueue(mockBand)
		assert.Nil(t, selected, "SelectQueue should return a nil queue on incompatible comparator error")
		require.Error(t, err, "SelectQueue should return an error for incompatible comparators")
		assert.ErrorIs(t, err, framework.ErrIncompatiblePriorityType, "Error should be ErrIncompatiblePriorityType")
	})

	t.Run("SelectQueue_EdgeCase", func(t *testing.T) {
		// Scenarios like NilBand, NoQueuesInBand, AllQueuesEmpty are covered by functional tests.
		// These tests focus on edge cases more specific to BestHead's logic.
		t.Parallel()

		t.Run("QueuePeekHeadErrors", func(t *testing.T) {
			t.Parallel()
			queue1 := &frameworkmocks.MockFlowQueueAccessor{
				LenV:         1, // Non-empty
				PeekHeadErrV: errors.New("internal peek error"),
				FlowSpecV:    types.FlowSpecification{ID: flow1},
				ComparatorV:  newTestComparator(),
			}
			queue2 := &frameworkmocks.MockFlowQueueAccessor{
				LenV:        1,
				PeekHeadV:   typesmocks.NewMockQueueItemAccessor(10, "item2", flow2),
				FlowSpecV:   types.FlowSpecification{ID: flow2},
				ComparatorV: newTestComparator(),
			}

			mockBand := &frameworkmocks.MockPriorityBandAccessor{
				FlowIDsV: []string{flow1, flow2},
				QueueFuncV: func(id string) framework.FlowQueueAccessor {
					if id == flow1 {
						return queue1
					}
					if id == flow2 {
						return queue2
					}
					return nil
				},
			}
			selected, err := policy.SelectQueue(mockBand)
			require.NoError(t, err, "Policy should gracefully skip queue with PeekHead error")
			require.NotNil(t, selected, "Policy should select the valid queue (flow2)")
			assert.Equal(t, flow2, selected.FlowSpec().ID, "Should select the valid queue when the other has a peek error")
		})

		t.Run("QueueComparatorIsNil", func(t *testing.T) {
			t.Parallel()
			queue1 := &frameworkmocks.MockFlowQueueAccessor{
				LenV:        1,
				PeekHeadV:   typesmocks.NewMockQueueItemAccessor(10, "item1", flow1),
				FlowSpecV:   types.FlowSpecification{ID: flow1},
				ComparatorV: nil, // Nil Comparator
			}
			queue2 := &frameworkmocks.MockFlowQueueAccessor{ // A valid queue
				LenV:        1,
				PeekHeadV:   typesmocks.NewMockQueueItemAccessor(20, "item2", flow2),
				FlowSpecV:   types.FlowSpecification{ID: flow2},
				ComparatorV: newTestComparator(),
			}

			mockBand := &frameworkmocks.MockPriorityBandAccessor{
				FlowIDsV: []string{flow1, flow2},
				QueueFuncV: func(id string) framework.FlowQueueAccessor {
					if id == flow1 {
						return queue1
					}
					if id == flow2 {
						return queue2
					}
					return nil
				},
			}

			require.NotNil(t, queue1, "Setup: queue1 should not be nil")
			require.Nil(t, queue1.Comparator(), "Setup: queue1's comparator should be nil for this test")
			assert.Panics(t, func() {
				_, _ = policy.SelectQueue(mockBand)
			}, "Policy should panic if a queue's comparator is nil when it's about to be used.")
		})

		t.Run("ComparatorFuncIsNil", func(t *testing.T) {
			t.Parallel()
			queue1 := &frameworkmocks.MockFlowQueueAccessor{
				LenV:      1,
				PeekHeadV: typesmocks.NewMockQueueItemAccessor(10, "item1", flow1),
				FlowSpecV: types.FlowSpecification{ID: flow1},
				ComparatorV: &frameworkmocks.MockItemComparator{ // Comparator exists
					ScoreTypeV: commonScoreType,
					FuncV:      nil, // But its Func is nil
				},
			}
			queue2 := &frameworkmocks.MockFlowQueueAccessor{ // A valid queue
				LenV:        1,
				PeekHeadV:   typesmocks.NewMockQueueItemAccessor(20, "item2", flow2),
				FlowSpecV:   types.FlowSpecification{ID: flow2},
				ComparatorV: newTestComparator(),
			}
			mockBand := &frameworkmocks.MockPriorityBandAccessor{
				FlowIDsV: []string{flow1, flow2},
				QueueFuncV: func(id string) framework.FlowQueueAccessor {
					if id == flow1 {
						return queue1
					}
					if id == flow2 {
						return queue2
					}
					return nil
				},
			}

			// Similar to nil comparator, current bestHead panics if Func is nil.
			require.NotNil(t, queue1, "Setup: queue1 should not be nil")
			require.NotNil(t, queue1.Comparator(), "Setup: queue1's comparator should NOT be nil for this test")
			require.Nil(t, queue1.Comparator().Func(), "Setup: queue1's comparator Func should be nil for this test")
			assert.Panics(t, func() {
				_, _ = policy.SelectQueue(mockBand)
			}, "Policy should panic if a comparator's func is nil when it's about to be used.")
		})
	})
}
