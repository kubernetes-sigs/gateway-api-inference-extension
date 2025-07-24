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
	flow1           = "flow1"
	flow2           = "flow2"
	commonScoreType = "enqueue_time_ns_asc"
)

// enqueueTimeComparatorFunc is a test utility. Lower enqueue time is better.
func enqueueTimeComparatorFunc(a, b types.QueueItemAccessor) bool {
	return a.EnqueueTime().Before(b.EnqueueTime())
}

func newTestComparator() *frameworkmocks.MockItemComparator {
	return &frameworkmocks.MockItemComparator{
		ScoreTypeV: commonScoreType,
		FuncV:      enqueueTimeComparatorFunc,
	}
}

func newTestBand(queues ...framework.FlowQueueAccessor) *frameworkmocks.MockPriorityBandAccessor {
	flowIDs := make([]string, 0, len(queues))
	queuesByID := make(map[string]framework.FlowQueueAccessor, len(queues))
	for _, q := range queues {
		flowIDs = append(flowIDs, q.FlowSpec().ID)
		queuesByID[q.FlowSpec().ID] = q
	}
	return &frameworkmocks.MockPriorityBandAccessor{
		FlowIDsFunc: func() []string { return flowIDs },
		QueueFunc: func(id string) framework.FlowQueueAccessor {
			return queuesByID[id]
		},
		IterateQueuesFunc: func(iterator func(queue framework.FlowQueueAccessor) bool) {
			for _, id := range flowIDs {
				if !iterator(queuesByID[id]) {
					break
				}
			}
		},
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
	now := time.Now()

	itemBetter := typesmocks.NewMockQueueItemAccessor(10, "itemBetter", flow1)
	itemBetter.EnqueueTimeV = now.Add(-10 * time.Second)
	itemWorse := typesmocks.NewMockQueueItemAccessor(20, "itemWorse", flow2)
	itemWorse.EnqueueTimeV = now.Add(-5 * time.Second)

	queue1 := &frameworkmocks.MockFlowQueueAccessor{
		LenV:        1,
		PeekHeadV:   itemBetter,
		FlowSpecV:   types.FlowSpecification{ID: flow1},
		ComparatorV: newTestComparator(),
	}
	queue2 := &frameworkmocks.MockFlowQueueAccessor{
		LenV:        1,
		PeekHeadV:   itemWorse,
		FlowSpecV:   types.FlowSpecification{ID: flow2},
		ComparatorV: newTestComparator(),
	}
	queueEmpty := &frameworkmocks.MockFlowQueueAccessor{
		LenV:         0,
		PeekHeadErrV: framework.ErrQueueEmpty,
		FlowSpecV:    types.FlowSpecification{ID: "flowEmpty"},
		ComparatorV:  newTestComparator(),
	}

	testCases := []struct {
		name            string
		band            framework.PriorityBandAccessor
		expectedQueueID string
		expectedErr     error
		shouldPanic     bool
	}{
		{
			name:            "BasicSelection_TwoQueues",
			band:            newTestBand(queue1, queue2),
			expectedQueueID: flow1,
		},
		{
			name:            "IgnoresEmptyQueues",
			band:            newTestBand(queue1, queueEmpty, queue2),
			expectedQueueID: flow1,
		},
		{
			name:            "SingleNonEmptyQueue",
			band:            newTestBand(queue1),
			expectedQueueID: flow1,
		},
		{
			name: "ComparatorCompatibility",
			band: newTestBand(
				&frameworkmocks.MockFlowQueueAccessor{
					LenV:        1,
					PeekHeadV:   itemBetter,
					FlowSpecV:   types.FlowSpecification{ID: flow1},
					ComparatorV: &frameworkmocks.MockItemComparator{ScoreTypeV: "typeA", FuncV: enqueueTimeComparatorFunc},
				},
				&frameworkmocks.MockFlowQueueAccessor{
					LenV:        1,
					PeekHeadV:   itemWorse,
					FlowSpecV:   types.FlowSpecification{ID: flow2},
					ComparatorV: &frameworkmocks.MockItemComparator{ScoreTypeV: "typeB", FuncV: enqueueTimeComparatorFunc},
				},
			),
			expectedErr: framework.ErrIncompatiblePriorityType,
		},
		{
			name: "QueuePeekHeadErrors",
			band: newTestBand(
				&frameworkmocks.MockFlowQueueAccessor{
					LenV:         1,
					PeekHeadErrV: errors.New("peek error"),
					FlowSpecV:    types.FlowSpecification{ID: flow1},
					ComparatorV:  newTestComparator(),
				},
				queue2,
			),
			expectedQueueID: flow2,
		},
		{
			name: "QueueComparatorIsNil",
			band: newTestBand(
				&frameworkmocks.MockFlowQueueAccessor{
					LenV:        1,
					PeekHeadV:   itemBetter,
					FlowSpecV:   types.FlowSpecification{ID: flow1},
					ComparatorV: nil,
				},
				queue2,
			),
			shouldPanic: true,
		},
		{
			name: "ComparatorFuncIsNil",
			band: newTestBand(
				&frameworkmocks.MockFlowQueueAccessor{
					LenV:        1,
					PeekHeadV:   itemBetter,
					FlowSpecV:   types.FlowSpecification{ID: flow1},
					ComparatorV: &frameworkmocks.MockItemComparator{ScoreTypeV: commonScoreType, FuncV: nil},
				},
				queue2,
			),
			shouldPanic: true,
		},
		{
			name: "AllQueuesEmpty",
			band: newTestBand(
				queueEmpty,
				&frameworkmocks.MockFlowQueueAccessor{
					LenV:         0,
					PeekHeadErrV: framework.ErrQueueEmpty,
					FlowSpecV:    types.FlowSpecification{ID: "flowEmpty2"},
					ComparatorV:  newTestComparator(),
				},
			),
		},
		{
			name: "NilBand",
			band: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			if tc.shouldPanic {
				assert.Panics(t, func() { _, _ = policy.SelectQueue(tc.band) }, "SelectQueue should panic for this edge case")
				return
			}

			selected, err := policy.SelectQueue(tc.band)

			if tc.expectedErr != nil {
				require.Error(t, err, "SelectQueue should return an error")
				assert.ErrorIs(t, err, tc.expectedErr, "The returned error should match the expected error type")
				assert.Nil(t, selected, "No queue should be selected when an error occurs")
			} else {
				require.NoError(t, err, "SelectQueue should not return an error for valid inputs")
				if tc.expectedQueueID == "" {
					assert.Nil(t, selected, "No queue should be selected")
				} else {
					require.NotNil(t, selected, "A queue should have been selected")
					assert.Equal(t, tc.expectedQueueID, selected.FlowSpec().ID, "The selected queue should have the expected ID")
				}
			}
		})
	}
}
