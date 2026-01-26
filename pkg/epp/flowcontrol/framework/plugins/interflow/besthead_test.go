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

package interflow

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	typesmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

func newTestOrderingPolicy() *frameworkmocks.MockOrderingPolicy {
	return &frameworkmocks.MockOrderingPolicy{
		TypedNameV: plugin.TypedName{Type: "enqueue_time_ns_asc"},
		LessFunc: func(a, b types.QueueItemAccessor) bool {
			return a.EnqueueTime().Before(b.EnqueueTime())
		},
	}
}

func newTestBand(queues ...flowcontrol.FlowQueueAccessor) *frameworkmocks.MockPriorityBandAccessor {
	flowKeys := make([]types.FlowKey, 0, len(queues))
	queuesByID := make(map[string]flowcontrol.FlowQueueAccessor, len(queues))
	for _, q := range queues {
		key := q.FlowKey()
		flowKeys = append(flowKeys, key)
		queuesByID[key.ID] = q
	}
	return &frameworkmocks.MockPriorityBandAccessor{
		FlowKeysFunc: func() []types.FlowKey { return flowKeys },
		QueueFunc: func(id string) flowcontrol.FlowQueueAccessor {
			return queuesByID[id]
		},
		IterateQueuesFunc: func(iterator func(flow flowcontrol.FlowQueueAccessor) bool) {
			for _, key := range flowKeys {
				if !iterator(queuesByID[key.ID]) {
					break
				}
			}
		},
	}
}

func TestGlobalStrict_Name(t *testing.T) {
	t.Parallel()
	policy := newGlobalStrict("test-gs")
	assert.Equal(t, "test-gs", policy.TypedName().Name)
	assert.Equal(t, GlobalStrictFairnessPolicyType, policy.TypedName().Type)
}

func TestGlobalStrict_Pick(t *testing.T) {
	t.Parallel()

	const flow1ID = "flow1"
	policy := newGlobalStrict("")
	ctx := context.Background()
	now := time.Now()

	itemBetter := typesmocks.NewMockQueueItemAccessor(10, "itemBetter", flow1Key)
	itemBetter.EnqueueTimeV = now.Add(-10 * time.Second)
	itemWorse := typesmocks.NewMockQueueItemAccessor(20, "itemWorse", flow2Key)
	itemWorse.EnqueueTimeV = now.Add(-5 * time.Second)

	queue1 := &frameworkmocks.MockFlowQueueAccessor{
		LenV:            1,
		PeekHeadV:       itemBetter,
		FlowKeyV:        flow1Key,
		OrderingPolicyV: newTestOrderingPolicy(),
	}
	queue2 := &frameworkmocks.MockFlowQueueAccessor{
		LenV:            1,
		PeekHeadV:       itemWorse,
		FlowKeyV:        flow2Key,
		OrderingPolicyV: newTestOrderingPolicy(),
	}
	queueEmpty := &frameworkmocks.MockFlowQueueAccessor{
		LenV:            0,
		PeekHeadV:       nil,
		FlowKeyV:        types.FlowKey{ID: "flowEmpty"},
		OrderingPolicyV: newTestOrderingPolicy(),
	}

	testCases := []struct {
		name            string
		band            flowcontrol.PriorityBandAccessor
		expectedQueueID string
		expectedErr     error
		shouldPanic     bool
	}{
		{
			name:            "BasicSelection_TwoQueues",
			band:            newTestBand(queue1, queue2),
			expectedQueueID: flow1ID,
		},
		{
			name:            "IgnoresEmptyQueues",
			band:            newTestBand(queue1, queueEmpty, queue2),
			expectedQueueID: flow1ID,
		},
		{
			name:            "SingleNonEmptyQueue",
			band:            newTestBand(queue1),
			expectedQueueID: flow1ID,
		},
		{
			name: "OrderingPolicyCompatibility",
			band: newTestBand(
				&frameworkmocks.MockFlowQueueAccessor{
					LenV:      1,
					PeekHeadV: itemBetter,
					FlowKeyV:  flow1Key,
					OrderingPolicyV: &frameworkmocks.MockOrderingPolicy{
						TypedNameV: plugin.TypedName{Type: "typeA"},
						LessFunc: func(a, b types.QueueItemAccessor) bool {
							return a.EnqueueTime().Before(b.EnqueueTime())
						},
					},
				},
				&frameworkmocks.MockFlowQueueAccessor{
					LenV:      1,
					PeekHeadV: itemWorse,
					FlowKeyV:  flow2Key,
					OrderingPolicyV: &frameworkmocks.MockOrderingPolicy{
						TypedNameV: plugin.TypedName{Type: "typeB"},
						LessFunc: func(a, b types.QueueItemAccessor) bool {
							return a.EnqueueTime().Before(b.EnqueueTime())
						},
					},
				},
			),
			expectedErr: flowcontrol.ErrIncompatiblePriorityType,
		},
		{
			name: "OrderingPolicyIsNil",
			band: newTestBand(
				&frameworkmocks.MockFlowQueueAccessor{
					LenV:            1,
					PeekHeadV:       itemBetter,
					FlowKeyV:        flow1Key,
					OrderingPolicyV: nil,
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
					LenV:            0,
					PeekHeadV:       nil,
					FlowKeyV:        types.FlowKey{ID: "flowEmpty2"},
					OrderingPolicyV: newTestOrderingPolicy(),
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
				assert.Panics(t, func() { _, _ = policy.Pick(ctx, tc.band) }, "Pick should panic for this edge case")
				return
			}

			selected, err := policy.Pick(ctx, tc.band)

			if tc.expectedErr != nil {
				require.Error(t, err, "Pick should return an error")
				assert.ErrorIs(t, err, tc.expectedErr, "The returned error should match the expected error type")
				assert.Nil(t, selected, "No queue should be selected when an error occurs")
			} else {
				require.NoError(t, err, "Pick should not return an error for valid inputs")
				if tc.expectedQueueID == "" {
					assert.Nil(t, selected, "No queue should be selected")
				} else {
					require.NotNil(t, selected, "A queue should have been selected")
					assert.Equal(t, tc.expectedQueueID, selected.FlowKey().ID, "The selected queue should have the expected ID")
				}
			}
		})
	}
}
