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

package fcfs

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	typesmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

var testFlowKey = types.FlowKey{ID: "test-flow", Priority: 0}

func TestFCFS_Name(t *testing.T) {
	t.Parallel()
	policy := newFCFS()
	assert.Equal(t, FCFSPolicyName, policy.Name())
}

func TestFCFS_RequiredQueueCapabilities(t *testing.T) {
	t.Parallel()
	policy := newFCFS()
	caps := policy.RequiredQueueCapabilities()
	require.Empty(t, caps, "No required capabilities should be returned")
}

func TestFCFS_SelectItem(t *testing.T) {
	t.Parallel()
	// Note: The conformance suite validates the policy's contract for nil and empty queues.
	// This unit test focuses on the policy-specific success path.
	policy := newFCFS()

	mockItem := typesmocks.NewMockQueueItemAccessor(1, "item1", testFlowKey)
	mockQueue := &frameworkmocks.MockFlowQueueAccessor{
		PeekHeadV: mockItem,
		LenV:      1,
	}

	item, err := policy.SelectItem(mockQueue)
	require.NoError(t, err)
	assert.Equal(t, mockItem, item, "Should return the item from the head of the queue")
}

func TestEnqueueTimeComparator_Func(t *testing.T) {
	t.Parallel()
	comparator := &enqueueTimeComparator{} // Test the internal comparator directly
	compareFunc := comparator.Func()
	require.NotNil(t, compareFunc)

	now := time.Now()
	itemA := typesmocks.NewMockQueueItemAccessor(10, "itemA", testFlowKey)
	itemA.EnqueueTimeV = now

	itemB := typesmocks.NewMockQueueItemAccessor(20, "itemB", testFlowKey)
	itemB.EnqueueTimeV = now.Add(time.Second) // B is later than A

	itemC := typesmocks.NewMockQueueItemAccessor(30, "itemC", testFlowKey)
	itemC.EnqueueTimeV = now // C is same time as A

	testCases := []struct {
		name     string
		item1    types.QueueItemAccessor
		item2    types.QueueItemAccessor
		expected bool // true if item1 is higher priority (earlier) than item2
	}{
		{"A before B", itemA, itemB, true},
		{"B after A", itemB, itemA, false},
		{"A same as C (A not strictly before C)", itemA, itemC, false},
		{"C same as A (C not strictly before A)", itemC, itemA, false},
		{"A vs nil B (A is preferred)", itemA, nil, true},
		{"nil A vs B (B is preferred)", nil, itemB, false},
		{"nil A vs nil B (no preference)", nil, nil, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.expected, compareFunc(tc.item1, tc.item2))
		})
	}
}

func TestEnqueueTimeComparator_ScoreType(t *testing.T) {
	t.Parallel()
	comparator := &enqueueTimeComparator{}
	assert.Equal(t, string(framework.EnqueueTimePriorityScoreType), comparator.ScoreType())
}
