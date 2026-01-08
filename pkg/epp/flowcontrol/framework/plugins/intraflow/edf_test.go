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

package intraflow

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

func TestEDFPolicy_Name(t *testing.T) {
	t.Parallel()
	policy := newEDFPolicy()
	assert.Equal(t, EDFPolicyName, policy.Name())
}

func TestEDFPolicy_RequiredQueueCapabilities(t *testing.T) {
	t.Parallel()
	policy := newEDFPolicy()
	caps := policy.RequiredQueueCapabilities()
	require.Len(t, caps, 1)
	assert.Equal(t, framework.CapabilityPriorityConfigurable, caps[0])
}

func TestEDFPolicy_SelectItem(t *testing.T) {
	t.Parallel()
	policy := newEDFPolicy()

	mockItem := typesmocks.NewMockQueueItemAccessor(1, "item1", testFlowKey)
	mockQueue := &frameworkmocks.MockFlowQueueAccessor{
		PeekHeadV: mockItem,
		LenV:      1,
	}

	item, err := policy.SelectItem(mockQueue)
	require.NoError(t, err)
	assert.Equal(t, mockItem, item, "Should return the head of the queue")
}

func TestEDFComparator_Func(t *testing.T) {
	t.Parallel()
	comparator := &edfComparator{}
	compareFunc := comparator.Func()
	require.NotNil(t, compareFunc)

	now := time.Now()

	// Item A: TTL=10s → deadline = now + 10s
	itemA := typesmocks.NewMockQueueItemAccessor(10, "itemA", testFlowKey)
	itemA.EnqueueTimeV = now
	itemA.EffectiveTTLV = 10 * time.Second

	// Item B: TTL=5s → deadline = now + 5s (earlier than A)
	itemB := typesmocks.NewMockQueueItemAccessor(20, "itemB", testFlowKey)
	itemB.EnqueueTimeV = now.Add(1 * time.Second) // enqueued later, but tighter deadline
	itemB.EffectiveTTLV = 5 * time.Second

	// Item C: TTL <= 0 → treated as far-future deadline
	itemC := typesmocks.NewMockQueueItemAccessor(30, "itemC", testFlowKey)
	itemC.EnqueueTimeV = now.Add(-5 * time.Second) // enqueued earlier, but no deadline
	itemC.EffectiveTTLV = -1 * time.Second

	// Item D: same deadline as B, but enqueued earlier → should win tie-breaker
	itemD := typesmocks.NewMockQueueItemAccessor(40, "itemD", testFlowKey)
	itemD.EnqueueTimeV = now.Add(-1 * time.Second)
	itemD.EffectiveTTLV = 6 * time.Second // deadline = now + 5s (same as B)

	// Item E: another non-deadline item
	itemE := typesmocks.NewMockQueueItemAccessor(40, "itemD", testFlowKey)
	itemE.EnqueueTimeV = now.Add(-10 * time.Second) // earlier than C
	itemE.EffectiveTTLV = 0

	testCases := []struct {
		name     string
		a        types.QueueItemAccessor
		b        types.QueueItemAccessor
		expected bool // true if a should be dispatched before b
	}{
		{"B before A (earlier deadline)", itemB, itemA, true},
		{"A after B", itemA, itemB, false},

		{"Deadline-bound before non-deadline (A vs C)", itemA, itemC, true},
		{"Non-deadline after deadline-bound (C vs A)", itemC, itemA, false},

		{"Tie-breaker: D before B (same deadline, D enqueued earlier)", itemD, itemB, true},
		{"Tie-breaker: B after D", itemB, itemD, false},

		{"Non-deadline items: FCFS (E enqueued earlier than C → E before C)", itemE, itemC, true},
		{"Non-deadline items: C after E", itemC, itemE, false},

		{"a is nil → b wins", nil, itemA, false},
		{"b is nil → a wins", itemA, nil, true},
		{"both nil → false", nil, nil, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.expected, compareFunc(tc.a, tc.b))
		})
	}
}

func TestEDFComparator_ScoreType(t *testing.T) {
	t.Parallel()
	comparator := &edfComparator{}
	assert.Equal(t, string(framework.EDFPriorityScoreType), comparator.ScoreType())
}

func TestCalculateDeadline(t *testing.T) {
	t.Parallel()

	now := time.Now()

	// Valid TTL
	itemWithTTL := typesmocks.NewMockQueueItemAccessor(1, "test", testFlowKey)
	itemWithTTL.EnqueueTimeV = now
	itemWithTTL.EffectiveTTLV = 5 * time.Second

	deadline := calculateDeadline(itemWithTTL)
	assert.Equal(t, now.Add(5*time.Second), deadline)

	// Zero TTL → far future
	itemZeroTTL := typesmocks.NewMockQueueItemAccessor(2, "test2", testFlowKey)
	itemZeroTTL.EnqueueTimeV = now
	itemZeroTTL.EffectiveTTLV = 0

	deadlineZero := calculateDeadline(itemZeroTTL)
	assert.Equal(t, maxDeadlineTime, deadlineZero)

	// Negative TTL → far future
	itemNegTTL := typesmocks.NewMockQueueItemAccessor(3, "test3", testFlowKey)
	itemNegTTL.EnqueueTimeV = now
	itemNegTTL.EffectiveTTLV = -10 * time.Second

	deadlineNeg := calculateDeadline(itemNegTTL)
	assert.Equal(t, maxDeadlineTime, deadlineNeg)
}
