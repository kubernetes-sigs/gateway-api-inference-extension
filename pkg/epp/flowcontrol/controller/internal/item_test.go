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

package internal

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	typesmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

func TestFlowItem_New(t *testing.T) {
	t.Parallel()
	req := typesmocks.NewMockFlowControlRequest(100, "req-1", types.FlowKey{})

	enqueueTime := time.Now()
	item := NewItem(req, time.Minute, enqueueTime)

	require.NotNil(t, item, "NewItem should not return a nil item")
	assert.Equal(t, enqueueTime, item.EnqueueTime(), "EnqueueTime should be populated")
	assert.Equal(t, time.Minute, item.EffectiveTTL(), "EffectiveTTL should be populated")
	assert.Same(t, req, item.OriginalRequest(), "OriginalRequest should be populated")
	assert.Nil(t, item.FinalState(), "a new item must not have a final state")
	select {
	case <-item.Done():
		t.Fatal("Done() channel for a new item must block, but it was closed")
	default:
		// This is the expected path, as the channel would have blocked.
	}
}

func TestFlowItem_Handle(t *testing.T) {
	t.Parallel()
	item := &FlowItem{}
	handle := &typesmocks.MockQueueItemHandle{}
	item.SetHandle(handle)
	assert.Same(t, handle, item.Handle(), "Handle() must retrieve the identical handle instance set by SetHandle()")
}

func TestFlowItem_Finalize_Idempotency(t *testing.T) {
	t.Parallel()
	now := time.Now()
	req := typesmocks.NewMockFlowControlRequest(100, "req-1", types.FlowKey{})

	testCases := []struct {
		name            string
		firstCall       func(item *FlowItem)
		secondCall      func(item *FlowItem)
		expectedOutcome types.QueueOutcome
		expectedErrIs   error
	}{
		{
			name: "Finalize then Finalize",
			firstCall: func(item *FlowItem) {
				item.Finalize(types.ErrTTLExpired)
			},
			secondCall: func(item *FlowItem) {
				item.Finalize(context.Canceled)
			},
			expectedOutcome: types.QueueOutcomeRejectedOther,
			expectedErrIs:   types.ErrTTLExpired,
		},
		{
			name: "Finalize then FinalizeWithOutcome",
			firstCall: func(item *FlowItem) {
				item.Finalize(types.ErrTTLExpired)
			},
			secondCall: func(item *FlowItem) {
				item.FinalizeWithOutcome(types.QueueOutcomeDispatched, nil)
			},
			expectedOutcome: types.QueueOutcomeRejectedOther,
			expectedErrIs:   types.ErrTTLExpired,
		},
		{
			name: "FinalizeWithOutcome then FinalizeWithOutcome",
			firstCall: func(item *FlowItem) {
				item.FinalizeWithOutcome(types.QueueOutcomeDispatched, nil)
			},
			secondCall: func(item *FlowItem) {
				item.FinalizeWithOutcome(types.QueueOutcomeRejectedCapacity, errors.New("rejected"))
			},
			expectedOutcome: types.QueueOutcomeDispatched,
			expectedErrIs:   nil,
		},
		{
			name: "FinalizeWithOutcome then Finalize",
			firstCall: func(item *FlowItem) {
				item.FinalizeWithOutcome(types.QueueOutcomeDispatched, nil)
			},
			secondCall: func(item *FlowItem) {
				item.Finalize(types.ErrTTLExpired)
			},
			expectedOutcome: types.QueueOutcomeDispatched,
			expectedErrIs:   nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			item := NewItem(req, time.Minute, now)

			// First call
			tc.firstCall(item)

			// Second call
			tc.secondCall(item)

			// Check FinalState()
			finalState := item.FinalState()
			require.NotNil(t, finalState, "FinalState should not be nil")
			assert.Equal(t, tc.expectedOutcome, finalState.Outcome, "Outcome should match the first call")
			if tc.expectedErrIs != nil {
				assert.ErrorIs(t, finalState.Err, tc.expectedErrIs, "Error should match the first call")
			} else {
				assert.NoError(t, finalState.Err, "Error should be nil")
			}

			// Check Done channel
			select {
			case state, ok := <-item.Done():
				require.True(t, ok, "Done channel should be readable")
				assert.Equal(t, tc.expectedOutcome, state.Outcome, "Done channel outcome should match the first call")
				if tc.expectedErrIs != nil {
					assert.ErrorIs(t, state.Err, tc.expectedErrIs, "Done channel error should match the first call")
				} else {
					assert.NoError(t, state.Err, "Done channel error should be nil")
				}
			case <-time.After(50 * time.Millisecond):
				t.Fatal("Done channel should have received the state")
			}
		})
	}
}

func TestFlowItem_Finalize_InferOutcome(t *testing.T) {
	t.Parallel()
	now := time.Now()

	testCases := []struct {
		name          string
		cause         error
		isQueued      bool
		expectOutcome types.QueueOutcome
		expectErrIs   error
	}{
		{
			name:          "queued TTL expired",
			cause:         types.ErrTTLExpired,
			isQueued:      true,
			expectOutcome: types.QueueOutcomeEvictedTTL,
			expectErrIs:   types.ErrTTLExpired,
		},
		{
			name:          "queued context cancelled",
			cause:         context.Canceled,
			isQueued:      true,
			expectOutcome: types.QueueOutcomeEvictedContextCancelled,
			expectErrIs:   types.ErrContextCancelled,
		},
		{
			name:          "queued other error",
			cause:         errors.New("other cause"),
			isQueued:      true,
			expectOutcome: types.QueueOutcomeEvictedOther,
			expectErrIs:   types.ErrEvicted,
		},
		{
			name:          "not queued TTL expired",
			cause:         types.ErrTTLExpired,
			isQueued:      false,
			expectOutcome: types.QueueOutcomeRejectedOther,
			expectErrIs:   types.ErrTTLExpired,
		},
		{
			name:          "not queued context cancelled",
			cause:         context.Canceled,
			isQueued:      false,
			expectOutcome: types.QueueOutcomeRejectedOther,
			expectErrIs:   types.ErrContextCancelled,
		},
		{
			name:          "nil cause queued",
			cause:         nil,
			isQueued:      true,
			expectOutcome: types.QueueOutcomeEvictedOther,
			expectErrIs:   types.ErrEvicted,
		},
		{
			name:          "nil cause not queued",
			cause:         nil,
			isQueued:      false,
			expectOutcome: types.QueueOutcomeRejectedOther,
			expectErrIs:   types.ErrRejected,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req := typesmocks.NewMockFlowControlRequest(100, "req-1", types.FlowKey{})
			item := NewItem(req, time.Minute, now)
			if tc.isQueued {
				item.SetHandle(&typesmocks.MockQueueItemHandle{})
			}

			item.Finalize(tc.cause)

			finalState := item.FinalState()
			require.NotNil(t, finalState, "FinalState should not be nil")
			assert.Equal(t, tc.expectOutcome, finalState.Outcome, "Unexpected outcome")
			require.Error(t, finalState.Err, "An error should be set")
			assert.ErrorIs(t, finalState.Err, tc.expectErrIs, "Unexpected error type")
		})
	}
}
