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
	req := typesmocks.NewMockFlowControlRequest(100, "req-1", types.FlowKey{}, context.Background())

	item := NewItem(req, time.Minute, time.Now())

	require.NotNil(t, item, "NewItem should not return a nil item")
	assert.False(t, item.isFinalized(), "a new item must not be in a finalized state")
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
	req := typesmocks.NewMockFlowControlRequest(100, "req-1", types.FlowKey{}, context.Background())
	item := NewItem(req, time.Minute, time.Now())
	expectedErr := errors.New("first-error")

	item.Finalize(types.QueueOutcomeEvictedTTL, expectedErr)
	item.Finalize(types.QueueOutcomeDispatched, nil) // Should take no effect

	assert.True(t, item.isFinalized(), "item must be in a finalized state after a call to finalize()")
	select {
	case finalState, ok := <-item.Done():
		require.True(t, ok, "Done() channel should be readable with a value, not just closed")
		assert.Equal(t, types.QueueOutcomeEvictedTTL, finalState.Outcome,
			"the outcome from Done() must match the first finalized outcome")
		assert.Equal(t, expectedErr, finalState.Err, "the error from Done() must match the first finalized error")
	case <-time.After(50 * time.Millisecond):
		t.Fatal("Done() channel must not block after finalization")
	}
}
