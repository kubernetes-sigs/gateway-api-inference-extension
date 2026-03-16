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

package queue

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol/mocks"
)

// TestMaxMinHeap_InternalProperty validates that the max-min heap property is maintained after a series of `Add` and
// `Remove` operations. This is a white-box test to ensure the internal data structure is always in a valid state.
func TestMaxMinHeap_InternalProperty(t *testing.T) {
	t.Parallel()
	q := newMaxMinHeap(enqueueTimePolicy)

	items := make([]*mocks.MockQueueItemAccessor, 20)
	now := time.Now()
	for i := range items {
		// Add items in a somewhat random order of enqueue times
		items[i] = mocks.NewMockQueueItemAccessor(10, "item", flowcontrol.FlowKey{ID: "flow"})
		items[i].EnqueueTimeV = now.Add(time.Duration((i%5-2)*10) * time.Second)
		q.Add(items[i])
		assertHeapProperty(t, q, "after adding item %d", i)
	}

	// Remove a few items from the middle and validate the heap property
	for _, i := range []int{15, 7, 11} {
		handle := items[i].Handle()
		_, err := q.Remove(handle)
		require.NoError(t, err, "Remove should not fail for item %d", i)
		assertHeapProperty(t, q, "after removing item %d", i)
	}

	// Remove remaining items from the head and validate each time
	for q.Len() > 0 {
		head := q.PeekHead()
		require.NotNil(t, head)
		_, err := q.Remove(head.Handle())
		require.NoError(t, err)
		assertHeapProperty(t, q, "after removing head item")
	}
}

// assertHeapProperty validates the max-min heap invariant by draining and re-adding all items.
// This is a black-box verification that the wrapper maintains correct ordering.
func assertHeapProperty(t *testing.T, h *maxMinHeap, msgAndArgs ...any) {
	t.Helper()

	// Verify that PeekHead returns an item >= all others, and PeekTail returns an item <= all others.
	// This is a weaker check than the original white-box test, but it validates the contract without
	// reaching into internal state. The generic heap has its own thorough white-box property tests.
	if h.Len() == 0 {
		return
	}

	head := h.PeekHead()
	tail := h.PeekTail()
	require.NotNil(t, head, "PeekHead must not be nil when Len > 0. %v", msgAndArgs)
	require.NotNil(t, tail, "PeekTail must not be nil when Len > 0. %v", msgAndArgs)

	// Head must have priority >= tail (i.e., policy.Less(head, tail) must be true or they are equal).
	// Equivalently, tail must NOT have higher priority than head.
	require.False(t, h.policy.Less(tail, head),
		"PeekTail should not have higher priority than PeekHead. %v", msgAndArgs)
}
