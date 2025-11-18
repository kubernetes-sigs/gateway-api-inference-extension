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
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	typesmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

// TestMaxMinHeap_InternalProperty validates that the max-min heap property is maintained after a series of `Add` and
// `Remove` operations. This is a white-box test to ensure the internal data structure is always in a valid state.
func TestMaxMinHeap_InternalProperty(t *testing.T) {
	t.Parallel()
	q := newMaxMinHeap(enqueueTimeComparator)

	items := make([]*typesmocks.MockQueueItemAccessor, 20)
	now := time.Now()
	for i := range items {
		// Add items in a somewhat random order of enqueue times
		items[i] = typesmocks.NewMockQueueItemAccessor(10, "item", types.FlowKey{ID: "flow"})
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

// assertHeapProperty checks if the slice of items satisfies the max-min heap property.
func assertHeapProperty(t *testing.T, h *maxMinHeap, msgAndArgs ...any) {
	t.Helper()
	if len(h.items) > 0 {
		verifyNode(t, h, 0, msgAndArgs...)
	}
}

// verifyNode recursively checks that the subtree at index `i` satisfies the max-min heap property.
func verifyNode(t *testing.T, h *maxMinHeap, i int, msgAndArgs ...any) {
	t.Helper()
	n := len(h.items)
	if i >= n {
		return
	}

	level := int(math.Floor(math.Log2(float64(i + 1))))
	isMinLevel := level%2 != 0

	leftChild := 2*i + 1
	rightChild := 2*i + 2

	// Check children
	if leftChild < n {
		if isMinLevel {
			require.False(t, h.comparator.Func()(h.items[i], h.items[leftChild]),
				"min-level node %d has child %d with smaller value. %v", i, leftChild, msgAndArgs)
		} else { // isMaxLevel
			require.False(t, h.comparator.Func()(h.items[leftChild], h.items[i]),
				"max-level node %d has child %d with larger value. %v", i, leftChild, msgAndArgs)
		}
		verifyNode(t, h, leftChild, msgAndArgs...)
	}

	if rightChild < n {
		if isMinLevel {
			require.False(t, h.comparator.Func()(h.items[i], h.items[rightChild]),
				"min-level node %d has child %d with smaller value. %v", i, rightChild, msgAndArgs)
		} else { // isMaxLevel
			require.False(t, h.comparator.Func()(h.items[rightChild], h.items[i]),
				"max-level node %d has child %d with larger value. %v", i, rightChild, msgAndArgs)
		}
		verifyNode(t, h, rightChild, msgAndArgs...)
	}
}
