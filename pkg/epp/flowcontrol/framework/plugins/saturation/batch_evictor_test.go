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

package saturation

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
)

// mockHandle implements QueueItemHandle for testing
type mockHandle struct{ id string }

func (m *mockHandle) Handle() any         { return m.id }
func (m *mockHandle) Invalidate()         { /* no-op */ }
func (m *mockHandle) IsInvalidated() bool { return false }

// mockEvictableQueue is a minimal mock for testing eviction
type mockEvictableQueue struct {
	removedHandles []string
}

func (m *mockEvictableQueue) Remove(handle flowcontrol.QueueItemHandle) (flowcontrol.QueueItemAccessor, error) {
	m.removedHandles = append(m.removedHandles, handle.Handle().(string))
	return nil, nil
}

// mockQueueItem is a minimal mock for testing
type mockQueueItem struct{ handle flowcontrol.QueueItemHandle }

func (m *mockQueueItem) Handle() flowcontrol.QueueItemHandle             { return m.handle }
func (m *mockQueueItem) OriginalRequest() flowcontrol.FlowControlRequest { return nil }
func (m *mockQueueItem) EnqueueTime() time.Time                          { return time.Time{} }
func (m *mockQueueItem) EffectiveTTL() time.Duration                     { return 0 }
func (m *mockQueueItem) SetHandle(handle flowcontrol.QueueItemHandle)    { m.handle = handle }

func queueItem(id string) *mockQueueItem { return &mockQueueItem{handle: &mockHandle{id: id}} }

func TestBatchEvictor_OnlyEvictsNegativePriorities(t *testing.T) {
	evictor := NewBatchEvictor()
	ctx := context.Background()

	queue := &mockEvictableQueue{}

	// Schedule items at various priorities
	evictor.ScheduleEvictionCandidate(ctx, queueItem("item1"), queue, 10, 0.8)  // high priority - should NOT evict
	evictor.ScheduleEvictionCandidate(ctx, queueItem("item2"), queue, 0, 0.8)   // zero priority - should NOT evict
	evictor.ScheduleEvictionCandidate(ctx, queueItem("item3"), queue, -5, 0.8)  // negative priority - SHOULD evict
	evictor.ScheduleEvictionCandidate(ctx, queueItem("item4"), queue, -10, 0.8) // negative priority - SHOULD evict

	evicted, err := evictor.ProcessScheduled(ctx)
	require.NoError(t, err)
	require.Equal(t, 2, evicted, "Expected 2 evictions, got %d", evicted)
	require.Equal(t, 2, len(queue.removedHandles), "Expected 2 items removed from queue, got %d", len(queue.removedHandles))
}

func TestBatchEvictor_EvictsInLIFOOrder(t *testing.T) {
	evictor := NewBatchEvictor()
	ctx := context.Background()

	queue := &mockEvictableQueue{}

	// Schedule items in order (FIFO)
	evictor.ScheduleEvictionCandidate(ctx, queueItem("first"), queue, -10, 0.8)
	evictor.ScheduleEvictionCandidate(ctx, queueItem("second"), queue, -10, 0.8)
	evictor.ScheduleEvictionCandidate(ctx, queueItem("third"), queue, -10, 0.8)

	evicted, err := evictor.ProcessScheduled(ctx)
	require.NoError(t, err)
	require.Equal(t, 3, evicted, "Expected 3 evictions, got %d", evicted)
	require.Equal(t, 3, len(queue.removedHandles), "Expected 3 items removed from queue, got %d", len(queue.removedHandles))
	require.Equal(t, "third", queue.removedHandles[0])
	require.Equal(t, "second", queue.removedHandles[1])
	require.Equal(t, "first", queue.removedHandles[2])
}

func TestBatchEvictor_ClearsScheduleAfterProcessing(t *testing.T) {
	evictor := NewBatchEvictor()
	ctx := context.Background()

	queue := &mockEvictableQueue{}

	// Schedule and process
	evictor.ScheduleEvictionCandidate(ctx, queueItem("item1"), queue, -10, 0.8)
	evicted, _ := evictor.ProcessScheduled(ctx)
	require.Equal(t, 1, evicted, "Expected 1 eviction, got %d", evicted)

	// Process again - should evict nothing (schedule was cleared)
	evicted, _ = evictor.ProcessScheduled(ctx)
	require.Equal(t, 0, evicted, "Expected 0 evictions on second call (schedule should be cleared), got %d", evicted)
}

func TestBatchEvictor_EmptySchedule(t *testing.T) {
	evictor := NewBatchEvictor()
	ctx := context.Background()

	// Process with no scheduled candidates
	evicted, err := evictor.ProcessScheduled(ctx)
	require.NoError(t, err)
	require.Equal(t, 0, evicted, "Expected 0 evictions on second call (no candidates), got %d", evicted)
}
