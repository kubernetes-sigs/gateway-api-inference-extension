/*
Copyright 2026 The Kubernetes Authors.

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

package eviction

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// --- Test policy implementations ---

type testOrdering struct{}

func (t *testOrdering) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "test-ordering", Name: "test"}
}

func (t *testOrdering) Less(a, b *flowcontrol.EvictionItem) bool {
	if a.Priority != b.Priority {
		return a.Priority < b.Priority
	}
	return a.DispatchTime.After(b.DispatchTime)
}

type testFilter struct {
	threshold int
}

func (t *testFilter) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "test-filter", Name: "test"}
}

func (t *testFilter) Accept(item *flowcontrol.EvictionItem) bool {
	return item.Priority < t.threshold
}

type acceptAllFilter struct{}

func (a *acceptAllFilter) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "accept-all", Name: "test"}
}

func (a *acceptAllFilter) Accept(_ *flowcontrol.EvictionItem) bool { return true }

// --- Helpers ---

func newItem(id string, priority int, dispatchOffset time.Duration) *flowcontrol.EvictionItem {
	return &flowcontrol.EvictionItem{
		RequestID:    id,
		Priority:     priority,
		DispatchTime: time.Now().Add(dispatchOffset),
	}
}

// --- Tests ---

func TestEvictionQueue_TrackAndUntrack(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                  string
		filter                flowcontrol.EvictionFilterPolicy
		trackItems            []*flowcontrol.EvictionItem
		untrackIDs            []string
		wantInFlight          int      // expected after track+untrack, before pop
		wantEvictable         int      // expected after track+untrack, before pop
		wantPopIDs            []string // if set, PopN is called and order is verified
		wantInFlightAfterPop  int      // expected after PopN (-1 to skip check)
		wantEvictableAfterPop int      // expected after PopN (-1 to skip check)
	}{
		{
			name:                  "track evictable and untrack",
			filter:                &acceptAllFilter{},
			trackItems:            []*flowcontrol.EvictionItem{newItem("req-1", -1, 0)},
			untrackIDs:            []string{"req-1"},
			wantInFlight:          0,
			wantEvictable:         0,
			wantInFlightAfterPop:  -1,
			wantEvictableAfterPop: -1,
		},
		{
			name:                  "track non-evictable goes to inflight only",
			filter:                &testFilter{threshold: 0},
			trackItems:            []*flowcontrol.EvictionItem{newItem("normal", 5, 0)},
			wantInFlight:          1,
			wantEvictable:         0,
			wantInFlightAfterPop:  -1,
			wantEvictableAfterPop: -1,
		},
		{
			name:   "mixed evictable and non-evictable",
			filter: &testFilter{threshold: 0},
			trackItems: []*flowcontrol.EvictionItem{
				newItem("sheddable", -1, 0),
				newItem("normal", 5, 0),
			},
			wantInFlight:          2,
			wantEvictable:         1,
			wantPopIDs:            []string{"sheddable"},
			wantInFlightAfterPop:  1, // "normal" remains in allInFlight
			wantEvictableAfterPop: 0,
		},
		{
			name:                  "untrack non-evictable cleans inflight",
			filter:                &testFilter{threshold: 0},
			trackItems:            []*flowcontrol.EvictionItem{newItem("normal", 5, 0)},
			untrackIDs:            []string{"normal"},
			wantInFlight:          0,
			wantEvictable:         0,
			wantInFlightAfterPop:  -1,
			wantEvictableAfterPop: -1,
		},
		{
			name:                  "untrack non-existent is no-op",
			filter:                &acceptAllFilter{},
			trackItems:            []*flowcontrol.EvictionItem{newItem("req-1", -1, 0)},
			untrackIDs:            []string{"does-not-exist"},
			wantInFlight:          1,
			wantEvictable:         1,
			wantInFlightAfterPop:  -1,
			wantEvictableAfterPop: -1,
		},
		{
			name:   "untrack evictable removes from heap",
			filter: &acceptAllFilter{},
			trackItems: []*flowcontrol.EvictionItem{
				newItem("req-1", -1, 0),
				newItem("req-2", -2, 0),
				newItem("req-3", -3, 0),
			},
			untrackIDs:            []string{"req-3"},
			wantInFlight:          2,
			wantEvictable:         2,
			wantPopIDs:            []string{"req-2", "req-1"},
			wantInFlightAfterPop:  0,
			wantEvictableAfterPop: 0,
		},
		{
			name:   "duplicate track replaces not doubles",
			filter: &acceptAllFilter{},
			trackItems: []*flowcontrol.EvictionItem{
				newItem("req-1", -1, 0),
				newItem("req-1", -1, 0), // duplicate
			},
			wantInFlight:          1,
			wantEvictable:         1,
			wantPopIDs:            []string{"req-1"},
			wantInFlightAfterPop:  0,
			wantEvictableAfterPop: 0,
		},
		{
			name:   "re-track with different priority updates heap",
			filter: &acceptAllFilter{},
			trackItems: []*flowcontrol.EvictionItem{
				newItem("req-a", 5, 0),
				newItem("req-a", -10, 0), // re-track with lower priority
				newItem("req-b", -5, 0),
			},
			wantInFlight:          2,
			wantEvictable:         2,
			wantPopIDs:            []string{"req-a", "req-b"}, // req-a (-10) before req-b (-5)
			wantInFlightAfterPop:  0,
			wantEvictableAfterPop: 0,
		},
		{
			name:   "pop order across many priorities",
			filter: &acceptAllFilter{},
			trackItems: []*flowcontrol.EvictionItem{
				newItem("p5", 5, 0),
				newItem("p1", 1, 0),
				newItem("p3", 3, 0),
				newItem("p-1", -1, 0),
				newItem("p0", 0, 0),
			},
			wantInFlight:          5,
			wantEvictable:         5,
			wantPopIDs:            []string{"p-1", "p0", "p1", "p3", "p5"},
			wantInFlightAfterPop:  0,
			wantEvictableAfterPop: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			q := NewEvictionQueue(&testOrdering{}, tt.filter)

			for _, item := range tt.trackItems {
				q.Track(item)
			}
			for _, id := range tt.untrackIDs {
				q.Untrack(id)
			}

			// Verify counts before popping.
			assert.Equal(t, tt.wantInFlight, q.InFlightLen(), "InFlightLen")
			assert.Equal(t, tt.wantEvictable, q.EvictableLen(), "EvictableLen")

			// Verify pop order and post-pop state if specified.
			if tt.wantPopIDs != nil {
				evicted := q.PopN(100)
				require.Len(t, evicted, len(tt.wantPopIDs))
				for i, wantID := range tt.wantPopIDs {
					assert.Equal(t, wantID, evicted[i].RequestID, "evicted[%d]", i)
				}
				if tt.wantInFlightAfterPop >= 0 {
					assert.Equal(t, tt.wantInFlightAfterPop, q.InFlightLen(), "InFlightLen after PopN")
				}
				if tt.wantEvictableAfterPop >= 0 {
					assert.Equal(t, tt.wantEvictableAfterPop, q.EvictableLen(), "EvictableLen after PopN")
				}
			}
		})
	}
}

func TestEvictionQueue_PopN_TiebreakByDispatchTime(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &acceptAllFilter{})

	q.Track(newItem("newer", 0, 100*time.Millisecond))
	q.Track(newItem("oldest", 0, -100*time.Millisecond))
	q.Track(newItem("middle", 0, 0))

	evicted := q.PopN(3)
	require.Len(t, evicted, 3)

	assert.Equal(t, "newer", evicted[0].RequestID)
	assert.Equal(t, "middle", evicted[1].RequestID)
	assert.Equal(t, "oldest", evicted[2].RequestID)
}

func TestEvictionQueue_PopN_Bounds(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &acceptAllFilter{})

	assert.Empty(t, q.PopN(5), "PopN on empty queue should return empty slice")
	assert.Empty(t, q.PopN(0), "PopN(0) should return empty slice")

	q.Track(newItem("req-1", 0, 0))
	q.Track(newItem("req-2", 0, time.Millisecond))

	assert.Empty(t, q.PopN(0), "PopN(0) with items should return empty slice")
	assert.Equal(t, 2, q.EvictableLen(), "PopN(0) should not remove items")

	evicted := q.PopN(10)
	assert.Len(t, evicted, 2, "PopN should return all items when n > heap size")
	assert.Equal(t, 0, q.EvictableLen())
	assert.Equal(t, 0, q.InFlightLen())
}

func TestEvictionQueue_Peek(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &acceptAllFilter{})

	assert.Nil(t, q.Peek(), "Peek on empty queue should return nil")

	q.Track(newItem("high", 5, 0))
	q.Track(newItem("low", -1, 0))
	q.Track(newItem("mid", 2, 0))

	peeked := q.Peek()
	require.NotNil(t, peeked)
	assert.Equal(t, "low", peeked.RequestID, "Peek should return the most-evictable item")
	assert.Equal(t, 3, q.EvictableLen(), "Peek should not change evictable count")

	// Mutating the copy should not corrupt the heap.
	peeked.Priority = 999
	peeked2 := q.Peek()
	assert.Equal(t, -1, peeked2.Priority, "Mutating Peek result should not affect the heap")
}

func TestEvictionQueue_Concurrency(t *testing.T) {
	t.Parallel()
	q := NewEvictionQueue(&testOrdering{}, &acceptAllFilter{})

	const goroutines = 10
	const opsPerGoroutine = 100

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := range goroutines {
		go func(id int) {
			defer wg.Done()
			for i := range opsPerGoroutine {
				reqID := fmt.Sprintf("req-%d-%d", id, i)
				item := newItem(reqID, id, time.Duration(i)*time.Millisecond)

				switch i % 4 {
				case 0:
					q.Track(item)
				case 1:
					q.Track(item)
					q.Untrack(reqID)
				case 2:
					q.PopN(1)
				case 3:
					q.EvictableLen()
					q.InFlightLen()
					q.Peek()
				}
			}
		}(g)
	}

	wg.Wait()

	// Verify invariants after concurrent operations.
	inFlight := q.InFlightLen()
	evictable := q.EvictableLen()
	assert.GreaterOrEqual(t, inFlight, 0)
	assert.GreaterOrEqual(t, evictable, 0)
	assert.GreaterOrEqual(t, inFlight, evictable,
		"In-flight count should always be >= evictable count")

	// Drain remaining items and verify they come out in order.
	remaining := q.PopN(inFlight + 1)
	for i := 1; i < len(remaining); i++ {
		assert.LessOrEqual(t, remaining[i-1].Priority, remaining[i].Priority,
			"Remaining items should pop in priority order")
	}
	assert.Equal(t, 0, q.EvictableLen(), "Queue should be empty after draining")
}
