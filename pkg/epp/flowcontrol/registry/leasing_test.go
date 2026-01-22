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

package registry

import (
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	testclock "k8s.io/utils/clock/testing"
)

// testResource is a simple struct that embeds leasedState for testing purposes.
type testResource struct {
	id string
	leasedState
}

func TestLeasedState_Lifecycle(t *testing.T) {
	t.Parallel()

	clock := testclock.NewFakeClock(time.Now())
	r := &testResource{}

	// 1. Initial State
	assert.False(t, r.isActive(), "new resource should not be active")
	assert.True(t, r.becameIdleAt.IsZero(), "new resource should have zero becameIdleAt")

	// 2. Pin
	success := r.tryPin()
	require.True(t, success, "tryPin should succeed")
	assert.True(t, r.isActive(), "resource should be active after pin")
	assert.Equal(t, 1, r.leaseCount)

	// 3. Unpin
	r.unpin(clock.Now())
	assert.False(t, r.isActive(), "resource should not be active after unpin")
	assert.Equal(t, 0, r.leaseCount)
	assert.Equal(t, clock.Now(), r.becameIdleAt, "becameIdleAt should be set to now")

	// 4. Re-Pin (Resurrection)
	clock.Step(1 * time.Minute)
	success = r.tryPin()
	require.True(t, success, "re-pin should succeed")
	assert.True(t, r.becameIdleAt.IsZero(), "becameIdleAt should be cleared after re-pin")
}

func TestCollectLeasedResources(t *testing.T) {
	t.Parallel()

	clock := testclock.NewFakeClock(time.Now())
	timeout := 10 * time.Minute
	m := &sync.Map{}

	add := func(id string) *testResource {
		r := &testResource{id: id}
		m.Store(id, r)
		return r
	}

	// Scenario 1: Active Resource
	active := add("active")
	active.tryPin()

	// Scenario 2: Idle Resource (Just became idle)
	idleFresh := add("idle-fresh")

	// Scenario 3: Expired Resource
	expired := add("expired")
	expired.becameIdleAt = clock.Now().Add(-20 * time.Minute) // Way past timeout

	// Run GC (First Pass)
	deleted := collectLeasedResources[string, *testResource](m, timeout, clock)

	require.Len(t, deleted, 1, "should delete exactly one resource (expired)")
	assert.Equal(t, expired.id, deleted[0].id, "expired resource should be deleted")

	_, exists := m.Load(idleFresh.id)
	assert.True(t, exists, "idleFresh should still exist")
	assert.False(t, idleFresh.becameIdleAt.IsZero(), "idleFresh should be marked with becameIdleAt")
	assert.Equal(t, clock.Now(), idleFresh.becameIdleAt)

	_, exists = m.Load(active.id)
	assert.True(t, exists, "active should still exist")
}

func TestCollectLeasedResources_Integration(t *testing.T) {
	t.Parallel()
	clock := testclock.NewFakeClock(time.Now())
	timeout := 10 * time.Second
	m := &sync.Map{}

	r1 := &testResource{id: "r1"} // Will stay active
	r1.tryPin()
	m.Store("r1", r1)

	r2 := &testResource{id: "r2"} // Will be idle, then expire
	m.Store("r2", r2)

	r3 := &testResource{id: "r3"} // Will be idle, but not expire
	m.Store("r3", r3)

	// Pass 1: Mark r2 and r3 as idle.
	results := collectLeasedResources[string, *testResource](m, timeout, clock)
	assert.Empty(t, results)
	assert.False(t, r2.becameIdleAt.IsZero())
	assert.False(t, r3.becameIdleAt.IsZero())

	// Advance time by 5 seconds (less than timeout).
	clock.Step(5 * time.Second)
	results = collectLeasedResources[string, *testResource](m, timeout, clock)
	assert.Empty(t, results)

	// Advance time by another 6 seconds (total 11s > 10s timeout).
	clock.Step(6 * time.Second)

	// Pin r3 before the GC runs to save it.
	r3.tryPin()

	results = collectLeasedResources[string, *testResource](m, timeout, clock)

	// r2 should be deleted.
	require.Len(t, results, 1)
	assert.Equal(t, "r2", results[0].id)
	_, exists := m.Load("r2")
	assert.False(t, exists, "r2 should be removed from map")

	// r3 should be safe and becameIdleAt cleared.
	_, exists = m.Load("r3")
	assert.True(t, exists, "r3 should remain")
	assert.True(t, r3.becameIdleAt.IsZero(), "r3 should be unmarked")
}

func TestCollectLeasedResources_Concurrency(t *testing.T) {
	t.Parallel()

	// Verify that pinLeasedResource and collectLeasedResources are safe to use concurrently.
	// Specifically, we want to ensure:
	// 1. No data races.
	// 2. Objects are not double-deleted or deleted while active.
	// 3. Resurrected objects (pinned during GC) are NOT deleted.

	clock := testclock.NewFakeClock(time.Now())
	timeout := 10 * time.Millisecond
	m := &sync.Map{}

	const numGoroutines = 100
	const iterations = 1000

	var (
		resurrectedCount atomic.Int64
		deletedCount     atomic.Int64
	)

	// Create a single resource that we will fight over.
	key := "contended-resource"

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Start goroutines that randomly Pin/Unpin or Run GC.
	for i := range numGoroutines {
		go func(id int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(id)))

			for range iterations {
				action := rng.Intn(3) // 0: Pin/Unpin, 1: GC, 2: Access

				switch action {
				case 0:
					// Simulates a request coming in: Pin -> Sleep -> Unpin.
					val, _ := pinLeasedResource(
						m,
						key,
						func() *testResource { return &testResource{id: key} },
						clock,
					)

					// Invariant Check: pinLeasedResource must never return an object that is marked for deletion.
					val.mu.Lock()
					isMarked := val.markedForDeletion
					val.mu.Unlock()
					if isMarked {
						t.Error("Invariant Violation: pinLeasedResource returned an object marked for deletion")
					}

					// Simulate "Work"
					time.Sleep(time.Duration(rng.Intn(100)) * time.Microsecond)

					val.unpin(clock.Now())

				case 1:
					// Simulate Background GC.
					// Advance clock slightly to allow timeouts to eventually trigger.
					if rng.Intn(100) == 0 {
						clock.Step(5 * time.Millisecond) // Half timeout
					}

					deleted := collectLeasedResources[string, *testResource](m, timeout, clock)
					deletedCount.Add(int64(len(deleted)))
					for _, d := range deleted {
						if d.id != key {
							t.Errorf("GC deleted unexpected key: %s", d.id)
						}
					}

				case 2:
					// Simulate "Resurrection" check.
					// If it exists, try to pin it specifically.
					if val, ok := m.Load(key); ok {
						r := val.(*testResource)
						if r.tryPin() {
							resurrectedCount.Add(1)
							r.unpin(clock.Now())
						}
					}
				}
			}
		}(i)
	}

	wg.Wait()

	// Final Consistency Check:
	// If the resource remains in the map, it MUST be valid (not marked for deletion).
	// A marked-for-deletion object should have been removed by collectLeasedResources.
	if val, ok := m.Load(key); ok {
		r := val.(*testResource)
		r.mu.Lock()
		isMarked := r.markedForDeletion
		r.mu.Unlock()

		assert.False(t, isMarked, "Invariant Violation: Found object in map that is marked for deletion")
	}
}
