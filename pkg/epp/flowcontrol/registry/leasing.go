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
	"runtime"
	"sync"
	"time"

	"k8s.io/utils/clock"
)

// leasedState implements a reference-counted "Leasing" pattern.
// Resources are active when leaseCount > 0 and idle when leaseCount == 0.
type leasedState struct {
	// mu protects the lifecycle fields.
	mu sync.Mutex

	// leaseCount tracks the number of active references to this resource.
	leaseCount int

	// becameIdleAt tracks when the resource became idle.
	// A zero value indicates the resource is currently active.
	becameIdleAt time.Time

	// markedForDeletion indicates the GC has selected this resource for deletion.
	markedForDeletion bool
}

// tryPin acquires a lease if the resource is not marked for deletion.
func (ls *leasedState) tryPin() bool {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	if ls.markedForDeletion {
		return false
	}
	ls.leaseCount++
	ls.becameIdleAt = time.Time{} // Mark as active.
	return true
}

// unpin releases a lease, marking the resource as idle if the count drops to zero.
func (ls *leasedState) unpin(now time.Time) {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	ls.leaseCount--
	if ls.leaseCount == 0 {
		ls.becameIdleAt = now
	}
}

// leasable is an interface for resources that embed leasedState.
type leasable interface {
	tryPin() bool
	unpin(now time.Time)
	isActive() bool
	checkAndMarkForDeletion(timeout time.Duration, now time.Time) bool
}

// isActive checks if the resource has any active leases.
func (ls *leasedState) isActive() bool {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	return ls.leaseCount > 0
}

// checkAndMarkForDeletion returns true if the resource is idle and has exceeded the timeout.
func (ls *leasedState) checkAndMarkForDeletion(timeout time.Duration, now time.Time) bool {
	ls.mu.Lock()
	defer ls.mu.Unlock()

	active := ls.leaseCount > 0
	if active {
		if !ls.becameIdleAt.IsZero() {
			ls.becameIdleAt = time.Time{}
		}
		return false
	}

	if ls.becameIdleAt.IsZero() {
		ls.becameIdleAt = now
		return false
	}

	if now.Sub(ls.becameIdleAt) < timeout {
		return false
	}

	ls.markedForDeletion = true
	return true
}

// collectLeasedResources removes idle resources from the map and returns them for cleanup.
func collectLeasedResources[K any, V interface {
	leasable
	comparable
}](
	m *sync.Map,
	timeout time.Duration,
	clock clock.Clock,
) []V {
	var resourcesToClean []V
	now := clock.Now()

	m.Range(func(key, value any) bool {
		k := key.(K)
		v := value.(V)

		if v.checkAndMarkForDeletion(timeout, now) {
			if val, loaded := m.LoadAndDelete(k); loaded {
				deletedVal := val.(V)
				resourcesToClean = append(resourcesToClean, deletedVal)
			}
		}
		return true
	})

	return resourcesToClean
}

// pinLeasedResource is a generic helper to perform the CAS loop for pinning a leased resource.
func pinLeasedResource[K any, V interface {
	leasable
	comparable
}](
	m *sync.Map,
	key K,
	createFn func() V,
	clock clock.Clock,
) (V, bool) {
	for {
		val, ok := m.Load(key)
		if !ok {
			val, ok = m.LoadOrStore(key, createFn())
		}
		state := val.(V)
		isNew := !ok

		if state.tryPin() {
			// Did the GC delete this object while we were acquiring it?
			currentVal, ok := m.Load(key)
			if !ok || currentVal != state {
				// We acquired a "stale" object. Back off and retry.
				state.unpin(clock.Now())
				continue
			}
			return state, isNew
		}

		// The GC has marked this resource for deletion. Back off and allow it to die.
		// We yield the processor to allow the GC goroutine to run and complete LoadAndDelete.
		runtime.Gosched()
		continue
	}
}
