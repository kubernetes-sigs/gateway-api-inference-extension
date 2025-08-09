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
	"sync"
	"time"

	"k8s.io/utils/clock"
)

// gcTimer represents a single cancellable garbage collection timer and the generation it is associated with.
type gcTimer struct {
	// timer is the active timer instance. We use the abstraction to allow for deterministic testing.
	timer      clock.Timer
	generation uint64
}

// gcTracker is a concurrent-safe, decoupled manager for garbage collection timers.
//
// # Role: Decoupled Timer Management
//
// Its sole responsibility is to start, stop, and report timer expirations via a channel. It is explicitly designed to
// have no knowledge of the `FlowRegistry`'s internal state or the meaning of the flows it tracks. This decoupling keeps
// the timer logic simple, thread-safe, and reusable.
//
// # Concurrency and Race Handling
//
// The gcTracker uses a `sync.Mutex` to protect its internal map of timers.
//
// A critical aspect of its design is the handling of the inherent race condition in `time.Timer.Stop()`. A timer might
// fire just as it is being stopped or replaced (`Stop()` returns false if the timer already fired). The `gcTracker`
// addresses this by associating a generation ID with each timer. When a timer fires, it sends this generation ID
// along with the event. The consumer (`FlowRegistry`) is responsible for checking if the generation in the event
// matches the current generation of the flow, thereby ignoring stale timer events. A stale event can occur if a flow
// becomes idle (timer starts, gen=N), then becomes active again, then idle again (timer is replaced, gen=N+1). If the
// original timer for gen=N fires after this sequence, the registry will correctly ignore it by comparing generations.
type gcTracker struct {
	mu sync.Mutex
	// timers maps a flow ID to its active garbage collection timer.
	timers map[string]*gcTimer
	// eventCh is the channel over which timer expiration events are sent.
	eventCh chan<- event
	// clock provides time-related functions. It allows injecting a mock clock for testing.
	// We use `WithTickerAndDelayedExecution` as it guarantees the `AfterFunc` method.
	clock clock.WithTickerAndDelayedExecution
}

// newGCTracker creates a new garbage collection timer manager.
// Requires a clock implementation (provided by `FlowRegistry`).
func newGCTracker(eventCh chan<- event, clk clock.WithTickerAndDelayedExecution) *gcTracker {
	return &gcTracker{
		timers:  make(map[string]*gcTimer),
		eventCh: eventCh,
		clock:   clk,
	}
}

// start begins a new GC timer for a given flow and generation. If a timer already exists for the flow, it is implicitly
// stopped and replaced. This is the desired behavior, as a new call to start a timer for a flow (e.g., because it just
// became idle) should always supersede any previous timer.
func (gc *gcTracker) start(flowID string, generation uint64, timeout time.Duration) {
	gc.mu.Lock()
	defer gc.mu.Unlock()

	// If a timer already exists for this flow, stop it before starting a new one. This handles cases where a flow might
	// flap between active and idle states.
	if existing, ok := gc.timers[flowID]; ok {
		existing.timer.Stop()
	}

	// We use AfterFunc which works efficiently with both `RealClock` and `FakeClock` (for tests).
	timer := gc.clock.AfterFunc(timeout, func() {
		// When the timer fires, send the event to the registry for processing.
		// This happens asynchronously (triggered by time passage or `FakeClock.Step`).
		gc.eventCh <- &gcTimerFiredEvent{
			flowID:     flowID,
			generation: generation,
		}
	})

	gc.timers[flowID] = &gcTimer{
		timer:      timer,
		generation: generation,
	}
}

// stop halts and deletes the timer for a given flow. It is safe to call even if no timer exists for the flow.
func (gc *gcTracker) stop(flowID string) {
	gc.mu.Lock()
	defer gc.mu.Unlock()

	if existing, ok := gc.timers[flowID]; ok {
		// Attempt to stop the timer. If `timer.Stop()` returns false, it means the timer has already fired and its callback
		// has been sent to the event channel. The `FlowRegistry`'s event handler is responsible for correctly handling this
		// race condition by checking the flow's generation ID.
		existing.timer.Stop()
		delete(gc.timers, flowID)
	}
}
