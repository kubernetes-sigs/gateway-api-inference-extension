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
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	clocktesting "k8s.io/utils/clock/testing"
)

// gcTestHarness encapsulates all the components needed for a gcTracker test.
type gcTestHarness struct {
	t         *testing.T
	fakeClock *clocktesting.FakeClock
	eventCh   chan event
	gc        *gcTracker
}

// newGCTrackerTestHarness creates a new test harness with a deterministic fake clock.
// The channel can be buffered or unbuffered depending on the test requirements.
func newGCTrackerTestHarness(t *testing.T, eventChanBuffer int) *gcTestHarness {
	t.Helper()
	eventCh := make(chan event, eventChanBuffer)
	fakeClock := clocktesting.NewFakeClock(time.Now())
	gc := newGCTracker(eventCh, fakeClock)

	return &gcTestHarness{
		t:         t,
		fakeClock: fakeClock,
		eventCh:   eventCh,
		gc:        gc,
	}
}

// assertEventReceived checks that a GC event was received on the channel with the expected properties.
func (h *gcTestHarness) assertEventReceived(expectedFlowID string, expectedGen uint64) {
	h.t.Helper()
	select {
	case evt := <-h.eventCh:
		timerEvt, ok := evt.(*gcTimerFiredEvent)
		require.True(h.t, ok, "Event should be of type gcTimerFiredEvent")
		assert.Equal(h.t, expectedFlowID, timerEvt.flowID, "Event should have the correct FlowID")
		assert.Equal(h.t, expectedGen, timerEvt.generation, "Event should have the correct Generation")
	case <-time.After(1 * time.Second): // Use a real timeout for the test itself to prevent hangs.
		h.t.Fatal("Timeout: Did not receive expected GC event")
	}
}

// assertNoEventReceived checks that no GC event was sent on the channel.
func (h *gcTestHarness) assertNoEventReceived(format string, args ...any) {
	h.t.Helper()
	select {
	case evt := <-h.eventCh:
		// Combine the static error with the custom message for a comprehensive failure report.
		customMsg := fmt.Sprintf(format, args...)
		h.t.Fatalf("Received unexpected GC event: %v. Assertion message: %s", evt, customMsg)
	default:
		// Success, no event received.
	}
}

func TestGCTracker(t *testing.T) {
	t.Parallel()

	const timeout = 10 * time.Second
	const flowID = "test-flow"

	t.Run("BasicLifecycle", func(t *testing.T) {
		t.Parallel()

		t.Run("WhenTimerFires_ShouldSendEvent", func(t *testing.T) {
			t.Parallel()
			h := newGCTrackerTestHarness(t, 1)

			h.gc.start(flowID, 1, timeout)
			h.assertNoEventReceived("No event should be sent before the timeout expires")

			// Advance the clock to fire the timer.
			h.fakeClock.Step(timeout)
			h.assertEventReceived(flowID, 1)
		})

		t.Run("WhenTimerIsStopped_ShouldNotFire", func(t *testing.T) {
			t.Parallel()
			h := newGCTrackerTestHarness(t, 1)

			h.gc.start(flowID, 2, timeout)
			h.gc.stop(flowID)

			// Advance the clock past the original timeout.
			h.fakeClock.Step(timeout)
			h.assertNoEventReceived("No event should be received for a stopped timer")
		})

		t.Run("WhenTimerIsReplaced_ShouldSupersedeOldTimer", func(t *testing.T) {
			t.Parallel()
			h := newGCTrackerTestHarness(t, 1)

			// Start the first timer (gen 3).
			h.gc.start(flowID, 3, timeout)
			// Start a new, longer timer immediately (gen 4). This should cancel the first one.
			h.gc.start(flowID, 4, timeout*3)

			// Advance the clock just enough to fire the first timer.
			h.fakeClock.Step(timeout)
			h.assertNoEventReceived("The superseded timer (gen 3) should not have fired")

			// Now advance the clock to fire the second, active timer.
			h.fakeClock.Step(timeout * 2)
			h.assertEventReceived(flowID, 4)
		})
	})

	t.Run("EdgeCasesAndSafety", func(t *testing.T) {
		t.Parallel()

		t.Run("WhenStoppingNonExistentTimer_ShouldNotPanic", func(t *testing.T) {
			t.Parallel()
			h := newGCTrackerTestHarness(t, 1)
			assert.NotPanics(t, func() {
				h.gc.stop("non-existent-flow")
			}, "stop() should not panic for a non-existent flow")
		})

		// This test verifies the behavior of the `gcTracker` when `stop()` is called for a timer that has already fired and
		// sent its event. The tracker's contract is that the consumer (`FlowRegistry`) will receive the event, and the
		// subsequent `stop()` call must gracefully clean up the internal state.
		t.Run("WhenStoppingTimer_ShouldCleanupMapAndPreserveEvent", func(t *testing.T) {
			t.Parallel()
			// Use a buffered channel so that `Step()` can complete without blocking.
			h := newGCTrackerTestHarness(t, 1)

			// Start the timer.
			h.gc.start(flowID, 5, timeout)

			// Advance the clock. This fires the timer and sends the event to the buffered channel.
			// The `Step()` call returns, and the event is now "in-flight".
			h.fakeClock.Step(timeout)

			// At this point, the timer has fired, but it might still exist in the `gcTracker`'s map until its `AfterFunc`
			// completes fully.
			// Now, call `stop()`. This simulates the case where `stop()` is called for a flow whose timer has already fired
			// and sent its event.
			h.gc.stop(flowID)

			// Assert that the timer has been removed from the internal map.
			// This is the primary responsibility of `stop()`.
			h.gc.mu.Lock()
			assert.NotContains(t, h.gc.timers, flowID, "stop() should clean up the internal map even for a fired timer")
			h.gc.mu.Unlock()

			// Assert that the event from the fired timer is still received.
			// This proves `stop()` doesn't interfere with an already-sent event.
			h.assertEventReceived(flowID, 5)
		})
	})
}
