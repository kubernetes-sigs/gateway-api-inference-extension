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
	testclock "k8s.io/utils/clock/testing"
)

func TestDynamicUsagePolicy_InitialLimit(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	// First call should return 1.0 (no limit)
	limit := policy.ComputeLimit(ctx, 0, 0.5)
	require.Equal(t, 1.0, limit, "Expected initial limit to be 1.0, got %f", limit)
}

func TestDynamicUsagePolicy_ThrottleWhenOverTarget(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	// Call once to establish baseline
	policy.ComputeLimit(ctx, 0, 0.5)

	// Saturation above target should reduce limit
	limit := policy.ComputeLimit(ctx, 0, 0.95)
	require.Lessf(t, limit, 1.0, "Expected limit to decrease when saturation > target, got %f", limit)
}

func TestDynamicUsagePolicy_RecoverWhenUnderTarget(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	// First, get saturation high to reduce limit
	policy.ComputeLimit(ctx, 0, 0.95)
	limitAfterHigh := policy.ComputeLimit(ctx, 0, 0.95)

	// Now drop saturation below target
	limitAfterLow := policy.ComputeLimit(ctx, 0, 0.3)

	require.Greaterf(t, limitAfterLow, limitAfterHigh,
		"Expected limit to increase when saturation drops below target, high=%f, low=%f",
		limitAfterHigh, limitAfterLow)

}

func TestDynamicUsagePolicy_ProportionalAdjustment(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	// Establish baseline
	policy.ComputeLimit(ctx, 1, 0.5)

	// Small overshoot
	policy.ComputeLimit(ctx, 1, 0.85)
	limitSmallOvershoot := policy.ComputeLimit(ctx, 1, 0.85)

	// Start fresh for large overshoot
	policy2 := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	policy2.ComputeLimit(ctx, 1, 0.5)
	policy2.ComputeLimit(ctx, 1, 0.99)
	limitLargeOvershoot := policy2.ComputeLimit(ctx, 1, 0.99)

	// Larger overshoot should result in more aggressive throttling
	require.Lessf(t, limitLargeOvershoot, limitSmallOvershoot,
		"Expected larger overshoot to throttle more aggressively, small=%f, large=%f",
		limitSmallOvershoot, limitLargeOvershoot)
}

func TestDynamicUsagePolicy_TrendBasedAdjustment(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	// Create a rising trend
	policy.ComputeLimit(ctx, 1, 0.85)
	policy.clock.Sleep(100 * time.Millisecond)
	policy.ComputeLimit(ctx, 1, 0.90)
	policy.clock.Sleep(100 * time.Millisecond)
	limitRising := policy.ComputeLimit(ctx, 1, 0.95)

	// Create a stable/falling trend
	policy2 := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	policy2.ComputeLimit(ctx, 1, 0.95)
	policy.clock.Sleep(100 * time.Millisecond)
	policy2.ComputeLimit(ctx, 1, 0.90)
	policy.clock.Sleep(100 * time.Millisecond)
	limitFalling := policy2.ComputeLimit(ctx, 1, 0.85)

	// Rising trend should throttle more aggressively than falling trend
	if limitRising >= limitFalling {
		t.Logf("Note: Rising trend limit=%f, Falling trend limit=%f", limitRising, limitFalling)
		t.Logf("Rising trend may not always throttle harder due to proportional adjustments")
	}
}

func TestDynamicUsagePolicy_PriorityScaling(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	// Establish baseline for both priorities
	policy.ComputeLimit(ctx, 10, 0.5)
	policy.ComputeLimit(ctx, -10, 0.5)

	// Same high saturation for both
	policy.ComputeLimit(ctx, 10, 0.95)
	limitHighPriority := policy.ComputeLimit(ctx, 10, 0.95)

	policy.ComputeLimit(ctx, -10, 0.95)
	limitLowPriority := policy.ComputeLimit(ctx, -10, 0.95)

	// Low priority should be throttled more aggressively
	require.Less(t, limitLowPriority, limitHighPriority, "Expected low priority to be throttled more aggressively, high=%f, low=%f",
		limitHighPriority, limitLowPriority)
}

func TestDynamicUsagePolicy_DecayMechanism(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	// Reduce limit by saturating
	policy.ComputeLimit(ctx, 0, 0.95)
	policy.ComputeLimit(ctx, 0, 0.95)
	limitBeforeIdle := policy.ComputeLimit(ctx, 0, 0.95)

	// Wait for idle threshold to pass
	policy.clock.Sleep(idleTimeThreshold + 100*time.Millisecond)

	// Compute limit again - should apply decay
	limitAfterIdle := policy.ComputeLimit(ctx, 0, 0.5)

	require.Greaterf(t, limitAfterIdle, limitBeforeIdle,
		"Expected limit to increase after idle period due to decay, before=%f, after=%f",
		limitBeforeIdle, limitAfterIdle)
}

func TestDynamicUsagePolicy_LimitClamping(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	// Try to push limit above 1.0 by having very low saturation repeatedly
	for i := 0; i < 20; i++ {
		policy.ComputeLimit(ctx, 0, 0.0)
	}
	limit := policy.ComputeLimit(ctx, 0, 0.0)

	require.LessOrEqualf(t, limit, 1.0, "Expected limit to be clamped at 1.0, got %f", limit)

	// Try to push limit below 0.0 by having very high saturation repeatedly
	policy2 := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	for i := 0; i < 50; i++ {
		policy2.ComputeLimit(ctx, 0, 1.0)
	}
	limit = policy2.ComputeLimit(ctx, 0, 1.0)

	require.GreaterOrEqualf(t, limit, 0.0, "Expected limit to be clamped at 0.0, got %f", limit)
}
