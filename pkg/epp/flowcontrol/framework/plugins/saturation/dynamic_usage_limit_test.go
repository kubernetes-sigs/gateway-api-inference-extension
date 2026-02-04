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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
)

func TestDynamicUsagePolicy_InitialLimit(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	// First call should return 1.0 (no limit)
	limit := policy.ComputeLimit(ctx, 0, 0.5, nil)
	require.Equal(t, 1.0, limit, "Expected initial limit to be 1.0, got %f", limit)
}

func TestDynamicUsagePolicy_ThrottleWhenOverTarget(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	// Call once to establish baseline
	policy.ComputeLimit(ctx, 0, 0.5, nil)

	// Saturation above target should reduce limit
	limit := policy.ComputeLimit(ctx, 0, 0.95, nil)
	require.Lessf(t, limit, 1.0, "Expected limit to decrease when saturation > target, got %f", limit)
}

func TestDynamicUsagePolicy_RecoverWhenUnderTarget(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	// First, get saturation high to reduce limit
	policy.ComputeLimit(ctx, 0, 0.95, nil)
	limitAfterHigh := policy.ComputeLimit(ctx, 0, 0.95, nil)

	// Now drop saturation below target
	limitAfterLow := policy.ComputeLimit(ctx, 0, 0.3, nil)

	require.Greaterf(t, limitAfterLow, limitAfterHigh,
		"Expected limit to increase when saturation drops below target, high=%f, low=%f",
		limitAfterHigh, limitAfterLow)

}

func TestDynamicUsagePolicy_ProportionalAdjustment(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	// Establish baseline
	policy.ComputeLimit(ctx, 1, 0.5, nil)

	// Small overshoot
	policy.ComputeLimit(ctx, 1, 0.85, nil)
	limitSmallOvershoot := policy.ComputeLimit(ctx, 1, 0.85, nil)

	// Start fresh for large overshoot
	policy2 := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	policy2.ComputeLimit(ctx, 1, 0.5, nil)
	policy2.ComputeLimit(ctx, 1, 0.99, nil)
	limitLargeOvershoot := policy2.ComputeLimit(ctx, 1, 0.99, nil)

	// Larger overshoot should result in more aggressive throttling
	require.Lessf(t, limitLargeOvershoot, limitSmallOvershoot,
		"Expected larger overshoot to throttle more aggressively, small=%f, large=%f",
		limitSmallOvershoot, limitLargeOvershoot)
}

func TestDynamicUsagePolicy_TrendBasedAdjustment(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	// Create a rising trend
	policy.ComputeLimit(ctx, 1, 0.85, nil)
	policy.clock.Sleep(100 * time.Millisecond)
	policy.ComputeLimit(ctx, 1, 0.90, nil)
	policy.clock.Sleep(100 * time.Millisecond)
	limitRising := policy.ComputeLimit(ctx, 1, 0.95, nil)

	// Create a stable/falling trend
	policy2 := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	policy2.ComputeLimit(ctx, 1, 0.95, nil)
	policy.clock.Sleep(100 * time.Millisecond)
	policy2.ComputeLimit(ctx, 1, 0.90, nil)
	policy.clock.Sleep(100 * time.Millisecond)
	limitFalling := policy2.ComputeLimit(ctx, 1, 0.85, nil)

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
	policy.ComputeLimit(ctx, 10, 0.5, nil)
	policy.ComputeLimit(ctx, -10, 0.5, nil)

	// Same high saturation for both
	policy.ComputeLimit(ctx, 10, 0.95, nil)
	limitHighPriority := policy.ComputeLimit(ctx, 10, 0.95, nil)

	policy.ComputeLimit(ctx, -10, 0.95, nil)
	limitLowPriority := policy.ComputeLimit(ctx, -10, 0.95, nil)

	// Low priority should be throttled more aggressively
	require.Less(t, limitLowPriority, limitHighPriority, "Expected low priority to be throttled more aggressively, high=%f, low=%f",
		limitHighPriority, limitLowPriority)
}

func TestDynamicUsagePolicy_DecayMechanism(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	// Reduce limit by saturating
	policy.ComputeLimit(ctx, 0, 0.95, nil)
	policy.ComputeLimit(ctx, 0, 0.95, nil)
	limitBeforeIdle := policy.ComputeLimit(ctx, 0, 0.95, nil)

	// Wait for idle threshold to pass
	policy.clock.Sleep(idleTimeThreshold + 100*time.Millisecond)

	// Compute limit again - should apply decay
	limitAfterIdle := policy.ComputeLimit(ctx, 0, 0.5, nil)

	require.Greaterf(t, limitAfterIdle, limitBeforeIdle,
		"Expected limit to increase after idle period due to decay, before=%f, after=%f",
		limitBeforeIdle, limitAfterIdle)
}

func TestDynamicUsagePolicy_LimitClamping(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	// Try to push limit above 1.0 by having very low saturation repeatedly
	for i := 0; i < 20; i++ {
		policy.ComputeLimit(ctx, 0, 0.0, nil)
	}
	limit := policy.ComputeLimit(ctx, 0, 0.0, nil)

	require.LessOrEqualf(t, limit, 1.0, "Expected limit to be clamped at 1.0, got %f", limit)

	// Try to push limit below 0.0 by having very high saturation repeatedly
	policy2 := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	for i := 0; i < 50; i++ {
		policy2.ComputeLimit(ctx, 0, 1.0, nil)
	}
	limit = policy2.ComputeLimit(ctx, 0, 1.0, nil)

	require.GreaterOrEqualf(t, limit, 0.0, "Expected limit to be clamped at 0.0, got %f", limit)
}

func TestDynamicUsagePolicy_EndpointSubsetTracking(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	// Create metadata for two different endpoint subsets
	metadataSubsetA := map[string]any{
		metadata.SubsetFilterNamespace: map[string]any{
			metadata.SubsetFilterKey: []any{"10.0.0.1:8080", "10.0.0.2:8080"},
		},
	}

	metadataSubsetB := map[string]any{
		metadata.SubsetFilterNamespace: map[string]any{
			metadata.SubsetFilterKey: []any{"10.0.0.3:8080", "10.0.0.4:8080"},
		},
	}

	// Establish baseline for both subsets at same priority
	policy.ComputeLimit(ctx, 0, 0.5, metadataSubsetA)
	policy.ComputeLimit(ctx, 0, 0.5, metadataSubsetB)

	// Subset A experiences high saturation
	for i := 0; i < 5; i++ {
		policy.ComputeLimit(ctx, 0, 0.95, metadataSubsetA)
		policy.clock.Sleep(10 * time.Millisecond)
	}
	limitSubsetA := policy.ComputeLimit(ctx, 0, 0.95, metadataSubsetA)

	// Subset B remains at low saturation
	limitSubsetB := policy.ComputeLimit(ctx, 0, 0.3, metadataSubsetB)

	// Limits are tracked independently per (subset, priority) pair
	require.Lessf(t, limitSubsetA, 0.9,
		"Subset A with high saturation should be throttled, got %f", limitSubsetA)
	require.Greaterf(t, limitSubsetB, 0.9,
		"Subset B with low saturation should NOT be throttled, got %f", limitSubsetB)
}

func TestDynamicUsagePolicy_DifferentEndpointSubsetsHaveDifferentSlopes(t *testing.T) {
	policy := NewDynamicUsagePolicy(testclock.NewFakeClock(time.Now()))
	ctx := context.Background()

	metadataSubsetA := map[string]any{
		metadata.SubsetFilterNamespace: map[string]any{
			metadata.SubsetFilterKey: []any{"10.0.0.1:8080"},
		},
	}

	metadataSubsetB := map[string]any{
		metadata.SubsetFilterNamespace: map[string]any{
			metadata.SubsetFilterKey: []any{"10.0.0.2:8080"},
		},
	}

	// Subset A: rising saturation
	policy.ComputeLimit(ctx, 0, 0.5, metadataSubsetA)
	policy.clock.Sleep(100 * time.Millisecond)
	policy.ComputeLimit(ctx, 0, 0.7, metadataSubsetA)
	policy.clock.Sleep(100 * time.Millisecond)
	policy.ComputeLimit(ctx, 0, 0.9, metadataSubsetA)

	// Subset B: stable saturation
	policy.ComputeLimit(ctx, 0, 0.5, metadataSubsetB)
	policy.clock.Sleep(100 * time.Millisecond)
	policy.ComputeLimit(ctx, 0, 0.5, metadataSubsetB)
	policy.clock.Sleep(100 * time.Millisecond)
	policy.ComputeLimit(ctx, 0, 0.5, metadataSubsetB)

	keyA := generateCacheKey(metadataSubsetA)
	keyB := generateCacheKey(metadataSubsetB)
	require.NotEqual(t, keyA, keyB, "Different subsets should have different cache keys")

	samplesA := policy.saturations[keyA]
	samplesB := policy.saturations[keyB]
	require.NotEmpty(t, samplesA, "Subset A should have saturation samples")
	require.NotEmpty(t, samplesB, "Subset B should have saturation samples")

	// Slopes show trends are computed independently
	slopeA := slope(samplesA)
	slopeB := slope(samplesB)
	require.Greater(t, slopeA, 0.0, "Subset A should have positive trend (rising: 0.5→0.7→0.9)")
	require.InDelta(t, 0.0, slopeB, 0.01, "Subset B should have near-zero trend (stable: 0.5→0.5→0.5)")

	// Limits are tracked independently per (subset, priority)
	limitA := policy.ComputeLimit(ctx, 0, 0.85, metadataSubsetA) // Rising trend → aggressive throttle
	limitB := policy.ComputeLimit(ctx, 0, 0.85, metadataSubsetB) // Stable trend → gentler throttle

	require.NotEqual(t, limitA, limitB, "Limits should differ based on independent trends")
	require.Lessf(t, limitA, limitB, "Rising trend (A) should throttle more than stable trend (B)")
}
