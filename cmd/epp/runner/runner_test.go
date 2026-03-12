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

package runner

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/usagelimits"
)

// TestLinearSpacingPolicy demonstrates a stateless UsageLimitPolicy that dynamically spaces
// ceilings based on the active priority domain. The highest-active priority always gets ceiling
// 1.0, and each subsequent tier drops by a fixed step (0.2). When priorities go idle and the
// domain shrinks, the policy recalculates so the new highest priority gets 1.0 again, preserving
// work-conservation. This is the example described in
// https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/2268#discussion_r2868128727
func TestLinearSpacingPolicy(t *testing.T) {
	t.Parallel()

	const step = 0.2
	linearSpacing := usagelimits.NewPolicyFunc("linear-spacing", func(_ context.Context, _ float64, priorities []int) []float64 {
		result := make([]float64, len(priorities))
		for i := range priorities {
			// priorities are ordered highest-first; ceiling drops by step per rank.
			result[i] = 1.0 - float64(i)*step
		}
		return result
	})

	ctx := context.Background()

	// Active domain [100, 0, -5] → ceilings [1.0, 0.8, 0.6]
	got := linearSpacing.ComputeLimit(ctx, 0.5, []int{100, 0, -5})
	assert.Equal(t, []float64{1.0, 0.8, 0.6}, got, "three active priorities should produce linearly spaced ceilings")

	// Priority 100 goes idle; active domain becomes [0, -5] → ceilings [1.0, 0.8]
	got = linearSpacing.ComputeLimit(ctx, 0.5, []int{0, -5})
	assert.Equal(t, []float64{1.0, 0.8}, got, "after highest priority goes idle, remaining priorities should be re-spaced")

	// Single active priority [0] → ceiling [1.0]
	got = linearSpacing.ComputeLimit(ctx, 0.5, []int{0})
	assert.Equal(t, []float64{1.0}, got, "single active priority should get full ceiling")
}
