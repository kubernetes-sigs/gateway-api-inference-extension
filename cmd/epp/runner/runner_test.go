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
	"encoding/json"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/usagelimits"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// TestSetupFlowControlPlugins verifies that setupFlowControlPlugins resolves a registered
// UsageLimitPolicy from the plugin registry and that the resolved policy returns the expected limit.
func TestSetupFlowControlPlugins(t *testing.T) {
	// Register the factory before any parallel sub-test starts so the write
	// to the global registry map does not race with reads in sub-tests.
	const pluginType = "func-plugin"
	fwkplugin.Register(pluginType, func(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
		return usagelimits.NewPolicyFunc(name, func(_ context.Context, _ float64, priorities []int) []float64 {
			result := make([]float64, len(priorities))
			for i := range result {
				result[i] = 0.8
			}
			return result
		}), nil
	})
	t.Cleanup(func() {
		delete(fwkplugin.Registry, pluginType)
	})

	r := &Runner{}

	t.Run("resolves registered plugin and returns 0.8", func(t *testing.T) {
		t.Parallel()

		policy := r.setupFlowControlPlugins(pluginType, logr.Discard())
		require.NotNil(t, policy)

		ctx := context.Background()
		for _, tc := range []struct {
			name       string
			priority   int
			saturation float64
		}{
			{"zero saturation", 0, 0.0},
			{"half saturation", 1, 0.5},
			{"full saturation", 5, 1.0},
		} {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				assert.Equal(t, []float64{0.8}, policy.ComputeLimit(ctx, tc.saturation, []int{tc.priority}))
			})
		}
	})

	t.Run("falls back to default policy when type is empty", func(t *testing.T) {
		t.Parallel()

		policy := r.setupFlowControlPlugins("", logr.Discard())
		require.NotNil(t, policy)

		ctx := context.Background()
		assert.Equal(t, []float64{1.0}, policy.ComputeLimit(ctx, 0.5, []int{0}))
	})

	t.Run("falls back to default policy when type is unknown", func(t *testing.T) {
		t.Parallel()

		policy := r.setupFlowControlPlugins("no-such-policy-type", logr.Discard())
		require.NotNil(t, policy)

		ctx := context.Background()
		assert.Equal(t, []float64{1.0}, policy.ComputeLimit(ctx, 0.5, []int{0}))
	})
}

const constantPointEightStructPolicyType = "test-constant-point-eight-struct-usage-limit-policy"

// constantPointEightPolicy is a hand-rolled UsageLimitPolicy implementation that always returns 0.8.
// It exists to show that any struct satisfying the interface can be registered and resolved,
// without relying on the usagelimits.NewPolicyFunc helper.
type constantPointEightPolicy struct{}

func (p *constantPointEightPolicy) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{
		Type: constantPointEightStructPolicyType + "-type",
		Name: constantPointEightStructPolicyType,
	}
}

func (p *constantPointEightPolicy) ComputeLimit(_ context.Context, _ float64, priorities []int) []float64 {
	result := make([]float64, len(priorities))
	for i := range result {
		result[i] = 0.8
	}
	return result
}

// compile-time check that constantPointEightPolicy satisfies the interface.
var _ flowcontrol.UsageLimitPolicy = (*constantPointEightPolicy)(nil)

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

// TestSetupFlowControlPlugins_WithUsageLimitPolicyType verifies that UsageLimitPolicyType on
// flowcontrol.Config (the field a downstream project would set from a config file) is read by
// setupFlowControlPlugins to resolve and return the correct registered plugin.
func TestSetupFlowControlPlugins_WithUsageLimitPolicyType(t *testing.T) {
	const pluginType = "struct-plugin-via-config-option"

	fwkplugin.Register(pluginType, func(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
		return &constantPointEightPolicy{}, nil
	})
	t.Cleanup(func() {
		delete(fwkplugin.Registry, pluginType)
	})

	cfg := &fc.Config{UsageLimitPolicyType: pluginType}

	policy := (&Runner{}).setupFlowControlPlugins(cfg.UsageLimitPolicyType, logr.Discard())
	require.NotNil(t, policy)

	ctx := context.Background()
	assert.Equal(t, []float64{0.8}, policy.ComputeLimit(ctx, 0.5, []int{0}))
}

// TestSetupFlowControlPlugins_StructPlugin verifies that a hand-rolled struct implementing
// UsageLimitPolicy (without using usagelimits.NewPolicyFunc) can be registered and resolved.
func TestSetupFlowControlPlugins_StructPlugin(t *testing.T) {

	fwkplugin.Register(constantPointEightStructPolicyType, func(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
		return &constantPointEightPolicy{}, nil
	})
	t.Cleanup(func() {
		delete(fwkplugin.Registry, constantPointEightStructPolicyType)
	})

	r := &Runner{}

	policy := r.setupFlowControlPlugins(constantPointEightStructPolicyType, logr.Discard())
	require.NotNil(t, policy)

	ctx := context.Background()
	for _, tc := range []struct {
		name       string
		priority   int
		saturation float64
	}{
		{"zero saturation", 0, 0.0},
		{"half saturation", 1, 0.5},
		{"full saturation", 5, 1.0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, []float64{0.8}, policy.ComputeLimit(ctx, tc.saturation, []int{tc.priority}))
		})
	}
}
