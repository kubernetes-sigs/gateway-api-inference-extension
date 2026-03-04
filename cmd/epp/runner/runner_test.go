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
		return usagelimits.NewPolicyFunc(name, func(_ context.Context, _ int, _ float64) float64 {
			return 0.8
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
				assert.Equal(t, 0.8, policy.ComputeLimit(ctx, tc.priority, tc.saturation))
			})
		}
	})

	t.Run("falls back to default policy when type is empty", func(t *testing.T) {
		t.Parallel()

		policy := r.setupFlowControlPlugins("", logr.Discard())
		require.NotNil(t, policy)

		ctx := context.Background()
		assert.Equal(t, 1.0, policy.ComputeLimit(ctx, 0, 0.5))
	})

	t.Run("falls back to default policy when type is unknown", func(t *testing.T) {
		t.Parallel()

		policy := r.setupFlowControlPlugins("no-such-policy-type", logr.Discard())
		require.NotNil(t, policy)

		ctx := context.Background()
		assert.Equal(t, 1.0, policy.ComputeLimit(ctx, 0, 0.5))
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

func (p *constantPointEightPolicy) ComputeLimit(_ context.Context, _ int, _ float64) float64 {
	return 0.8
}

// compile-time check that constantPointEightPolicy satisfies the interface.
var _ flowcontrol.UsageLimitPolicy = (*constantPointEightPolicy)(nil)

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
	assert.Equal(t, 0.8, policy.ComputeLimit(ctx, 0, 0.5))
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
			assert.Equal(t, 0.8, policy.ComputeLimit(ctx, tc.priority, tc.saturation))
		})
	}
}
