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

package flowcontrol

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/fairness"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/ordering"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/usagelimits"
	flowcontrolif "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

func TestNewConfigFromAPI(t *testing.T) {
	t.Parallel()

	handle := utils.NewTestHandle(context.Background())
	handle.AddPlugin(fairness.GlobalStrictFairnessPolicyType, &mocks.MockFairnessPolicy{
		TypedNameV: fwkplugin.TypedName{
			Name: fairness.GlobalStrictFairnessPolicyType,
			Type: fairness.GlobalStrictFairnessPolicyType,
		},
	})
	handle.AddPlugin(ordering.FCFSOrderingPolicyType, &mocks.MockOrderingPolicy{
		TypedNameV: fwkplugin.TypedName{
			Name: ordering.FCFSOrderingPolicyType,
			Type: ordering.FCFSOrderingPolicyType,
		},
	})
	handle.AddPlugin(usagelimits.NoopUsageLimitPolicyType, usagelimits.NoopPolicy())

	// A func-based custom policy that always returns 0.8 — demonstrates that users can define
	// their policy in the standard plugins section and reference it via UsageLimit.PluginRef.
	const funcPolicyName = "func-policy"
	handle.AddPlugin(funcPolicyName, usagelimits.NewPolicyFunc(funcPolicyName, func(_ context.Context, _ float64, priorities []int) []float64 {
		result := make([]float64, len(priorities))
		for i := range result {
			result[i] = 0.8
		}
		return result
	}))

	// A hand-rolled struct implementing UsageLimitPolicy — demonstrates that any struct
	// satisfying the interface can be registered and resolved, without relying on usagelimits helpers.
	const structPolicyName = "struct-policy"
	handle.AddPlugin(structPolicyName, &constantPointEightPolicy{})

	testCases := []struct {
		name        string
		apiConfig   *configapi.FlowControlConfig
		assertion   func(*testing.T, *Config)
		expectedErr string
	}{
		{
			name:      "Success - Nil Config defaults",
			apiConfig: nil,
			assertion: func(t *testing.T, cfg *Config) {
				assert.NotNil(t, cfg.Registry, "Registry config sub-struct should be initialized even when API config is nil")
				assert.NotNil(t, cfg.Controller,
					"Controller config sub-struct should be initialized even when API config is nil")
				assert.NotZero(t, cfg.Registry.InitialShardCount,
					"Registry should contain default values (InitialShardCount) when API config is nil")
				assert.NotZero(t, cfg.Controller.EnqueueChannelBufferSize,
					"Controller should contain default values (EnqueueChannelBufferSize) when API config is nil")
			},
		},
		{
			name: "Success - Explicit Values",
			apiConfig: &configapi.FlowControlConfig{
				MaxBytes: ptr.To(int64(2048)),
			},
			assertion: func(t *testing.T, cfg *Config) {
				assert.Equal(t, uint64(2048), cfg.Registry.MaxBytes,
					"MaxBytes should be correctly translated from int64 in API to uint64 in internal config")
			},
		},
		{
			name:      "Success - Default UsageLimitPolicy when UsageLimit is nil",
			apiConfig: nil,
			assertion: func(t *testing.T, cfg *Config) {
				require.NotNil(t, cfg.UsageLimitPolicy, "UsageLimitPolicy should be resolved even when not explicitly configured")
				ceilings := cfg.UsageLimitPolicy.ComputeLimit(context.Background(), 0.5, []int{0})
				assert.Equal(t, []float64{1.0}, ceilings, "Default noop policy should return 1.0 (no gating)")
			},
		},
		{
			name: "Success - UsageLimitPolicyRef is resolved",
			apiConfig: &configapi.FlowControlConfig{
				UsageLimit: &configapi.UsageLimitConfig{
					PluginRef: usagelimits.NoopUsageLimitPolicyType,
				},
			},
			assertion: func(t *testing.T, cfg *Config) {
				require.NotNil(t, cfg.UsageLimitPolicy, "UsageLimitPolicy should be resolved from the handle")
				ceilings := cfg.UsageLimitPolicy.ComputeLimit(context.Background(), 0.5, []int{0})
				assert.Equal(t, []float64{1.0}, ceilings, "Noop policy should return 1.0 (no gating)")
			},
		},
		{
			name: "Success - Func-based UsageLimitPolicy resolved via PluginRef",
			apiConfig: &configapi.FlowControlConfig{
				UsageLimit: &configapi.UsageLimitConfig{
					PluginRef: funcPolicyName,
				},
			},
			assertion: func(t *testing.T, cfg *Config) {
				require.NotNil(t, cfg.UsageLimitPolicy)
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
					assert.Equal(t, []float64{0.8}, cfg.UsageLimitPolicy.ComputeLimit(ctx, tc.saturation, []int{tc.priority}),
						"func-based policy should return 0.8 at %s", tc.name)
				}
			},
		},
		{
			name: "Success - Struct-based UsageLimitPolicy resolved via PluginRef",
			apiConfig: &configapi.FlowControlConfig{
				UsageLimit: &configapi.UsageLimitConfig{
					PluginRef: structPolicyName,
				},
			},
			assertion: func(t *testing.T, cfg *Config) {
				require.NotNil(t, cfg.UsageLimitPolicy)
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
					assert.Equal(t, []float64{0.8}, cfg.UsageLimitPolicy.ComputeLimit(ctx, tc.saturation, []int{tc.priority}),
						"struct-based policy should return 0.8 at %s", tc.name)
				}
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cfg, err := NewConfigFromAPI(tc.apiConfig, handle)

			require.NoError(t, err, "NewConfigFromAPI should not return an error for valid configuration")
			require.NotNil(t, cfg, "NewConfigFromAPI should return a non-nil Config object on success")

			if tc.assertion != nil {
				tc.assertion(t, cfg)
			}
		})
	}
}

// constantPointEightPolicy is a hand-rolled UsageLimitPolicy implementation that always returns 0.8.
// It exists to show that any struct satisfying the interface can be registered and resolved,
// without relying on the usagelimits.NewPolicyFunc helper.
type constantPointEightPolicy struct{}

func (p *constantPointEightPolicy) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{
		Type: "constant-point-eight-policy-type",
		Name: "constant-point-eight-policy",
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
var _ flowcontrolif.UsageLimitPolicy = (*constantPointEightPolicy)(nil)
