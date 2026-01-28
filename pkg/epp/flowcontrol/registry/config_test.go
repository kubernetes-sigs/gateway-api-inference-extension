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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/fairness"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/ordering"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

func newTestPluginsHandle(t *testing.T) plugin.Handle {
	t.Helper()
	handle := utils.NewTestHandle(t.Context())
	handle.AddPlugin(fairness.GlobalStrictFairnessPolicyType, &frameworkmocks.MockFairnessPolicy{
		TypedNameV: plugin.TypedName{
			Type: fairness.GlobalStrictFairnessPolicyType,
			Name: fairness.GlobalStrictFairnessPolicyType,
		},
	})
	handle.AddPlugin(fairness.RoundRobinFairnessPolicyType, &frameworkmocks.MockFairnessPolicy{
		TypedNameV: plugin.TypedName{
			Type: fairness.RoundRobinFairnessPolicyType,
			Name: fairness.RoundRobinFairnessPolicyType,
		},
	})
	handle.AddPlugin(ordering.FCFSOrderingPolicyType, &frameworkmocks.MockOrderingPolicy{
		TypedNameV: plugin.TypedName{
			Type: ordering.FCFSOrderingPolicyType,
			Name: ordering.FCFSOrderingPolicyType,
		},
	})
	handle.AddPlugin(ordering.EDFOrderingPolicyType, &frameworkmocks.MockOrderingPolicy{
		TypedNameV: plugin.TypedName{
			Type: ordering.EDFOrderingPolicyType,
			Name: ordering.EDFOrderingPolicyType,
		},
	})
	return handle
}

// mockCapabilityChecker is a test double for verifying that NewConfig correctly delegates compatibility checks.
type mockCapabilityChecker struct {
	checkCompatibilityFunc func(p flowcontrol.OrderingPolicy, q queue.RegisteredQueueName) error
}

func (m *mockCapabilityChecker) CheckCompatibility(p flowcontrol.OrderingPolicy, q queue.RegisteredQueueName) error {
	if m.checkCompatibilityFunc != nil {
		return m.checkCompatibilityFunc(p, q)
	}
	return nil
}

// mustBand is a helper to simplify test table setup.
// It panics if the band config creation fails, which should not happen with valid static inputs.
func mustBand(t *testing.T, priority int, name string, opts ...PriorityBandConfigOption) *PriorityBandConfig {
	handle := newTestPluginsHandle(t)
	pb, err := NewPriorityBandConfig(handle, priority, name, opts...)
	require.NoError(t, err, "failed to create test band")
	return pb
}

func TestNewConfig(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name          string
		opts          []ConfigOption
		handle        plugin.Handle
		expectErr     bool
		expectedErrIs error // Optional: check for specific wrapped error
		assertion     func(*testing.T, *Config)
	}{
		// --- Success Paths ---
		{
			name: "ShouldApplySystemDefaults_WhenNoOptionsProvided",
			opts: []ConfigOption{
				WithPriorityBand(mustBand(t, 1, "Default")),
			},
			handle: newTestPluginsHandle(t),
			assertion: func(t *testing.T, cfg *Config) {
				assert.Equal(t, defaultInitialShardCount, cfg.InitialShardCount, "InitialShardCount should be defaulted")
				assert.Equal(t, defaultFlowGCTimeout, cfg.FlowGCTimeout, "FlowGCTimeout should be defaulted")
				assert.Equal(t, defaultPriorityBandGCTimeout, cfg.PriorityBandGCTimeout, "PriorityBandGCTimeout should be defaulted")

				// Verify Band Defaults
				require.Contains(t, cfg.PriorityBands, 1)
				band := cfg.PriorityBands[1]
				assert.Equal(t, defaultOrderingPolicyRef, band.OrderingPolicy.TypedName().Name)
				require.NotNil(t, band.FairnessPolicy)
				assert.Equal(t, defaultFairnessPolicyRef, band.FairnessPolicy.TypedName().Name)
				assert.Equal(t, defaultQueue, band.Queue)
				assert.Equal(t, defaultPriorityBandMaxBytes, band.MaxBytes)
			},
		},
		{
			name: "ShouldRespectGlobalOverrides",
			opts: []ConfigOption{
				WithInitialShardCount(10),
				WithMaxBytes(5000),
				WithFlowGCTimeout(1 * time.Hour),
				WithPriorityBandGCTimeout(2 * time.Hour),
				WithPriorityBand(mustBand(t, 1, "High")),
			},
			handle: newTestPluginsHandle(t),
			assertion: func(t *testing.T, cfg *Config) {
				assert.Equal(t, 10, cfg.InitialShardCount)
				assert.Equal(t, uint64(5000), cfg.MaxBytes)
				assert.Equal(t, 1*time.Hour, cfg.FlowGCTimeout)
				assert.Equal(t, 2*time.Hour, cfg.PriorityBandGCTimeout)
			},
		},
		{
			name: "ShouldApplyBandDefaults_WithRawStructLiterals",
			opts: []ConfigOption{
				// Simulate a user passing a manually constructed struct (bypassing NewPriorityBandConfig).
				WithPriorityBand(&PriorityBandConfig{Priority: 1, PriorityName: "Raw"}),
			},
			handle: newTestPluginsHandle(t),
			assertion: func(t *testing.T, cfg *Config) {
				require.Contains(t, cfg.PriorityBands, 1)
				band := cfg.PriorityBands[1]
				assert.Equal(t, defaultQueue, band.Queue, "Queue should be defaulted even for raw struct inputs")
				assert.NotNil(t, band.FairnessPolicy)
				assert.Equal(t, defaultFairnessPolicyRef, band.FairnessPolicy.TypedName().Name)
				assert.Equal(t, defaultOrderingPolicyRef, band.OrderingPolicy.TypedName().Name)
			},
		},
		{
			name: "ShouldSucceed_WhenNoPriorityBandsDefined_WithDynamicDefaults",
			opts: []ConfigOption{
				// No WithPriorityBand options provided.
				// This relies entirely on dynamic provisioning.
			},
			handle: newTestPluginsHandle(t),
			assertion: func(t *testing.T, cfg *Config) {
				assert.Empty(t, cfg.PriorityBands, "PriorityBands map should be empty")
				require.NotNil(t, cfg.DefaultPriorityBand, "DefaultPriorityBand template must be initialized")
				assert.Equal(t, "Dynamic-Default", cfg.DefaultPriorityBand.PriorityName)
				assert.Equal(t, defaultQueue, cfg.DefaultPriorityBand.Queue)
				assert.NotNil(t, cfg.DefaultPriorityBand.FairnessPolicy)
				assert.Equal(t, defaultFairnessPolicyRef, cfg.DefaultPriorityBand.FairnessPolicy.TypedName().Name)
			},
		},
		{
			name: "ShouldRespectCustomDefaultPriorityBand",
			opts: []ConfigOption{
				WithDefaultPriorityBand(&PriorityBandConfig{
					PriorityName: "My-Custom-Template",
					Queue:        "CustomQueue",
				}),
				withCapabilityChecker(&mockCapabilityChecker{
					checkCompatibilityFunc: func(flowcontrol.OrderingPolicy, queue.RegisteredQueueName) error { return nil },
				}),
			},
			handle: newTestPluginsHandle(t),
			assertion: func(t *testing.T, cfg *Config) {
				require.NotNil(t, cfg.DefaultPriorityBand)
				assert.Equal(t, "My-Custom-Template", cfg.DefaultPriorityBand.PriorityName)
				assert.Equal(t, queue.RegisteredQueueName("CustomQueue"), cfg.DefaultPriorityBand.Queue)
				assert.NotNil(t, cfg.DefaultPriorityBand.FairnessPolicy)
				assert.Equal(t, defaultFairnessPolicyRef, cfg.DefaultPriorityBand.FairnessPolicy.TypedName().Name)
				assert.Equal(t, defaultOrderingPolicyRef, cfg.DefaultPriorityBand.OrderingPolicy.TypedName().Name)
			},
		},

		// --- Validation Errors (Global) ---
		{
			name:      "ShouldError_WhenInitialShardCountIsInvalid",
			opts:      []ConfigOption{WithInitialShardCount(0)}, // Option itself should return error.
			handle:    newTestPluginsHandle(t),
			expectErr: true,
		},
		{
			name:      "ShouldError_WhenFlowGCTimeoutIsInvalid",
			opts:      []ConfigOption{WithFlowGCTimeout(-1 * time.Second)},
			handle:    newTestPluginsHandle(t),
			expectErr: true,
		},
		{
			name:      "ShouldError_WhenPriorityBandGCTimeoutIsNegative",
			opts:      []ConfigOption{WithPriorityBandGCTimeout(-1 * time.Second)},
			handle:    newTestPluginsHandle(t),
			expectErr: true,
		},
		{
			name: "ShouldError_WhenPriorityBandGCTimeoutLessThanFlowGCTimeout",
			opts: []ConfigOption{
				WithFlowGCTimeout(10 * time.Minute),
				WithPriorityBandGCTimeout(5 * time.Minute), // Less than flow timeout
			},
			handle:    newTestPluginsHandle(t),
			expectErr: true,
		},
		{
			name: "ShouldSucceed_WhenPriorityBandGCTimeoutEqualToFlowGCTimeout",
			opts: []ConfigOption{
				WithFlowGCTimeout(10 * time.Minute),
				WithPriorityBandGCTimeout(10 * time.Minute), // Equal is OK
			},
			handle: newTestPluginsHandle(t),
			assertion: func(t *testing.T, cfg *Config) {
				assert.Equal(t, 10*time.Minute, cfg.PriorityBandGCTimeout)
			},
		},
		{
			name: "ShouldSucceed_WhenPriorityBandGCTimeoutGreaterThanFlowGCTimeout",
			opts: []ConfigOption{
				WithFlowGCTimeout(5 * time.Minute),
				WithPriorityBandGCTimeout(15 * time.Minute),
			},
			handle: newTestPluginsHandle(t),
			assertion: func(t *testing.T, cfg *Config) {
				assert.Equal(t, 15*time.Minute, cfg.PriorityBandGCTimeout)
			},
		},

		// --- Validation Errors (Bands) ---
		{
			name: "ShouldError_WhenDuplicatePriorityLevelAdded",
			opts: []ConfigOption{
				WithPriorityBand(mustBand(t, 1, "A")),
				WithPriorityBand(mustBand(t, 1, "B")), // Same priority level
			},
			handle:    newTestPluginsHandle(t),
			expectErr: true,
		},
		{
			name: "ShouldError_WhenDuplicatePriorityNameAdded",
			opts: []ConfigOption{
				WithPriorityBand(mustBand(t, 1, "High")),
				WithPriorityBand(mustBand(t, 2, "High")), // Same name
			},
			handle:    newTestPluginsHandle(t),
			expectErr: true,
		},
		{
			name:      "ShouldError_WhenBandIsNil",
			opts:      []ConfigOption{WithPriorityBand(nil)},
			handle:    newTestPluginsHandle(t),
			expectErr: true,
		},
		{
			name: "ShouldSynthesizeName_WhenBandNameMissing",
			opts: []ConfigOption{
				// Use raw struct to bypass NewPriorityBandConfig validation which forces a name argument.
				WithPriorityBand(&PriorityBandConfig{Priority: 1}),
			},
			handle: newTestPluginsHandle(t),
			assertion: func(t *testing.T, cfg *Config) {
				require.Contains(t, cfg.PriorityBands, 1, "Priority bands should contain the configured band")
				assert.Equal(t, "priority-1", cfg.PriorityBands[1].PriorityName,
					"PriorityName should be synthesized from priority level")
			},
		},

		// --- Hydration Failures ---
		{
			name:      "ShouldError_WhenDefaultPolicyMissingFromHandle",
			opts:      []ConfigOption{WithPriorityBand(&PriorityBandConfig{Priority: 1, PriorityName: "A"})},
			handle:    utils.NewTestHandle(t.Context()), // Handle has no plugin.
			expectErr: true,
		},

		// --- Compatibility Checks ---
		{
			name: "ShouldError_WhenCapabilityCheckerFails",
			opts: []ConfigOption{
				WithPriorityBand(mustBand(t, 1, "High")),
				withCapabilityChecker(&mockCapabilityChecker{
					checkCompatibilityFunc: func(flowcontrol.OrderingPolicy, queue.RegisteredQueueName) error {
						return contracts.ErrPolicyQueueIncompatible
					},
				}),
			},
			handle:        newTestPluginsHandle(t),
			expectErr:     true,
			expectedErrIs: contracts.ErrPolicyQueueIncompatible,
		},
		{
			name: "ShouldError_WhenDefaultRuntimeCheckerDetectsUnknownQueue",
			opts: []ConfigOption{
				WithPriorityBand(mustBand(t, 1, "BadBand", WithQueue("non-existent-queue"))),
			},
			handle:    newTestPluginsHandle(t),
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cfg, err := NewConfig(tc.handle, tc.opts...)

			if tc.expectErr {
				require.Error(t, err, "expected validation error")
				if tc.expectedErrIs != nil {
					assert.ErrorIs(t, err, tc.expectedErrIs)
				}
				assert.Nil(t, cfg, "config should be nil on error")
			} else {
				require.NoError(t, err, "unexpected configuration error")
				require.NotNil(t, cfg, "config should not be nil on success")
				if tc.assertion != nil {
					tc.assertion(t, cfg)
				}
			}
		})
	}
}

func TestNewPriorityBandConfig(t *testing.T) {
	t.Parallel()
	handle := newTestPluginsHandle(t)

	t.Run("ShouldApplyUserOverrides", func(t *testing.T) {
		t.Parallel()
		pb, err := NewPriorityBandConfig(handle, 1, "Custom",
			WithQueue(queue.RegisteredQueueName("CustomQueue")),
			WithBandMaxBytes(999),
			WithOrderingPolicy(ordering.EDFOrderingPolicyType, handle),
			WithFairnessPolicy(fairness.RoundRobinFairnessPolicyType, handle),
		)
		require.NoError(t, err)
		assert.Equal(t, queue.RegisteredQueueName("CustomQueue"), pb.Queue)
		assert.Equal(t, uint64(999), pb.MaxBytes)
		require.NotNil(t, pb.OrderingPolicy)
		assert.Equal(t, ordering.EDFOrderingPolicyType, pb.OrderingPolicy.TypedName().Name)
		require.NotNil(t, pb.FairnessPolicy)
		assert.Equal(t, fairness.RoundRobinFairnessPolicyType, pb.FairnessPolicy.TypedName().Name)
	})

	t.Run("ShouldError_OnInvalidOptions", func(t *testing.T) {
		t.Parallel()
		pb, err := NewPriorityBandConfig(handle, 1, "Bad", WithQueue(""))
		assert.Error(t, err, "Should error when setting empty queue")
		assert.Nil(t, pb)
	})

	t.Run("ShouldError_WhenPolicyRefUnknown", func(t *testing.T) {
		t.Parallel()
		pb, err := NewPriorityBandConfig(handle, 1, "Bad",
			WithFairnessPolicy("UnknownPolicy", handle),
		)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "no fairness policy registered for name")
		assert.Nil(t, pb)
	})
}

func TestConfig_Partition(t *testing.T) {
	t.Parallel()
	handle := newTestPluginsHandle(t)

	// Setup:
	// Global: 103 MaxBytes.
	// Band 1: 55 MaxBytes.
	// Band 2: 0 MaxBytes (will default to 1GB).
	// Band 3: 20 MaxBytes.
	cfg, err := NewConfig(
		handle,
		WithMaxBytes(103),
		WithPriorityBand(mustBand(t, 1, "Band1", WithBandMaxBytes(55))),
		WithPriorityBand(mustBand(t, 2, "Band2", WithBandMaxBytes(0))), // Explicit 0 implies default behavior via logic.
		WithPriorityBand(mustBand(t, 3, "Band3", WithBandMaxBytes(20))),
	)
	require.NoError(t, err)

	// NewConfig applies defaults. If we passed 0 to NewPriorityBandConfig, it became 1GB.
	// We need to check what the setup resulted in.
	expectedBand2Total := defaultPriorityBandMaxBytes
	assert.Equal(t, expectedBand2Total, cfg.PriorityBands[2].MaxBytes, "Band 2 should have been defaulted")

	t.Run("ShouldDistributeRemainderCorrectly", func(t *testing.T) {
		t.Parallel()
		totalShards := 10

		// Global: 103 / 10 = 10 rem 3. First 3 shards get 11.
		// Band 1: 55 / 10 = 5 rem 5. First 5 shards get 6.
		// Band 3: 20 / 10 = 2 rem 0. All get 2.

		var sumGlobal, sumBand1, sumBand2, sumBand3 uint64

		for i := 0; i < totalShards; i++ {
			shard := cfg.partition(i, totalShards)
			require.NotNil(t, shard)

			// Accumulate.
			sumGlobal += shard.MaxBytes
			sumBand1 += shard.PriorityBands[1].MaxBytes
			sumBand2 += shard.PriorityBands[2].MaxBytes
			sumBand3 += shard.PriorityBands[3].MaxBytes

			// Spot check specific shards.
			if i < 3 {
				assert.Equal(t, uint64(11), shard.MaxBytes, "Shard %d global bytes mismatch", i)
			} else {
				assert.Equal(t, uint64(10), shard.MaxBytes, "Shard %d global bytes mismatch", i)
			}
		}

		// Verify totals.
		assert.Equal(t, cfg.MaxBytes, sumGlobal, "Total global bytes preserved")
		assert.Equal(t, cfg.PriorityBands[1].MaxBytes, sumBand1, "Total Band 1 bytes preserved")
		assert.Equal(t, cfg.PriorityBands[2].MaxBytes, sumBand2, "Total Band 2 bytes preserved")
		assert.Equal(t, cfg.PriorityBands[3].MaxBytes, sumBand3, "Total Band 3 bytes preserved")
	})
}

func TestConfig_Clone(t *testing.T) {
	t.Parallel()
	handle := newTestPluginsHandle(t)

	original, err := NewConfig(
		handle,
		WithMaxBytes(1000),
		WithPriorityBand(mustBand(t, 1, "A")),
		WithPriorityBand(mustBand(t, 2, "B")),
	)
	require.NoError(t, err, "Setup failed")

	t.Run("ShouldReturnNil_ForNilReceiver", func(t *testing.T) {
		var nilConfig *Config
		assert.Nil(t, nilConfig.Clone())
	})

	t.Run("ShouldCreateDeepCopy", func(t *testing.T) {
		clone := original.Clone()

		require.NotSame(t, original, clone, "Struct pointers should differ")
		require.NotSame(t, original.PriorityBands[1], clone.PriorityBands[1],
			"Map values (pointers to bands) should differ")
		assert.Equal(t, original.MaxBytes, clone.MaxBytes)
		assert.Equal(t, original.PriorityBands[1].PriorityName, clone.PriorityBands[1].PriorityName)
	})

	t.Run("ShouldIsolateModifications", func(t *testing.T) {
		clone := original.Clone()

		// Modify the clone's map entry.
		clone.PriorityBands[1].MaxBytes = 99999
		clone.PriorityBands[1].PriorityName = "Modified"

		assert.Equal(t, defaultPriorityBandMaxBytes, original.PriorityBands[1].MaxBytes)
		assert.Equal(t, "A", original.PriorityBands[1].PriorityName)
		assert.Equal(t, uint64(99999), clone.PriorityBands[1].MaxBytes)
		assert.Equal(t, "Modified", clone.PriorityBands[1].PriorityName)
	})

	t.Run("ShouldDeepCopyDefaultPriorityBand", func(t *testing.T) {
		t.Parallel()
		original, err := NewConfig(newTestPluginsHandle(t))
		require.NoError(t, err)

		clone := original.Clone()

		require.NotSame(t, original.DefaultPriorityBand, clone.DefaultPriorityBand,
			"Clone should have a distinct pointer for DefaultPriorityBand")
		assert.Equal(t, original.DefaultPriorityBand.PriorityName, clone.DefaultPriorityBand.PriorityName)

		// Modify Clone.
		clone.DefaultPriorityBand.PriorityName = "Hacked"
		assert.Equal(t, "Dynamic-Default", original.DefaultPriorityBand.PriorityName,
			"Modifying clone template should not affect original")
	})
}

func TestNewConfigFromAPI(t *testing.T) {
	t.Parallel()
	handle := newTestPluginsHandle(t)

	testCases := []struct {
		name        string
		apiConfig   *configapi.FlowControlConfig
		assertion   func(*testing.T, *Config)
		expectedErr string
	}{
		// --- Happy Paths ---
		{
			name: "ShouldSucceed_WithFullConfiguration",
			apiConfig: &configapi.FlowControlConfig{
				MaxBytes: ptr.To(int64(100)),
				PriorityBands: []configapi.PriorityBandConfig{
					{
						Priority: 1,
						MaxBytes: ptr.To(int64(50)),
					},
				},
				DefaultPriorityBand: &configapi.PriorityBandConfig{
					MaxBytes: ptr.To(int64(10)),
				},
			},
			assertion: func(t *testing.T, cfg *Config) {
				assert.Equal(t, uint64(100), cfg.MaxBytes, "Global MaxBytes should be correctly translated")

				// Verify Explicit Band
				require.Contains(t, cfg.PriorityBands, 1, "Configured priority band should be present")
				assert.Equal(t, uint64(50), cfg.PriorityBands[1].MaxBytes, "Band MaxBytes should be correctly translated")
				assert.Equal(t, "priority-1", cfg.PriorityBands[1].PriorityName,
					"PriorityName should be defaulted if not provided")

				// Verify Default Template
				require.NotNil(t, cfg.DefaultPriorityBand, "DefaultPriorityBand should be configured")
				assert.Equal(t, uint64(10), cfg.DefaultPriorityBand.MaxBytes,
					"DefaultPriorityBand template MaxBytes should be translated")
			},
		},
		{
			name:      "ShouldSucceed_WithNilConfig_AndApplySystemDefaults",
			apiConfig: nil,
			assertion: func(t *testing.T, cfg *Config) {
				assert.Equal(t, uint64(0), cfg.MaxBytes, "Default global limit should be 0 (unlimited)")
				require.NotNil(t, cfg.DefaultPriorityBand,
					"Default priority band template should be initialized automatically")
				assert.Equal(t, defaultPriorityBandMaxBytes, cfg.DefaultPriorityBand.MaxBytes,
					"Default template should use system default capacity")
			},
		},
		{
			name: "ShouldSynthesizePriorityName_WhenMissing",
			apiConfig: &configapi.FlowControlConfig{
				PriorityBands: []configapi.PriorityBandConfig{
					{
						Priority: 1,
						// No PriorityName provided
					},
				},
			},
			assertion: func(t *testing.T, cfg *Config) {
				require.Contains(t, cfg.PriorityBands, 1, "Priority band should be present")
				assert.Equal(t, "priority-1", cfg.PriorityBands[1].PriorityName,
					"PriorityName should be synthesized from priority index")
			},
		},

		// --- Defaulting Logic (Nil vs Zero) ---
		{
			name: "ShouldApplyDefault_WhenBandMaxBytesIsNil",
			apiConfig: &configapi.FlowControlConfig{
				PriorityBands: []configapi.PriorityBandConfig{
					{
						Priority: 1,
						MaxBytes: nil, // Omitted
					},
				},
			},
			assertion: func(t *testing.T, cfg *Config) {
				require.Contains(t, cfg.PriorityBands, 1)
				assert.Equal(t, defaultPriorityBandMaxBytes, cfg.PriorityBands[1].MaxBytes,
					"Omitted MaxBytes (nil) should result in system default capacity (1GB)")
			},
		},
		{
			name: "ShouldApplyDefault_WhenBandMaxBytesIsZero",
			apiConfig: &configapi.FlowControlConfig{
				PriorityBands: []configapi.PriorityBandConfig{
					{
						Priority: 1,
						MaxBytes: ptr.To(int64(0)), // Explicitly zero
					},
				},
			},
			assertion: func(t *testing.T, cfg *Config) {
				require.Contains(t, cfg.PriorityBands, 1)
				assert.Equal(t, defaultPriorityBandMaxBytes, cfg.PriorityBands[1].MaxBytes,
					"Explicit MaxBytes (0) should be treated as 'Use Default' (1GB)")
			},
		},
		{
			name: "ShouldApplyDefault_WhenDefaultPriorityBandMaxBytesIsZero",
			apiConfig: &configapi.FlowControlConfig{
				DefaultPriorityBand: &configapi.PriorityBandConfig{
					MaxBytes: ptr.To(int64(0)), // Explicitly zero
				},
			},
			assertion: func(t *testing.T, cfg *Config) {
				require.NotNil(t, cfg.DefaultPriorityBand)
				assert.Equal(t, defaultPriorityBandMaxBytes, cfg.DefaultPriorityBand.MaxBytes,
					"Explicit 0 in DefaultPriorityBand template should be treated as 'Use Default'")
			},
		},

		// --- Validation Errors ---
		{
			name: "ShouldError_WithNegativeGlobalMaxBytes",
			apiConfig: &configapi.FlowControlConfig{
				MaxBytes: ptr.To(int64(-1)),
			},
			expectedErr: "MaxBytes must be non-negative",
		},
		{
			name: "ShouldError_WithNegativePriorityBandMaxBytes",
			apiConfig: &configapi.FlowControlConfig{
				PriorityBands: []configapi.PriorityBandConfig{
					{
						Priority: 1,
						MaxBytes: ptr.To(int64(-100)),
					},
				},
			},
			expectedErr: "priority band 1 MaxBytes must be non-negative",
		},
		{
			name: "ShouldError_WithNegativeDefaultPriorityBandMaxBytes",
			apiConfig: &configapi.FlowControlConfig{
				DefaultPriorityBand: &configapi.PriorityBandConfig{
					MaxBytes: ptr.To(int64(-5)),
				},
			},
			expectedErr: "DefaultPriorityBand MaxBytes must be non-negative",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			cfg, err := NewConfigFromAPI(tc.apiConfig, handle)

			if tc.expectedErr != "" {
				require.Error(t, err, "NewConfigFromAPI should return an error")
				assert.Contains(t, err.Error(), tc.expectedErr, "Error message should contain expected text")
				assert.Nil(t, cfg, "Config should be nil when error occurs")
			} else {
				require.NoError(t, err, "NewConfigFromAPI should not return an error for valid input")
				require.NotNil(t, cfg, "Config should not be nil on success")
				if tc.assertion != nil {
					tc.assertion(t, cfg)
				}
			}
		})
	}
}
