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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	intra "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch/fcfs"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
)

// mockCapabilityChecker is a test double for verifying that NewConfig correctly delegates compatibility checks.
type mockCapabilityChecker struct {
	checkCompatibilityFunc func(intra intra.RegisteredPolicyName, q queue.RegisteredQueueName) error
}

func (m *mockCapabilityChecker) CheckCompatibility(p intra.RegisteredPolicyName, q queue.RegisteredQueueName) error {
	if m.checkCompatibilityFunc != nil {
		return m.checkCompatibilityFunc(p, q)
	}
	return nil
}

// mustBand is a helper to simplify test table setup.
// It panics if the band config creation fails, which should not happen with valid static inputs.
func mustBand(t *testing.T, priority int, name string, opts ...PriorityBandConfigOption) *PriorityBandConfig {
	pb, err := NewPriorityBandConfig(priority, name, opts...)
	require.NoError(t, err, "failed to create test band")
	return pb
}

func TestNewConfig(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name          string
		opts          []ConfigOption
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
			assertion: func(t *testing.T, cfg *Config) {
				assert.Equal(t, defaultInitialShardCount, cfg.InitialShardCount, "InitialShardCount should be defaulted")
				assert.Equal(t, defaultFlowGCTimeout, cfg.FlowGCTimeout, "FlowGCTimeout should be defaulted")
				assert.Equal(t, defaultEventChannelBufferSize, cfg.EventChannelBufferSize,
					"EventChannelBufferSize should be defaulted")

				// Verify Band Defaults
				require.Contains(t, cfg.PriorityBands, 1)
				band := cfg.PriorityBands[1]
				assert.Equal(t, defaultIntraFlowDispatchPolicy, band.IntraFlowDispatchPolicy)
				assert.Equal(t, defaultInterFlowDispatchPolicy, band.InterFlowDispatchPolicy)
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
				WithPriorityBand(mustBand(t, 1, "High")),
			},
			assertion: func(t *testing.T, cfg *Config) {
				assert.Equal(t, 10, cfg.InitialShardCount)
				assert.Equal(t, uint64(5000), cfg.MaxBytes)
				assert.Equal(t, 1*time.Hour, cfg.FlowGCTimeout)
			},
		},
		{
			name: "ShouldApplyBandDefaults_WithRawStructLiterals",
			opts: []ConfigOption{
				// Simulate a user passing a manually constructed struct (bypassing NewPriorityBandConfig).
				WithPriorityBand(&PriorityBandConfig{Priority: 1, PriorityName: "Raw"}),
			},
			assertion: func(t *testing.T, cfg *Config) {
				require.Contains(t, cfg.PriorityBands, 1)
				band := cfg.PriorityBands[1]
				assert.Equal(t, defaultQueue, band.Queue, "Queue should be defaulted even for raw struct inputs")
				assert.Equal(t, defaultIntraFlowDispatchPolicy, band.IntraFlowDispatchPolicy)
			},
		},

		// --- Validation Errors (Global) ---
		{
			name:          "ShouldError_WhenNoPriorityBandsDefined",
			opts:          []ConfigOption{WithInitialShardCount(1)},
			expectErr:     true,
			expectedErrIs: nil, // Generic error expected.
		},
		{
			name:      "ShouldError_WhenInitialShardCountIsInvalid",
			opts:      []ConfigOption{WithInitialShardCount(0)}, // Option itself should return error.
			expectErr: true,
		},
		{
			name:      "ShouldError_WhenFlowGCTimeoutIsInvalid",
			opts:      []ConfigOption{WithFlowGCTimeout(-1 * time.Second)},
			expectErr: true,
		},

		// --- Validation Errors (Bands) ---
		{
			name: "ShouldError_WhenDuplicatePriorityLevelAdded",
			opts: []ConfigOption{
				WithPriorityBand(mustBand(t, 1, "A")),
				WithPriorityBand(mustBand(t, 1, "B")), // Same priority level
			},
			expectErr: true,
		},
		{
			name: "ShouldError_WhenDuplicatePriorityNameAdded",
			opts: []ConfigOption{
				WithPriorityBand(mustBand(t, 1, "High")),
				WithPriorityBand(mustBand(t, 2, "High")), // Same name
			},
			expectErr: true,
		},
		{
			name: "ShouldError_WhenBandIsNil",
			opts: []ConfigOption{
				WithPriorityBand(nil),
			},
			expectErr: true,
		},
		{
			name: "ShouldError_WhenBandNameMissing",
			opts: []ConfigOption{
				// Use raw struct to bypass NewPriorityBandConfig validation.
				WithPriorityBand(&PriorityBandConfig{Priority: 1}),
			},
			expectErr: true,
		},

		// --- Compatibility Checks ---
		{
			name: "ShouldError_WhenCapabilityCheckerFails",
			opts: []ConfigOption{
				WithPriorityBand(mustBand(t, 1, "High")),
				withCapabilityChecker(&mockCapabilityChecker{
					checkCompatibilityFunc: func(p intra.RegisteredPolicyName, q queue.RegisteredQueueName) error {
						return contracts.ErrPolicyQueueIncompatible
					},
				}),
			},
			expectErr:     true,
			expectedErrIs: contracts.ErrPolicyQueueIncompatible,
		},
		{
			name: "ShouldError_WhenDefaultRuntimeCheckerDetectsUnknownPolicy",
			opts: []ConfigOption{
				WithPriorityBand(mustBand(t, 1, "BadBand", WithIntraFlowPolicy("non-existent-policy"))),
			},
			expectErr: true,
		},
		{
			name: "ShouldError_WhenDefaultRuntimeCheckerDetectsUnknownQueue",
			opts: []ConfigOption{
				WithPriorityBand(mustBand(t, 1, "BadBand",
					WithIntraFlowPolicy(fcfs.FCFSPolicyName),
					WithQueue("non-existent-queue"),
				)),
			},
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cfg, err := NewConfig(tc.opts...)

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

	t.Run("ShouldApplyUserOverrides", func(t *testing.T) {
		t.Parallel()
		pb, err := NewPriorityBandConfig(1, "Custom",
			WithQueue(queue.RegisteredQueueName("CustomQueue")),
			WithBandMaxBytes(999),
			WithIntraFlowPolicy("CustomPolicy"),
		)
		require.NoError(t, err)
		assert.Equal(t, queue.RegisteredQueueName("CustomQueue"), pb.Queue)
		assert.Equal(t, uint64(999), pb.MaxBytes)
		assert.Equal(t, intra.RegisteredPolicyName("CustomPolicy"), pb.IntraFlowDispatchPolicy)
		assert.Equal(t, defaultInterFlowDispatchPolicy, pb.InterFlowDispatchPolicy) // Unchanged default
	})

	t.Run("ShouldError_OnInvalidOptions", func(t *testing.T) {
		t.Parallel()
		pb, err := NewPriorityBandConfig(1, "Bad", WithQueue(""))
		assert.Error(t, err, "Should error when setting empty queue")
		assert.Nil(t, pb)
	})
}

func TestConfig_Partition(t *testing.T) {
	t.Parallel()

	// Setup:
	// Global: 103 MaxBytes.
	// Band 1: 55 MaxBytes.
	// Band 2: 0 MaxBytes (will default to 1GB).
	// Band 3: 20 MaxBytes.
	cfg, err := NewConfig(
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

	original, err := NewConfig(
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
}
