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
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/interflow"
	intra "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch/fcfs"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
)

func TestConfig_ValidateAndApplyDefaults(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name          string
		input         Config
		opts          []configOption
		expectErr     bool
		expectedErrIs error
		assertion     func(*testing.T, Config, *Config)
	}{
		// --- Success Paths ---
		{
			name: "ShouldApplyDefaults_WhenFieldsAreUnspecified",
			input: Config{
				PriorityBands: []PriorityBandConfig{
					{Priority: 1, PriorityName: "High"},
					{Priority: 2, PriorityName: "Low", MaxBytes: 500},
				},
			},
			assertion: func(t *testing.T, _ Config, newCfg *Config) {
				assert.Equal(t, defaultInitialShardCount, newCfg.InitialShardCount, "InitialShardCount should be defaulted")
				assert.Equal(t, defaultFlowGCTimeout, newCfg.FlowGCTimeout, "FlowGCTimeout should be defaulted")
				assert.Equal(t, defaultEventChannelBufferSize, newCfg.EventChannelBufferSize,
					"EventChannelBufferSize should be defaulted")
				require.Len(t, newCfg.PriorityBands, 2, "Should have two priority bands")
				band1 := newCfg.PriorityBands[0]
				assert.Equal(t, defaultIntraFlowDispatchPolicy, band1.IntraFlowDispatchPolicy,
					"Band 1 IntraFlowDispatchPolicy should be defaulted")
				assert.Equal(t, defaultInterFlowDispatchPolicy, band1.InterFlowDispatchPolicy,
					"Band 1 InterFlowDispatchPolicy should be defaulted")
				assert.Equal(t, defaultQueue, band1.Queue, "Band 1 Queue should be defaulted")
				assert.Equal(t, defaultPriorityBandMaxBytes, band1.MaxBytes, "Band 1 MaxBytes should be defaulted")
				band2 := newCfg.PriorityBands[1]
				assert.Equal(t, uint64(500), band2.MaxBytes, "Band 2 MaxBytes should retain its specified value")
			},
		},
		{
			name: "ShouldRemainUnchanged_WhenConfigIsFullySpecifiedAndValid",
			input: Config{
				MaxBytes:               1000,
				InitialShardCount:      10,
				FlowGCTimeout:          10 * time.Minute,
				EventChannelBufferSize: 5000,
				PriorityBands: []PriorityBandConfig{{
					Priority:                1,
					PriorityName:            "Critical",
					IntraFlowDispatchPolicy: fcfs.FCFSPolicyName,
					InterFlowDispatchPolicy: interflow.BestHeadPolicyName,
					Queue:                   queue.ListQueueName,
					MaxBytes:                500,
				}},
			},
			assertion: func(t *testing.T, originalCfg Config, newCfg *Config) {
				assert.Equal(t, originalCfg.MaxBytes, newCfg.MaxBytes, "MaxBytes should remain unchanged")
				assert.Equal(t, originalCfg.InitialShardCount, newCfg.InitialShardCount,
					"InitialShardCount should remain unchanged")
				assert.Equal(t, originalCfg.FlowGCTimeout, newCfg.FlowGCTimeout, "FlowGCTimeout should remain unchanged")
				assert.Equal(t, originalCfg.EventChannelBufferSize, newCfg.EventChannelBufferSize,
					"EventChannelBufferSize should remain unchanged")
				assert.Equal(t, originalCfg.PriorityBands, newCfg.PriorityBands, "PriorityBands should remain unchanged")
			},
		},
		{
			name: "ShouldSucceed_WhenPolicyHasNoRequirements",
			input: Config{
				PriorityBands: []PriorityBandConfig{{
					Priority:                1,
					PriorityName:            "High",
					IntraFlowDispatchPolicy: intra.RegisteredPolicyName("policy-without-req"),
				}},
			},
			opts: []configOption{
				withIntraFlowDispatchPolicyFactory(
					func(_ intra.RegisteredPolicyName) (framework.IntraFlowDispatchPolicy, error) {
						return &mocks.MockIntraFlowDispatchPolicy{
							NameV:                      "policy-without-req",
							RequiredQueueCapabilitiesV: []framework.QueueCapability{},
						}, nil
					}),
			},
			expectErr: false,
		},
		{
			name: "ShouldSucceed_WhenPolicyAndQueueAreCompatible",
			input: Config{
				PriorityBands: []PriorityBandConfig{{
					Priority:                1,
					PriorityName:            "High",
					IntraFlowDispatchPolicy: intra.RegisteredPolicyName("policy-with-reqs"),
					Queue:                   queue.RegisteredQueueName("queue-with-reqs-capability"),
				}},
			},
			opts: []configOption{
				withIntraFlowDispatchPolicyFactory(
					func(_ intra.RegisteredPolicyName) (framework.IntraFlowDispatchPolicy, error) {
						return &mocks.MockIntraFlowDispatchPolicy{
							RequiredQueueCapabilitiesV: []framework.QueueCapability{"capability-A", "capability-B"},
						}, nil
					}),
				withQueueFactory(
					func(_ queue.RegisteredQueueName, _ framework.ItemComparator) (framework.SafeQueue, error) {
						return &mocks.MockSafeQueue{
							CapabilitiesV: []framework.QueueCapability{"capability-A", "capability-B", "capability-C"},
						}, nil
					}),
			},
			expectErr: false,
		},
		// --- Core Behavioral Contracts ---
		{
			name: "ShouldReturnDeepCopy_AndNotMutateInput",
			input: Config{
				PriorityBands: []PriorityBandConfig{{Priority: 1, PriorityName: "High"}},
			},
			assertion: func(t *testing.T, originalCfg Config, newCfg *Config) {
				newCfg.PriorityBands[0].PriorityName = "changed"
				assert.NotEqual(t, originalCfg.PriorityBands[0].PriorityName, newCfg.PriorityBands[0].PriorityName,
					"Original config should not be mutated by changes to the new config")
			},
		},
		// --- Input Validation Errors ---
		{
			name:      "ShouldError_WhenNoPriorityBandsAreDefined",
			input:     Config{PriorityBands: []PriorityBandConfig{}},
			expectErr: true,
		},
		{
			name:      "ShouldError_WhenPriorityNameIsMissing",
			input:     Config{PriorityBands: []PriorityBandConfig{{Priority: 1}}},
			expectErr: true,
		},
		{
			name: "ShouldError_WhenPriorityLevelsAreDuplicated",
			input: Config{
				PriorityBands: []PriorityBandConfig{
					{Priority: 1, PriorityName: "High"},
					{Priority: 1, PriorityName: "Also High"},
				},
			},
			expectErr: true,
		},
		{
			name: "ShouldError_WhenPriorityNamesAreDuplicated",
			input: Config{
				PriorityBands: []PriorityBandConfig{
					{Priority: 1, PriorityName: "High"},
					{Priority: 2, PriorityName: "High"},
				},
			},
			expectErr: true,
		},
		// --- Plugin and Compatibility Errors ---
		{
			name: "ShouldError_WhenPolicyAndQueueAreIncompatible",
			input: Config{
				PriorityBands: []PriorityBandConfig{{
					Priority:                1,
					PriorityName:            "High",
					IntraFlowDispatchPolicy: intra.RegisteredPolicyName("policy-with-req"),
					Queue:                   queue.ListQueueName,
				}},
			},
			opts: []configOption{withIntraFlowDispatchPolicyFactory(
				func(_ intra.RegisteredPolicyName) (framework.IntraFlowDispatchPolicy, error) {
					return &mocks.MockIntraFlowDispatchPolicy{
						RequiredQueueCapabilitiesV: []framework.QueueCapability{framework.QueueCapability("required-capability")},
					}, nil
				})},
			expectErr:     true,
			expectedErrIs: contracts.ErrPolicyQueueIncompatible,
		},
		{
			name: "ShouldError_WhenQueueFactoryFails",
			input: Config{
				PriorityBands: []PriorityBandConfig{{
					Priority:                1,
					PriorityName:            "High",
					Queue:                   queue.RegisteredQueueName("failing-queue"),
					IntraFlowDispatchPolicy: intra.RegisteredPolicyName("policy-with-req"),
				}},
			},
			expectErr: true,
			opts: []configOption{
				withIntraFlowDispatchPolicyFactory( // Forces queue instance creation for validating capabilities.
					func(name intra.RegisteredPolicyName) (framework.IntraFlowDispatchPolicy, error) {
						return &mocks.MockIntraFlowDispatchPolicy{
							NameV:                      string(name),
							RequiredQueueCapabilitiesV: []framework.QueueCapability{"required-capability"},
						}, nil
					},
				),
				withQueueFactory(func(_ queue.RegisteredQueueName, _ framework.ItemComparator) (framework.SafeQueue, error) {
					return nil, errors.New("queue creation failed")
				}),
			},
		},
		{
			name: "ShouldError_WhenPolicyFactoryFails",
			input: Config{
				PriorityBands: []PriorityBandConfig{{
					Priority:                1,
					PriorityName:            "High",
					IntraFlowDispatchPolicy: intra.RegisteredPolicyName("failing-policy"),
				}},
			},
			opts: []configOption{withIntraFlowDispatchPolicyFactory(
				func(_ intra.RegisteredPolicyName) (framework.IntraFlowDispatchPolicy, error) {
					return nil, errors.New("policy creation failed")
				})},
			expectErr: true,
		},
		// --- Defaulting of Invalid Values ---
		{
			name: "ShouldApplyDefault_WhenInitialShardCountIsInvalid",
			input: Config{
				InitialShardCount: -1,
				PriorityBands:     []PriorityBandConfig{{Priority: 1, PriorityName: "High"}},
			},
			assertion: func(t *testing.T, _ Config, newCfg *Config) {
				assert.Equal(t, defaultInitialShardCount, newCfg.InitialShardCount,
					"Invalid InitialShardCount should be corrected to the default")
			},
		},
		{
			name: "ShouldApplyDefault_WhenFlowGCTimeoutIsInvalid",
			input: Config{
				FlowGCTimeout: 0,
				PriorityBands: []PriorityBandConfig{{Priority: 1, PriorityName: "High"}},
			},
			assertion: func(t *testing.T, _ Config, newCfg *Config) {
				assert.Equal(t, defaultFlowGCTimeout, newCfg.FlowGCTimeout,
					"Invalid FlowGCTimeout should be corrected to the default")
			},
		},
		{
			name: "ShouldApplyDefault_WhenEventChannelBufferSizeIsInvalid",
			input: Config{
				EventChannelBufferSize: -1,
				PriorityBands:          []PriorityBandConfig{{Priority: 1, PriorityName: "High"}},
			},
			assertion: func(t *testing.T, _ Config, newCfg *Config) {
				assert.Equal(t, defaultEventChannelBufferSize, newCfg.EventChannelBufferSize,
					"Invalid EventChannelBufferSize should be corrected to the default")
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			originalInput := tc.input.deepCopy()
			validatedCfg, err := newConfig(tc.input, tc.opts...)

			if tc.expectErr {
				require.Error(t, err, "expected an error but got nil")
				if tc.expectedErrIs != nil {
					assert.ErrorIs(t, err, tc.expectedErrIs, "error should wrap the expected error type")
				}
				assert.Nil(t, validatedCfg, "validatedCfg should be nil on error")
			} else {
				require.NoError(t, err, "expected no error but got: %v", err)
				require.NotNil(t, validatedCfg, "validatedCfg should not be nil on success")
				if tc.assertion != nil {
					tc.assertion(t, *originalInput, validatedCfg)
				}

				// Ensure the original config is not mutated.
				assert.Equal(t, *originalInput, tc.input, "input config should not be mutated")
			}
		})
	}
}

func TestConfig_Partition(t *testing.T) {
	t.Parallel()
	baseCfg, err := newConfig(Config{
		MaxBytes: 103, // Will not distribute evenly
		PriorityBands: []PriorityBandConfig{
			{Priority: 1, PriorityName: "High", MaxBytes: 55}, // Will not distribute evenly
			{Priority: 2, PriorityName: "Low", MaxBytes: 0},   // Will be defaulted to 1,000,000,000
			{Priority: 3, PriorityName: "Mid", MaxBytes: 20},  // Will distribute evenly
		},
	})
	require.NoError(t, err, "Test setup: creating the base config should not fail")

	t.Run("ShouldDistributeWithRemainderCorrectly", func(t *testing.T) {
		t.Parallel()
		totalShards := 10
		// Global: 103 / 10 = 10 rem 3. First 3 shards get 11, rest get 10.
		// Band 1: 55 / 10 = 5 rem 5. First 5 shards get 6, rest get 5.
		// Band 2 (defaulted): 1,000,000,000 / 10 = 100,000,000 for all shards.
		// Band 3: 20 / 10 = 2 rem 0. All shards get 2.
		expectedGlobalBytes := []uint64{11, 11, 11, 10, 10, 10, 10, 10, 10, 10}
		expectedBand1Bytes := []uint64{6, 6, 6, 6, 6, 5, 5, 5, 5, 5}

		var totalGlobal, totalBand1, totalBand2, totalBand3 uint64
		for i := range totalShards {
			shardCfg := baseCfg.partition(i, totalShards)
			require.NotNil(t, shardCfg, "Partitioned config for shard %d should not be nil", i)
			assert.Equal(t, expectedGlobalBytes[i], shardCfg.MaxBytes, "Global MaxBytes for shard %d is incorrect", i)
			require.Len(t, shardCfg.PriorityBands, 3, "Partitioned config should have 3 bands")
			assert.Equal(t, expectedBand1Bytes[i], shardCfg.PriorityBands[0].MaxBytes,
				"Band 1 MaxBytes for shard %d is incorrect", i)

			expectedBand2Bytes := defaultPriorityBandMaxBytes / uint64(totalShards)
			assert.Equal(t, expectedBand2Bytes, shardCfg.PriorityBands[1].MaxBytes,
				"Band 2 MaxBytes for shard %d is incorrect", i)
			assert.Equal(t, uint64(2), shardCfg.PriorityBands[2].MaxBytes,
				"Band 3 MaxBytes for shard %d is incorrect", i)

			totalGlobal += shardCfg.MaxBytes
			totalBand1 += shardCfg.PriorityBands[0].MaxBytes
			totalBand2 += shardCfg.PriorityBands[1].MaxBytes
			totalBand3 += shardCfg.PriorityBands[2].MaxBytes
		}
		assert.Equal(t, baseCfg.MaxBytes, totalGlobal, "Sum of partitioned global MaxBytes should equal original")
		assert.Equal(t, baseCfg.PriorityBands[0].MaxBytes, totalBand1,
			"Sum of partitioned band 1 MaxBytes should equal original")
		assert.Equal(t, defaultPriorityBandMaxBytes, totalBand2,
			"Sum of partitioned band 2 MaxBytes should equal the default value")
		assert.Equal(t, baseCfg.PriorityBands[2].MaxBytes, totalBand3,
			"Sum of partitioned band 3 MaxBytes should equal original")
	})

	t.Run("ShouldHandleSingleShard", func(t *testing.T) {
		t.Parallel()
		shardCfg := baseCfg.partition(0, 1)
		require.NotNil(t, shardCfg, "Partitioned config for single shard should not be nil")
		assert.Equal(t, baseCfg.MaxBytes, shardCfg.MaxBytes, "Global MaxBytes should be unchanged for a single shard")
		require.Len(t, shardCfg.PriorityBands, 3, "Partitioned config should have 3 bands")
		assert.Equal(t, baseCfg.PriorityBands[0].MaxBytes, shardCfg.PriorityBands[0].MaxBytes,
			"Band 1 MaxBytes should be unchanged")
	})

	t.Run("ShouldPanic_WhenTotalShardsIsInvalid", func(t *testing.T) {
		t.Parallel()
		testCases := []struct {
			name        string
			totalShards int
		}{
			{"ZeroShards", 0},
			{"NegativeShards", -1},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				assert.PanicsWithValue(t,
					fmt.Sprintf("invariant violation: cannot partition into %d partitions", tc.totalShards),
					func() { baseCfg.partition(0, tc.totalShards) },
					"Should panic with a specific message when totalShards is not positive",
				)
			})
		}
	})
}

func TestConfig_GetBandConfig(t *testing.T) {
	t.Parallel()
	cfg, err := newConfig(Config{
		PriorityBands: []PriorityBandConfig{
			{Priority: 10, PriorityName: "High"},
		},
	})
	require.NoError(t, err, "Test setup: creating the config should not fail")

	t.Run("ShouldReturnCorrectConfig_ForExistingPriority", func(t *testing.T) {
		t.Parallel()
		bandCfg, err := cfg.getBandConfig(10)
		require.NoError(t, err, "Should not error for an existing priority")
		require.NotNil(t, bandCfg, "Returned band config should not be nil")
		assert.Equal(t, "High", bandCfg.PriorityName, "Should return the correct band config")
	})

	t.Run("ShouldReturnError_ForNonExistentPriority", func(t *testing.T) {
		t.Parallel()
		_, err := cfg.getBandConfig(99)
		assert.ErrorIs(t, err, contracts.ErrPriorityBandNotFound, "Error should wrap ErrPriorityBandNotFound")
	})
}

func TestConfig_DeepCopy(t *testing.T) {
	t.Parallel()

	baseCfg := Config{
		MaxBytes:               1000,
		InitialShardCount:      5,
		FlowGCTimeout:          10 * time.Minute,
		EventChannelBufferSize: 2048,
		PriorityBands: []PriorityBandConfig{
			{Priority: 1, PriorityName: "A"},
			{Priority: 2, PriorityName: "B"},
		},
	}
	// Create a fully initialized "original" config to be the source of the copy.
	original, err := newConfig(baseCfg)
	require.NoError(t, err, "Setup for deep copy should not fail")

	t.Run("ShouldReturnNil_ForNilReceiver", func(t *testing.T) {
		t.Parallel()
		var nilConfig *Config
		assert.Nil(t, nilConfig.deepCopy(), "Deep copy of a nil config should be nil")
	})

	t.Run("ShouldCreateIdenticalButSeparateObject", func(t *testing.T) {
		t.Parallel()
		clone := original.deepCopy()

		// Assert that the clone is not the same object in memory.
		require.NotSame(t, original, clone, "Clone should be a new object in memory")
		require.NotSame(t, &original.PriorityBands, &clone.PriorityBands, "PriorityBands slice should be a new object")
		require.NotSame(t, &original.priorityBandMap, &clone.priorityBandMap, "priorityBandMap should be a new object")
		assert.Equal(t, original.MaxBytes, clone.MaxBytes, "MaxBytes should be copied")
		assert.Equal(t, original.InitialShardCount, clone.InitialShardCount, "InitialShardCount should be copied")
		assert.Equal(t, original.FlowGCTimeout, clone.FlowGCTimeout, "FlowGCTimeout should be copied")
		assert.Equal(t, original.EventChannelBufferSize, clone.EventChannelBufferSize,
			"EventChannelBufferSize should be copied")
		assert.Equal(t, original.PriorityBands, clone.PriorityBands, "PriorityBands should be deeply copied")
	})

	t.Run("ShouldBeIndependent_AfterCopying", func(t *testing.T) {
		t.Parallel()
		clone := original.deepCopy()

		// Modify the clone
		clone.MaxBytes = 2000
		clone.PriorityBands[0].PriorityName = "modified"
		// This modification through the map should also modify the slice element, as the map holds pointers.
		clone.priorityBandMap[clone.PriorityBands[0].Priority].PriorityName = "modified-in-map"
		clone.queueFactory = func(_ queue.RegisteredQueueName, _ framework.ItemComparator) (framework.SafeQueue, error) {
			return nil, nil
		}

		// Assert that the original remains unchanged by comparing field by field for clarity.
		assert.Equal(t, uint64(1000), original.MaxBytes, "Original MaxBytes should not be changed")
		assert.Equal(t, 5, original.InitialShardCount, "Original InitialShardCount should not be changed")
		assert.Equal(t, "A", original.PriorityBands[0].PriorityName, "Original PriorityBands slice should not be changed")
		assert.Equal(t, "A", original.priorityBandMap[1].PriorityName, "Original priorityBandMap should not be changed")
		assert.NotNil(t, original.queueFactory, "Original queueFactory should not be nil")

		// Verify the change was contained to the clone. The final value should be the one set via the map.
		assert.Equal(t, "modified-in-map", clone.PriorityBands[0].PriorityName,
			"Clone's slice should reflect the change from the map")
		assert.Equal(t, "modified-in-map", clone.priorityBandMap[1].PriorityName, "Clone's map should reflect the change")
	})
}

func TestShardConfig_GetBandConfig(t *testing.T) {
	t.Parallel()
	baseCfg, err := newConfig(Config{
		PriorityBands: []PriorityBandConfig{
			{Priority: 10, PriorityName: "High"},
			{Priority: 20, PriorityName: "Low"},
		},
	})
	require.NoError(t, err, "Test setup: creating the base config should not fail")
	shardCfg := baseCfg.partition(0, 2)
	require.NotNil(t, shardCfg, "Test setup: partitioned config should not be nil")

	t.Run("ShouldReturnCorrectConfig_ForExistingPriority", func(t *testing.T) {
		t.Parallel()
		bandCfg, err := shardCfg.getBandConfig(20)
		require.NoError(t, err, "Should not error for an existing priority")
		require.NotNil(t, bandCfg, "Returned band config should not be nil")
		assert.Equal(t, "Low", bandCfg.PriorityName, "Should return the correct partitioned band config")
	})

	t.Run("ShouldReturnError_ForNonExistentPriority", func(t *testing.T) {
		t.Parallel()
		_, err := shardCfg.getBandConfig(99)
		assert.ErrorIs(t, err, contracts.ErrPriorityBandNotFound, "Error should wrap ErrPriorityBandNotFound")
	})
}
