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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/interflow/dispatch/besthead"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/interflow/dispatch/roundrobin"
	intra "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch/fcfs"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue/listqueue"
)

func TestConfig_ValidateAndApplyDefaults(t *testing.T) {
	t.Parallel()

	// Setup for failure injection tests
	failingPolicyName := intra.RegisteredPolicyName("failing-policy-for-config-test")
	intra.MustRegisterPolicy(failingPolicyName, func() (framework.IntraFlowDispatchPolicy, error) {
		return nil, errors.New("policy instantiation failed")
	})
	failingQueueName := queue.RegisteredQueueName("failing-queue-for-config-test")
	queue.MustRegisterQueue(failingQueueName, func(_ framework.ItemComparator) (framework.SafeQueue, error) {
		return nil, errors.New("queue instantiation failed")
	})

	// Setup a mock policy with a specific capability requirement to test the compatibility check.
	const mockCapability = framework.QueueCapability("TEST_CAPABILITY_FOR_CONFIG")
	policyWithReqName := intra.RegisteredPolicyName("policy-with-req-for-config-test")
	intra.MustRegisterPolicy(policyWithReqName, func() (framework.IntraFlowDispatchPolicy, error) {
		return &mocks.MockIntraFlowDispatchPolicy{
			NameV: string(policyWithReqName),
			RequiredQueueCapabilitiesV: []framework.QueueCapability{
				mockCapability,
			},
		}, nil
	})

	testCases := []struct {
		name          string
		input         *Config
		expectErr     bool
		expectedErrIs error
		expectedCfg   *Config
	}{
		{
			name: "Valid config with missing defaults",
			input: &Config{
				PriorityBands: []PriorityBandConfig{
					{Priority: 1, PriorityName: "High"},
					{Priority: 2, PriorityName: "Low", InterFlowDispatchPolicy: roundrobin.RoundRobinPolicyName},
				},
			},
			expectErr: false,
			expectedCfg: &Config{
				PriorityBands: []PriorityBandConfig{
					{
						Priority:                1,
						PriorityName:            "High",
						IntraFlowDispatchPolicy: fcfs.FCFSPolicyName,
						InterFlowDispatchPolicy: besthead.BestHeadPolicyName,
						Queue:                   listqueue.ListQueueName,
					},
					{
						Priority:                2,
						PriorityName:            "Low",
						IntraFlowDispatchPolicy: fcfs.FCFSPolicyName,
						InterFlowDispatchPolicy: roundrobin.RoundRobinPolicyName,
						Queue:                   listqueue.ListQueueName,
					},
				},
			},
		},
		{
			name: "Config with all fields specified and compatible",
			input: &Config{
				MaxBytes: 1000,
				PriorityBands: []PriorityBandConfig{
					{
						Priority:                1,
						PriorityName:            "High",
						IntraFlowDispatchPolicy: fcfs.FCFSPolicyName, // Compatible with ListQueue
						InterFlowDispatchPolicy: besthead.BestHeadPolicyName,
						Queue:                   listqueue.ListQueueName,
						MaxBytes:                500,
					},
				},
			},
			expectErr: false,
			expectedCfg: &Config{ // Should be unchanged
				MaxBytes: 1000,
				PriorityBands: []PriorityBandConfig{
					{
						Priority:                1,
						PriorityName:            "High",
						IntraFlowDispatchPolicy: fcfs.FCFSPolicyName,
						InterFlowDispatchPolicy: besthead.BestHeadPolicyName,
						Queue:                   listqueue.ListQueueName,
						MaxBytes:                500,
					},
				},
			},
		},
		{
			name:      "Error: No priority bands",
			input:     &Config{PriorityBands: []PriorityBandConfig{}},
			expectErr: true,
		},
		{
			name: "Error: Missing PriorityName",
			input: &Config{
				PriorityBands: []PriorityBandConfig{
					{Priority: 1},
				},
			},
			expectErr: true,
		},
		{
			name: "Error: Duplicate priority level",
			input: &Config{
				PriorityBands: []PriorityBandConfig{
					{Priority: 1, PriorityName: "High"},
					{Priority: 1, PriorityName: "Also High"},
				},
			},
			expectErr: true,
		},
		{
			name: "Error: Incompatible policy and queue",
			input: &Config{
				PriorityBands: []PriorityBandConfig{
					{
						Priority:                1,
						PriorityName:            "High",
						IntraFlowDispatchPolicy: policyWithReqName,       // Requires mock capability
						Queue:                   listqueue.ListQueueName, // Does not provide it
					},
				},
			},
			expectErr:     true,
			expectedErrIs: contracts.ErrPolicyQueueIncompatible,
		},
		{
			name: "Error: Failing policy instantiation",
			input: &Config{
				PriorityBands: []PriorityBandConfig{
					{
						Priority:                1,
						PriorityName:            "High",
						IntraFlowDispatchPolicy: failingPolicyName,
						Queue:                   listqueue.ListQueueName,
					},
				},
			},
			expectErr: true,
		},
		{
			name: "Error: Failing queue instantiation",
			input: &Config{
				PriorityBands: []PriorityBandConfig{
					{
						Priority:                1,
						PriorityName:            "High",
						IntraFlowDispatchPolicy: fcfs.FCFSPolicyName,
						Queue:                   failingQueueName,
					},
				},
			},
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			err := tc.input.validateAndApplyDefaults()
			if tc.expectErr {
				require.Error(t, err, "Expected an error for this test case")
				if tc.expectedErrIs != nil {
					assert.ErrorIs(t, err, tc.expectedErrIs, "Error should be of the expected type")
				}
			} else {
				require.NoError(t, err, "Did not expect an error for this test case")
				assert.Equal(t, tc.expectedCfg, tc.input, "Config after applying defaults does not match expected config")
			}
		})
	}
}

func TestConfig_Partition(t *testing.T) {
	t.Parallel()

	baseConfig := &Config{
		MaxBytes: 103,
		PriorityBands: []PriorityBandConfig{
			{Priority: 1, PriorityName: "High", MaxBytes: 55},
			{Priority: 2, PriorityName: "Low", MaxBytes: 0}, // Should remain 0
		},
	}

	t.Run("EvenDistributionWithRemainder", func(t *testing.T) {
		t.Parallel()
		totalShards := 10
		// Global: 103 / 10 = 10 remainder 3. First 3 shards get 11, rest get 10.
		// Band 1: 55 / 10 = 5 remainder 5. First 5 shards get 6, rest get 5.
		expectedGlobalBytes := []uint64{11, 11, 11, 10, 10, 10, 10, 10, 10, 10}
		expectedBand1Bytes := []uint64{6, 6, 6, 6, 6, 5, 5, 5, 5, 5}

		var totalGlobal, totalBand1 uint64
		for i := range totalShards {
			partitioned, err := baseConfig.partition(i, totalShards)
			require.NoError(t, err, "Partitioning should not fail for shard %d", i)
			assert.Equal(t, expectedGlobalBytes[i], partitioned.MaxBytes, "Global MaxBytes for shard %d is incorrect", i)
			require.Len(t, partitioned.PriorityBands, 2, "Partitioned config should have the same number of bands")
			assert.Equal(t, expectedBand1Bytes[i], partitioned.PriorityBands[0].MaxBytes,
				"Band 1 MaxBytes for shard %d is incorrect", i)
			assert.Zero(t, partitioned.PriorityBands[1].MaxBytes, "Band 2 MaxBytes should remain zero for shard %d", i)
			totalGlobal += partitioned.MaxBytes
			totalBand1 += partitioned.PriorityBands[0].MaxBytes
		}
		assert.Equal(t, baseConfig.MaxBytes, totalGlobal, "Sum of partitioned global MaxBytes should equal original")
		assert.Equal(t, baseConfig.PriorityBands[0].MaxBytes, totalBand1,
			"Sum of partitioned band 1 MaxBytes should equal original")
	})

	t.Run("SingleShard", func(t *testing.T) {
		t.Parallel()
		partitioned, err := baseConfig.partition(0, 1)
		require.NoError(t, err, "Partitioning for a single shard should not fail")
		assert.Equal(t, baseConfig.MaxBytes, partitioned.MaxBytes, "Global MaxBytes should be unchanged for a single shard")
		require.Len(t, partitioned.PriorityBands, 2, "Partitioned config should have the same number of bands")
		assert.Equal(t, baseConfig.PriorityBands[0].MaxBytes, partitioned.PriorityBands[0].MaxBytes,
			"Band 1 MaxBytes should be unchanged for a single shard")
	})

	t.Run("EmptyPriorityBands", func(t *testing.T) {
		t.Parallel()
		config := &Config{
			MaxBytes:      100,
			PriorityBands: []PriorityBandConfig{},
		}
		partitioned, err := config.partition(1, 3)
		require.NoError(t, err, "Partitioning should not fail for empty priority bands")
		assert.Equal(t, uint64(33), partitioned.MaxBytes, "Global MaxBytes should be partitioned correctly")
		assert.Empty(t, partitioned.PriorityBands, "PriorityBands slice should be empty")
		assert.NotNil(t, partitioned.PriorityBands, "PriorityBands slice should not be nil")
	})

	t.Run("ErrorHandling", func(t *testing.T) {
		t.Parallel()
		testCases := []struct {
			name        string
			shardIndex  int
			totalShards int
		}{
			{"NegativeShardIndex", -1, 5},
			{"ShardIndexOutOfBounds", 5, 5},
			{"ZeroTotalShards", 0, 0},
			{"NegativeTotalShards", 0, -1},
		}
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				_, err := baseConfig.partition(tc.shardIndex, tc.totalShards)
				assert.Error(t, err, "Expected an error for invalid partitioning arguments")
			})
		}
	})
}
