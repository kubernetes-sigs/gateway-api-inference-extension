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
	"sort"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	inter "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/interflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/interflow/dispatch/besthead"
	intra "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch/fcfs"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue/listqueue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

// shardTestFixture holds the components needed for a `registryShard` test.
type shardTestFixture struct {
	config *Config
	shard  *registryShard
}

// setupTestShard creates a new test fixture for testing the `registryShard`.
func setupTestShard(t *testing.T) *shardTestFixture {
	t.Helper()

	config := &Config{
		PriorityBands: []PriorityBandConfig{
			{
				Priority:     10,
				PriorityName: "High",
			},
			{
				Priority:     20,
				PriorityName: "Low",
			},
		},
	}
	// Apply defaults to the master config first, as the parent registry would.
	err := config.validateAndApplyDefaults()
	require.NoError(t, err, "Setup: validating and defaulting config should not fail")

	// The parent registry would partition the config. For a single shard test, we can use the defaulted one directly.
	shard, err := newShard("test-shard-1", config, logr.Discard(), nil)
	require.NoError(t, err, "Setup: newShard should not return an error")
	require.NotNil(t, shard, "Setup: newShard should return a non-nil shard")

	return &shardTestFixture{
		config: config,
		shard:  shard,
	}
}

// _reconcileFlow_testOnly is a test helper that simulates the future admin logic for adding or updating a flow.
// It creates a `managedQueue` and correctly populates the `priorityBands` and `activeFlows` maps.
// This helper is intended to be replaced by the real `reconcileFlow` method in a future PR.
func (s *registryShard) _reconcileFlow_testOnly(
	t *testing.T,
	flowSpec types.FlowSpecification,
	isActive bool,
) *managedQueue {
	t.Helper()

	band, ok := s.priorityBands[flowSpec.Priority]
	require.True(t, ok, "Setup: priority band %d should exist", flowSpec.Priority)

	lq, err := queue.NewQueueFromName(listqueue.ListQueueName, nil)
	require.NoError(t, err, "Setup: creating a real listqueue should not fail")

	mq := newManagedQueue(
		lq,
		band.defaultIntraFlowDispatchPolicy,
		flowSpec,
		logr.Discard(),
		func(lenDelta, byteSizeDelta int64) { s.reconcileStats(flowSpec.Priority, lenDelta, byteSizeDelta) },
	)
	require.NotNil(t, mq, "Setup: newManagedQueue should not return nil")

	band.queues[flowSpec.ID] = mq
	if isActive {
		s.activeFlows[flowSpec.ID] = mq
	}

	return mq
}

func TestNewShard(t *testing.T) {
	t.Parallel()
	f := setupTestShard(t)

	assert.Equal(t, "test-shard-1", f.shard.ID(), "ID should be set correctly")
	assert.True(t, f.shard.IsActive(), "A new shard should be active")
	require.Len(t, f.shard.priorityBands, 2, "Should have 2 priority bands")

	// Check that priority levels are sorted correctly
	assert.Equal(t, []uint{10, 20}, f.shard.AllOrderedPriorityLevels(), "Priority levels should be ordered")

	// Check band 10
	band10, ok := f.shard.priorityBands[10]
	require.True(t, ok, "Priority band 10 should exist")
	assert.Equal(t, uint(10), band10.config.Priority, "Band 10 should have correct priority")
	assert.Equal(t, "High", band10.config.PriorityName, "Band 10 should have correct name")
	assert.NotNil(t, band10.interFlowDispatchPolicy, "Inter-flow policy for band 10 should be instantiated")
	assert.NotNil(t, band10.defaultIntraFlowDispatchPolicy,
		"Default intra-flow policy for band 10 should be instantiated")
	assert.Equal(t, besthead.BestHeadPolicyName, band10.interFlowDispatchPolicy.Name(),
		"Correct default inter-flow policy should be used")
	assert.Equal(t, fcfs.FCFSPolicyName, band10.defaultIntraFlowDispatchPolicy.Name(),
		"Correct default intra-flow policy should be used")

	// Check band 20
	band20, ok := f.shard.priorityBands[20]
	require.True(t, ok, "Priority band 20 should exist")
	assert.Equal(t, uint(20), band20.config.Priority, "Band 20 should have correct priority")
	assert.Equal(t, "Low", band20.config.PriorityName, "Band 20 should have correct name")
	assert.NotNil(t, band20.interFlowDispatchPolicy, "Inter-flow policy for band 20 should be instantiated")
	assert.NotNil(t, band20.defaultIntraFlowDispatchPolicy,
		"Default intra-flow policy for band 20 should be instantiated")
}

// TestNewShard_ErrorPaths modifies global plugin registries, so it cannot be run in parallel with other tests.
func TestNewShard_ErrorPaths(t *testing.T) {
	baseConfig := &Config{
		PriorityBands: []PriorityBandConfig{{
			Priority:                10,
			PriorityName:            "High",
			IntraFlowDispatchPolicy: fcfs.FCFSPolicyName,
			InterFlowDispatchPolicy: besthead.BestHeadPolicyName,
			Queue:                   listqueue.ListQueueName,
		}},
	}
	require.NoError(t, baseConfig.validateAndApplyDefaults(), "Setup: base config should be valid")

	t.Run("Invalid InterFlow Policy", func(t *testing.T) {
		// Register a mock policy that always fails to instantiate
		failingPolicyName := inter.RegisteredPolicyName("failing-inter-policy")
		inter.MustRegisterPolicy(failingPolicyName, func() (framework.InterFlowDispatchPolicy, error) {
			return nil, errors.New("inter-flow instantiation failed")
		})

		badConfig := *baseConfig
		badConfig.PriorityBands[0].InterFlowDispatchPolicy = failingPolicyName

		_, err := newShard("test", &badConfig, logr.Discard(), nil)
		require.Error(t, err, "newShard should fail with an invalid inter-flow policy")
	})

	t.Run("Invalid IntraFlow Policy", func(t *testing.T) {
		// Register a mock policy that always fails to instantiate
		failingPolicyName := intra.RegisteredPolicyName("failing-intra-policy")
		intra.MustRegisterPolicy(failingPolicyName, func() (framework.IntraFlowDispatchPolicy, error) {
			return nil, errors.New("intra-flow instantiation failed")
		})

		badConfig := *baseConfig
		badConfig.PriorityBands[0].IntraFlowDispatchPolicy = failingPolicyName

		_, err := newShard("test", &badConfig, logr.Discard(), nil)
		require.Error(t, err, "newShard should fail with an invalid intra-flow policy")
	})
}

func TestShard_Stats(t *testing.T) {
	t.Parallel()
	f := setupTestShard(t)

	// Add a queue and some items to test stats aggregation
	mq := f.shard._reconcileFlow_testOnly(t, types.FlowSpecification{ID: "flow1", Priority: 10}, true)

	// Add items
	require.NoError(t, mq.Add(mocks.NewMockQueueItemAccessor(100, "req1", "flow1")), "Adding item should not fail")
	require.NoError(t, mq.Add(mocks.NewMockQueueItemAccessor(50, "req2", "flow1")), "Adding item should not fail")

	stats := f.shard.Stats()

	// Check shard-level stats
	assert.Equal(t, uint64(2), stats.TotalLen, "Total length should be 2")
	assert.Equal(t, uint64(150), stats.TotalByteSize, "Total byte size should be 150")

	// Check per-band stats
	require.Len(t, stats.PerPriorityBandStats, 2, "Should have stats for 2 bands")
	band10Stats := stats.PerPriorityBandStats[10]
	assert.Equal(t, uint(10), band10Stats.Priority, "Band 10 stats should have correct priority")
	assert.Equal(t, uint64(2), band10Stats.Len, "Band 10 length should be 2")
	assert.Equal(t, uint64(150), band10Stats.ByteSize, "Band 10 byte size should be 150")

	band20Stats := stats.PerPriorityBandStats[20]
	assert.Equal(t, uint(20), band20Stats.Priority, "Band 20 stats should have correct priority")
	assert.Zero(t, band20Stats.Len, "Band 20 length should be 0")
	assert.Zero(t, band20Stats.ByteSize, "Band 20 byte size should be 0")
}

func TestShard_Accessors(t *testing.T) {
	t.Parallel()
	f := setupTestShard(t)

	flowID := "test-flow"
	activePriority := uint(10)
	drainingPriority := uint(20)

	// Setup state with one active and one draining queue for the same flow
	activeQueue := f.shard._reconcileFlow_testOnly(t, types.FlowSpecification{
		ID:       flowID,
		Priority: activePriority,
	}, true)
	drainingQueue := f.shard._reconcileFlow_testOnly(t, types.FlowSpecification{
		ID:       flowID,
		Priority: drainingPriority,
	}, false)

	t.Run("ActiveManagedQueue", func(t *testing.T) {
		t.Parallel()
		retrievedActiveQueue, err := f.shard.ActiveManagedQueue(flowID)
		require.NoError(t, err, "ActiveManagedQueue should not error for an existing flow")
		assert.Same(t, activeQueue, retrievedActiveQueue, "Should return the correct active queue")

		_, err = f.shard.ActiveManagedQueue("non-existent-flow")
		require.Error(t, err, "ActiveManagedQueue should error for a non-existent flow")
		assert.ErrorIs(t, err, contracts.ErrFlowInstanceNotFound, "Error should be ErrFlowInstanceNotFound")
	})

	t.Run("ManagedQueue", func(t *testing.T) {
		t.Parallel()
		retrievedDrainingQueue, err := f.shard.ManagedQueue(flowID, drainingPriority)
		require.NoError(t, err, "ManagedQueue should not error for a draining queue")
		assert.Same(t, drainingQueue, retrievedDrainingQueue, "Should return the correct draining queue")

		_, err = f.shard.ManagedQueue(flowID, 99) // Non-existent priority
		require.Error(t, err, "ManagedQueue should error for a non-existent priority")
		assert.ErrorIs(t, err, contracts.ErrPriorityBandNotFound, "Error should be ErrPriorityBandNotFound")

		_, err = f.shard.ManagedQueue("non-existent-flow", activePriority)
		require.Error(t, err, "ManagedQueue should error for a non-existent flow in an existing priority")
		assert.ErrorIs(t, err, contracts.ErrFlowInstanceNotFound, "Error should be ErrFlowInstanceNotFound")
	})

	t.Run("IntraFlowDispatchPolicy", func(t *testing.T) {
		t.Parallel()
		retrievedActivePolicy, err := f.shard.IntraFlowDispatchPolicy(flowID, activePriority)
		require.NoError(t, err, "IntraFlowDispatchPolicy should not error for an active instance")
		assert.Same(t, activeQueue.dispatchPolicy, retrievedActivePolicy,
			"Should return the policy from the active instance")

		_, err = f.shard.IntraFlowDispatchPolicy("non-existent-flow", activePriority)
		require.Error(t, err, "IntraFlowDispatchPolicy should error for a non-existent flow")
		assert.ErrorIs(t, err, contracts.ErrFlowInstanceNotFound, "Error should be ErrFlowInstanceNotFound")

		_, err = f.shard.IntraFlowDispatchPolicy(flowID, 99) // Non-existent priority
		require.Error(t, err, "IntraFlowDispatchPolicy should error for a non-existent priority")
		assert.ErrorIs(t, err, contracts.ErrPriorityBandNotFound, "Error should be ErrPriorityBandNotFound")
	})

	t.Run("InterFlowDispatchPolicy", func(t *testing.T) {
		t.Parallel()
		retrievedInterPolicy, err := f.shard.InterFlowDispatchPolicy(activePriority)
		require.NoError(t, err, "InterFlowDispatchPolicy should not error for an existing priority")
		assert.Same(t, f.shard.priorityBands[activePriority].interFlowDispatchPolicy, retrievedInterPolicy,
			"Should return the correct inter-flow policy")

		_, err = f.shard.InterFlowDispatchPolicy(99) // Non-existent priority
		require.Error(t, err, "InterFlowDispatchPolicy should error for a non-existent priority")
		assert.ErrorIs(t, err, contracts.ErrPriorityBandNotFound, "Error should be ErrPriorityBandNotFound")
	})
}

func TestShard_PriorityBandAccessor(t *testing.T) {
	t.Parallel()
	f := setupTestShard(t)

	// Setup shard state for the tests
	p1, p2 := uint(10), uint(20)
	f.shard._reconcileFlow_testOnly(t, types.FlowSpecification{ID: "flow1", Priority: p1}, true)
	f.shard._reconcileFlow_testOnly(t, types.FlowSpecification{ID: "flow1", Priority: p2}, false)
	f.shard._reconcileFlow_testOnly(t, types.FlowSpecification{ID: "flow2", Priority: p1}, true)

	t.Run("Accessor for existing priority", func(t *testing.T) {
		t.Parallel()
		accessor, err := f.shard.PriorityBandAccessor(p1)
		require.NoError(t, err, "PriorityBandAccessor should not fail for existing priority")
		require.NotNil(t, accessor, "Accessor should not be nil")

		t.Run("Properties", func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, p1, accessor.Priority(), "Accessor should have correct priority")
			assert.Equal(t, "High", accessor.PriorityName(), "Accessor should have correct priority name")
		})

		t.Run("FlowIDs", func(t *testing.T) {
			t.Parallel()
			flowIDs := accessor.FlowIDs()
			sort.Strings(flowIDs)
			assert.Equal(t, []string{"flow1", "flow2"}, flowIDs,
				"Accessor should return correct flow IDs for the priority band")
		})

		t.Run("Queue", func(t *testing.T) {
			t.Parallel()
			q := accessor.Queue("flow1")
			require.NotNil(t, q, "Accessor should return queue for flow1")
			assert.Equal(t, p1, q.FlowSpec().Priority, "Queue should have the correct priority")
			assert.Nil(t, accessor.Queue("non-existent"), "Accessor should return nil for non-existent flow")
		})

		t.Run("IterateQueues", func(t *testing.T) {
			t.Parallel()
			var iteratedFlows []string
			accessor.IterateQueues(func(queue framework.FlowQueueAccessor) bool {
				iteratedFlows = append(iteratedFlows, queue.FlowSpec().ID)
				return true
			})
			sort.Strings(iteratedFlows)
			assert.Equal(t, []string{"flow1", "flow2"}, iteratedFlows, "IterateQueues should visit all flows in the band")
		})

		t.Run("IterateQueues with early exit", func(t *testing.T) {
			t.Parallel()
			var iteratedFlows []string
			accessor.IterateQueues(func(queue framework.FlowQueueAccessor) bool {
				iteratedFlows = append(iteratedFlows, queue.FlowSpec().ID)
				return false // Exit after first item
			})
			assert.Len(t, iteratedFlows, 1, "IterateQueues should exit early if callback returns false")
		})
	})

	t.Run("Error on non-existent priority", func(t *testing.T) {
		t.Parallel()
		_, err := f.shard.PriorityBandAccessor(99)
		require.Error(t, err, "PriorityBandAccessor should fail for non-existent priority")
		assert.ErrorIs(t, err, contracts.ErrPriorityBandNotFound, "Error should be ErrPriorityBandNotFound")
	})
}
