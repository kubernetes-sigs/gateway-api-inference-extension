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

package interflow

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugin"
)

// TestFairnessPolicyConformance is the main conformance test suite for FairnessPolicy implementations.
// It iterates over all policy implementations registered in the interflow package and runs a series of sub-tests to
// ensure they adhere to the FairnessPolicy contract.
func TestFairnessPolicyConformance(t *testing.T) {
	t.Parallel()

	if len(fwkplugin.Registry) == 0 {
		t.Log("No plugins registered. Skipping conformance tests.")
		return
	}

	for name, f := range fwkplugin.Registry {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			// We pass nil for the handle as our core policies (RoundRobin, GlobalStrict) do not depend on it.
			plugin, err := f(name, json.RawMessage{}, nil)
			require.NoError(t, err, "Factory failed for plugin %s", name)
			require.NotNil(t, plugin, "Factory returned nil for plugin %s", name)

			policy, ok := plugin.(framework.FairnessPolicy)
			if !ok {
				t.Skipf("Plugin %s is not a FairnessPolicy. Skipping.", name)
			}

			t.Run("Initialization", func(t *testing.T) {
				assert.Equal(t, policy.TypedName().Name, name, "TypedName().Name should match registered name")
			})

			t.Run("Pick", func(t *testing.T) {
				runPickConformanceTests(t, policy)
			})
		})
	}
}

func runPickConformanceTests(t *testing.T, policy framework.FairnessPolicy) {
	t.Helper()
	ctx := context.Background()

	// Initialize the Flyweight state for this test run.
	state := policy.NewState(ctx)

	flowIDEmpty := "flow-empty"
	mockQueueEmpty := &frameworkmocks.MockFlowQueueAccessor{
		LenV:      0,
		PeekHeadV: nil,
		FlowKeyV:  types.FlowKey{ID: flowIDEmpty},
	}

	testCases := []struct {
		name          string
		band          framework.PriorityBandAccessor
		expectErr     bool
		expectNil     bool
		expectedQueue framework.FlowQueueAccessor
	}{
		{
			name:      "With a nil priority band accessor",
			band:      nil,
			expectErr: false,
			expectNil: true,
		},
		{
			name: "With an empty priority band accessor",
			band: &frameworkmocks.MockPriorityBandAccessor{
				PolicyStateV:      state,
				FlowKeysFunc:      func() []types.FlowKey { return []types.FlowKey{} },
				IterateQueuesFunc: func(callback func(flow framework.FlowQueueAccessor) bool) { /* no-op */ },
			},
			expectErr: false,
			expectNil: true,
		},
		{
			name: "With a band that has one empty queue",
			band: &frameworkmocks.MockPriorityBandAccessor{
				PolicyStateV: state,
				FlowKeysFunc: func() []types.FlowKey { return []types.FlowKey{{ID: flowIDEmpty}} },
				QueueFunc: func(fID string) framework.FlowQueueAccessor {
					if fID == flowIDEmpty {
						return mockQueueEmpty
					}
					return nil
				},
				IterateQueuesFunc: func(callback func(flow framework.FlowQueueAccessor) bool) { callback(mockQueueEmpty) },
			},
			expectErr: false,
			expectNil: true,
		},
		{
			name: "With a band that has multiple empty queues",
			band: &frameworkmocks.MockPriorityBandAccessor{
				PolicyStateV: state,
				FlowKeysFunc: func() []types.FlowKey { return []types.FlowKey{{ID: flowIDEmpty}, {ID: "flow-empty-2"}} },
				QueueFunc:    func(fID string) framework.FlowQueueAccessor { return mockQueueEmpty },
				IterateQueuesFunc: func(callback func(flow framework.FlowQueueAccessor) bool) {
					// Iterate over two empty queues.
					if !callback(mockQueueEmpty) {
						return
					}
					callback(mockQueueEmpty)
				},
			},
			expectErr: false,
			expectNil: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			selectedQueue, err := policy.Pick(ctx, tc.band)

			if tc.expectErr {
				require.Error(t, err, "Pick() should return an error")
			} else {
				require.NoError(t, err, "Pick() should not return an error")
			}

			if tc.expectNil {
				assert.Nil(t, selectedQueue, "Pick() should return a nil queue")
			} else {
				assert.NotNil(t, selectedQueue, "Pick() should not return a nil queue")
			}

			if tc.expectedQueue != nil {
				assert.Equal(t, tc.expectedQueue.FlowKey(), selectedQueue.FlowKey(),
					"Pick() returned an unexpected queue")
			}
		})
	}
}
