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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// TestInterFlowDispatchPolicy_Conformance is the main conformance test suite for `framework.InterFlowDispatchPolicy`
// implementations.
// It iterates over all policy implementations registered via `dispatch.MustRegisterPolicy` and runs a series of
// sub-tests to ensure they adhere to the `framework.InterFlowDispatchPolicy` contract.
func TestInterFlowDispatchPolicyConformance(t *testing.T) {
	t.Parallel()

	for policyName, constructor := range RegisteredPolicies {
		t.Run(string(policyName), func(t *testing.T) {
			t.Parallel()

			policy, err := constructor()
			require.NoError(t, err, "Policy constructor for %s failed", policyName)
			require.NotNil(t, policy, "Constructor for %s should return a non-nil policy instance", policyName)

			t.Run("Initialization", func(t *testing.T) {
				t.Parallel()
				assert.NotEmpty(t, policy.Name(), "Name() for %s should return a non-empty string", policyName)
			})

			t.Run("SelectQueue", func(t *testing.T) {
				t.Parallel()
				runSelectQueueConformanceTests(t, policy)
			})
		})
	}
}

func runSelectQueueConformanceTests(t *testing.T, policy framework.InterFlowDispatchPolicy) {
	t.Helper()

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
				FlowKeysFunc:      func() []types.FlowKey { return []types.FlowKey{} },
				IterateQueuesFunc: func(callback func(queue framework.FlowQueueAccessor) bool) { /* no-op */ },
			},
			expectErr: false,
			expectNil: true,
		},
		{
			name: "With a band that has one empty queue",
			band: &frameworkmocks.MockPriorityBandAccessor{
				FlowKeysFunc: func() []types.FlowKey { return []types.FlowKey{{ID: flowIDEmpty}} },
				QueueFunc: func(fID string) framework.FlowQueueAccessor {
					if fID == flowIDEmpty {
						return mockQueueEmpty
					}
					return nil
				},
			},
			expectErr: false,
			expectNil: true,
		},
		{
			name: "With a band that has multiple empty queues",
			band: &frameworkmocks.MockPriorityBandAccessor{
				FlowKeysFunc: func() []types.FlowKey { return []types.FlowKey{{ID: flowIDEmpty}, {ID: "flow-empty-2"}} },
				QueueFunc: func(fID string) framework.FlowQueueAccessor {
					return mockQueueEmpty
				},
			},
			expectErr: false,
			expectNil: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			selectedQueue, err := policy.SelectQueue(tc.band)

			if tc.expectErr {
				require.Error(t, err, "SelectQueue for policy %s should return an error", policy.Name())
			} else {
				require.NoError(t, err, "SelectQueue for policy %s should not return an error", policy.Name())
			}

			if tc.expectNil {
				assert.Nil(t, selectedQueue, "SelectQueue for policy %s should return a nil queue", policy.Name())
			} else {
				assert.NotNil(t, selectedQueue, "SelectQueue for policy %s should not return a nil queue", policy.Name())
			}

			if tc.expectedQueue != nil {
				assert.Equal(t, tc.expectedQueue.FlowKey(), selectedQueue.FlowKey(),
					"SelectQueue for policy %s returned an unexpected queue", policy.Name())
			}
		})
	}
}
