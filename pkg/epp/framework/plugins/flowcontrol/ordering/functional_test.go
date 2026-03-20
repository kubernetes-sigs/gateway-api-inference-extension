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

package ordering

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// TestOrderingPolicyConformance is the main conformance test suite for OrderingPolicy implementations.
// It iterates over all policy implementations registered via plugin.Register and runs a series of sub-tests to
// ensure they adhere to the OrderingPolicy contract.
func TestOrderingPolicyConformance(t *testing.T) {
	t.Parallel()

	if len(plugin.Registry) == 0 {
		t.Log("No plugins registered. Skipping conformance tests.")
		return
	}

	for name, factory := range plugin.Registry {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			plugin, err := factory(name, nil, nil)
			require.NoError(t, err, "Factory failed for plugin %s", name)
			require.NotNil(t, plugin, "Factory returned nil for plugin %s", name)
			policy, ok := plugin.(flowcontrol.OrderingPolicy)
			if !ok {
				return
			}

			t.Run("Initialization", func(t *testing.T) {
				t.Parallel()
				assert.Equal(t, name, policy.TypedName().Name, "TypedName().Name should match registered name")

				caps := policy.RequiredQueueCapabilities()
				assert.NotNil(t, caps, "RequiredQueueCapabilities() should not return nil slice")
			})

			t.Run("Less_Sanity", func(t *testing.T) {
				t.Parallel()
				res := policy.Less(nil, nil)
				assert.False(t, res, "Less(nil, nil) should be false")

				mockItem := mocks.NewMockQueueItemAccessor(0, "test", flowcontrol.FlowKey{})
				assert.True(t, policy.Less(mockItem, nil), "Less(item, nil) should be true (item preferred over nil)")
				assert.False(t, policy.Less(nil, mockItem), "Less(nil, item) should be false")
			})
		})
	}
}
