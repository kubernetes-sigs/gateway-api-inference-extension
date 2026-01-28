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
