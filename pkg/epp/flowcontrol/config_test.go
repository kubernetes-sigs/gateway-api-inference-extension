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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/controller"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/registry"
)

func TestConfig_ValidateAndApplyDefaults(t *testing.T) {
	t.Parallel()

	// A minimal valid registry config, which is required for the success case.
	validRegistryConfig := registry.Config{
		PriorityBands: []registry.PriorityBandConfig{
			{Priority: 1, PriorityName: "TestBand"},
		},
	}

	testCases := []struct {
		name          string
		input         Config
		expectErr     bool
		expectedErrIs error
	}{
		{
			name: "ShouldSucceed_WhenSubConfigsAreValid",
			input: Config{
				Controller: controller.Config{},
				Registry:   validRegistryConfig,
			},
			expectErr: false,
		},
		{
			name: "ShouldFail_WhenControllerConfigIsInvalid",
			input: Config{
				Controller: controller.Config{
					DefaultRequestTTL: -1 * time.Second,
				},
				Registry: validRegistryConfig,
			},
			expectErr: true,
		},
		{
			name: "ShouldFail_WhenRegistryConfigIsInvalid",
			input: Config{
				Controller: controller.Config{},
				Registry: registry.Config{
					PriorityBands: []registry.PriorityBandConfig{},
				},
			},
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			originalInput := tc.input
			validatedCfg, err := tc.input.ValidateAndApplyDefaults()

			if tc.expectErr {
				require.Error(t, err, "expected an error but got nil")
			} else {
				require.NoError(t, err, "expected no error but got: %v", err)
				require.NotNil(t, validatedCfg, "validatedCfg should not be nil on success")
			}

			assert.Equal(t, originalInput, tc.input, "input config should not be mutated")
		})
	}
}
