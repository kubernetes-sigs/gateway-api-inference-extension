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

package controller

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConfig_ValidateAndApplyDefaults(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name          string
		input         Config
		expectErr     bool
		expectedCfg   Config
		shouldDefault bool
	}{
		{
			name: "ValidConfig_NoChanges",
			input: Config{
				DefaultRequestTTL:               10 * time.Second,
				ExpiryCleanupInterval:           2 * time.Second,
				ProcessorReconciliationInterval: 10 * time.Second,
				EnqueueChannelBufferSize:        200,
			},
			expectErr: false,
			expectedCfg: Config{
				DefaultRequestTTL:               10 * time.Second,
				ExpiryCleanupInterval:           2 * time.Second,
				ProcessorReconciliationInterval: 10 * time.Second,
				EnqueueChannelBufferSize:        200,
			},
		},
		{
			name:      "EmptyConfig_ShouldApplyDefaults",
			input:     Config{},
			expectErr: false,
			expectedCfg: Config{
				DefaultRequestTTL:               0,
				ExpiryCleanupInterval:           defaultExpiryCleanupInterval,
				ProcessorReconciliationInterval: defaultProcessorReconciliationInterval,
				EnqueueChannelBufferSize:        defaultEnqueueChannelBufferSize,
			},
			shouldDefault: true,
		},
		{
			name:      "NegativeDefaultRequestTTL_Invalid",
			input:     Config{DefaultRequestTTL: -1},
			expectErr: true,
		},
		{
			name:      "NegativeExpiryCleanupInterval_Invalid",
			input:     Config{ExpiryCleanupInterval: -1},
			expectErr: true,
		},
		{
			name:      "NegativeProcessorReconciliationInterval_Invalid",
			input:     Config{ProcessorReconciliationInterval: -1},
			expectErr: true,
		},
		{
			name:      "NegativeEnqueueChannelBufferSize_Invalid",
			input:     Config{EnqueueChannelBufferSize: -1},
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			originalInput := tc.input.deepCopy()
			validatedCfg, err := tc.input.ValidateAndApplyDefaults()

			if tc.expectErr {
				require.Error(t, err, "expected an error but got nil")
				assert.Nil(t, validatedCfg, "validatedCfg should be nil on error")
			} else {
				require.NoError(t, err, "expected no error but got: %v", err)
				require.NotNil(t, validatedCfg, "validatedCfg should not be nil on success")
				assert.Equal(t, tc.expectedCfg, *validatedCfg, "validatedCfg should match expected config")

				// Ensure the original config is not mutated.
				assert.Equal(t, *originalInput, tc.input, "input config should not be mutated")
			}
		})
	}
}

func TestConfig_DeepCopy(t *testing.T) {
	t.Parallel()

	t.Run("ShouldReturnNil_ForNilReceiver", func(t *testing.T) {
		t.Parallel()
		var nilConfig *Config
		assert.Nil(t, nilConfig.deepCopy(), "Deep copy of a nil config should be nil")
	})

	t.Run("ShouldCreateIdenticalButSeparateObject", func(t *testing.T) {
		t.Parallel()
		original := &Config{
			DefaultRequestTTL:               1 * time.Second,
			ExpiryCleanupInterval:           2 * time.Second,
			ProcessorReconciliationInterval: 3 * time.Second,
			EnqueueChannelBufferSize:        4,
		}
		clone := original.deepCopy()

		require.NotSame(t, original, clone, "Clone should be a new object in memory")
		assert.Equal(t, *original, *clone, "Cloned object should have identical values")

		// Modify the clone and ensure the original is unchanged.
		clone.DefaultRequestTTL = 99 * time.Second
		assert.NotEqual(t, original.DefaultRequestTTL, clone.DefaultRequestTTL,
			"Original should not be mutated after clone is changed")
	})
}
