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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

func TestNewConfig(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name        string
		opts        []ConfigOption
		expectErr   bool
		expectedCfg Config
	}{
		{
			name:      "Defaults_ShouldBeApplied_WhenNoOptionsProvided",
			opts:      nil,
			expectErr: false,
			expectedCfg: Config{
				DefaultRequestTTL:               0,
				ExpiryCleanupInterval:           defaultExpiryCleanupInterval,
				ProcessorReconciliationInterval: defaultProcessorReconciliationInterval,
				EnqueueChannelBufferSize:        defaultEnqueueChannelBufferSize,
			},
		},
		{
			name: "WithDefaultRequestTTL_ShouldUpdateConfig",
			opts: []ConfigOption{
				WithDefaultRequestTTL(10 * time.Second),
			},
			expectErr: false,
			expectedCfg: Config{
				DefaultRequestTTL:               10 * time.Second,
				ExpiryCleanupInterval:           defaultExpiryCleanupInterval,
				ProcessorReconciliationInterval: defaultProcessorReconciliationInterval,
				EnqueueChannelBufferSize:        defaultEnqueueChannelBufferSize,
			},
		},
		{
			name: "WithAllOptions_ShouldUpdateConfig",
			opts: []ConfigOption{
				WithDefaultRequestTTL(10 * time.Second),
				WithExpiryCleanupInterval(2 * time.Second),
				WithProcessorReconciliationInterval(10 * time.Second),
				WithEnqueueChannelBufferSize(50),
			},
			expectErr: false,
			expectedCfg: Config{
				DefaultRequestTTL:               10 * time.Second,
				ExpiryCleanupInterval:           2 * time.Second,
				ProcessorReconciliationInterval: 10 * time.Second,
				EnqueueChannelBufferSize:        50,
			},
		},
		{
			name: "NegativeDefaultRequestTTL_ShouldError",
			opts: []ConfigOption{
				WithDefaultRequestTTL(-1 * time.Second),
			},
			expectErr: true,
		},
		{
			name: "InvalidExpiryCleanupInterval_ShouldError",
			opts: []ConfigOption{
				WithExpiryCleanupInterval(-1 * time.Second),
			},
			expectErr: true,
		},
		{
			name: "ZeroExpiryCleanupInterval_ShouldError",
			opts: []ConfigOption{
				WithExpiryCleanupInterval(0),
			},
			expectErr: true,
		},
		{
			name: "InvalidProcessorReconciliationInterval_ShouldError",
			opts: []ConfigOption{
				WithProcessorReconciliationInterval(-1 * time.Second),
			},
			expectErr: true,
		},
		{
			name: "InvalidEnqueueChannelBufferSize_ShouldError",
			opts: []ConfigOption{
				WithEnqueueChannelBufferSize(-1),
			},
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			cfg, err := NewConfig(tc.opts...)

			if tc.expectErr {
				require.Error(t, err, "NewConfig should return an error for invalid input")
				assert.Nil(t, cfg, "Config should be nil when error occurs")
			} else {
				require.NoError(t, err, "NewConfig should not error for valid input")
				require.NotNil(t, cfg, "Config should not be nil on success")
				assert.Equal(t, tc.expectedCfg, *cfg, "Config should match expected structure")
			}
		})
	}
}

func TestNewConfigFromAPI(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name        string
		apiConfig   *configapi.FlowControlConfig
		assertion   func(*testing.T, *Config)
		expectedErr string
	}{
		{
			name:      "NilConfig_ShouldReturnSystemDefaults",
			apiConfig: nil,
			assertion: func(t *testing.T, cfg *Config) {
				assert.Equal(t, defaultExpiryCleanupInterval, cfg.ExpiryCleanupInterval,
					"ExpiryCleanupInterval should be defaulted")
				assert.Equal(t, defaultProcessorReconciliationInterval, cfg.ProcessorReconciliationInterval,
					"ProcessorReconciliationInterval should be defaulted")
				assert.Equal(t, defaultEnqueueChannelBufferSize, cfg.EnqueueChannelBufferSize,
					"EnqueueChannelBufferSize should be defaulted")
				assert.Equal(t, time.Duration(0), cfg.DefaultRequestTTL, "DefaultRequestTTL should default to 0 (disabled)")
			},
		},
		{
			name: "ValidConfig_ShouldTranslateFields",
			apiConfig: &configapi.FlowControlConfig{
				DefaultRequestTTL: &metav1.Duration{Duration: 5 * time.Minute},
			},
			assertion: func(t *testing.T, cfg *Config) {
				assert.Equal(t, 5*time.Minute, cfg.DefaultRequestTTL, "DefaultRequestTTL should be translated")
				// Defaults should still be applied for others
				assert.Equal(t, defaultExpiryCleanupInterval, cfg.ExpiryCleanupInterval)
			},
		},
		{
			name: "ValidConfig_ShouldTranslateAllExposedFields",
			apiConfig: &configapi.FlowControlConfig{
				DefaultRequestTTL:     &metav1.Duration{Duration: 1 * time.Minute},
				ExpiryCleanupInterval: &metav1.Duration{Duration: 500 * time.Millisecond},
			},
			assertion: func(t *testing.T, cfg *Config) {
				assert.Equal(t, 1*time.Minute, cfg.DefaultRequestTTL)
				assert.Equal(t, 500*time.Millisecond, cfg.ExpiryCleanupInterval)
				// ProcessorReconciliationInterval is not exposed, so it should stay default.
				assert.Equal(t, defaultProcessorReconciliationInterval, cfg.ProcessorReconciliationInterval)
			},
		},
		{
			name: "ExplicitZeroRequestTTL_ShouldBeRespected",
			apiConfig: &configapi.FlowControlConfig{
				DefaultRequestTTL: &metav1.Duration{Duration: 0},
			},
			assertion: func(t *testing.T, cfg *Config) {
				assert.Equal(t, time.Duration(0), cfg.DefaultRequestTTL, "Explicit 0s TTL should be respected")
			},
		},
		{
			name: "InvalidConfig_NegativeRequestTTL_ShouldError",
			apiConfig: &configapi.FlowControlConfig{
				DefaultRequestTTL: &metav1.Duration{Duration: -1 * time.Minute},
			},
			expectedErr: "DefaultRequestTTL cannot be negative",
		},
		{
			name: "InvalidConfig_NegativeExpiryInterval_ShouldError",
			apiConfig: &configapi.FlowControlConfig{
				ExpiryCleanupInterval: &metav1.Duration{Duration: -1 * time.Second},
			},
			expectedErr: "ExpiryCleanupInterval must be positive",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			cfg, err := NewConfigFromAPI(tc.apiConfig)

			if tc.expectedErr != "" {
				require.Error(t, err, "NewConfigFromAPI should error for invalid config")
				assert.Contains(t, err.Error(), tc.expectedErr)
				assert.Nil(t, cfg)
			} else {
				require.NoError(t, err, "NewConfigFromAPI should not error")
				require.NotNil(t, cfg, "Config should not be nil")
				if tc.assertion != nil {
					tc.assertion(t, cfg)
				}
			}
		})
	}
}
