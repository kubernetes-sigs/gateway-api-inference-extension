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

package config

import (
	"time"

	// Import the upstream Gateway API timeout config
	gatewayconfig "sigs.k8s.io/gateway-api/conformance/utils/config"
)

// InferenceExtensionTimeoutConfig embeds the upstream TimeoutConfig and adds
// extension-specific timeout values.
type InferenceExtensionTimeoutConfig struct {
	// All fields from gatewayconfig.TimeoutConfig will be available directly.
	gatewayconfig.TimeoutConfig

	// InferencePoolMustHaveConditionTimeout represents the maximum time to wait for an InferencePool to have a specific condition.
	InferencePoolMustHaveConditionTimeout time.Duration

	// InferencePoolMustHaveConditionInterval represents the polling interval for checking an InferencePool's condition.
	InferencePoolMustHaveConditionInterval time.Duration

	// GatewayObjectPollInterval is the polling interval used when waiting for a Gateway object to appear.
	GatewayObjectPollInterval time.Duration

	// HTTPRouteDeletionReconciliationTimeout is the time to wait for controllers to reconcile
	// state after an HTTPRoute is deleted, before checking dependent resources or traffic.
	HTTPRouteDeletionReconciliationTimeout time.Duration
}

// DefaultInferenceExtensionTimeoutConfig returns a new InferenceExtensionTimeoutConfig with default values.
func DefaultInferenceExtensionTimeoutConfig() InferenceExtensionTimeoutConfig {
	return InferenceExtensionTimeoutConfig{
		TimeoutConfig:                          gatewayconfig.DefaultTimeoutConfig(), // Initialize embedded struct
		InferencePoolMustHaveConditionTimeout:  300 * time.Second,
		InferencePoolMustHaveConditionInterval: 10 * time.Second,
		GatewayObjectPollInterval:              5 * time.Second,
		HTTPRouteDeletionReconciliationTimeout: 5 * time.Second,
	}
}
