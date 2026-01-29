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
	"fmt"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/controller"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/registry"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const FeatureGate = "flowControl"

// Config is the top-level configuration for the entire flow control module.
// It embeds the configurations for the controller and the registry, providing a single point of entry for validation
// and initialization.
type Config struct {
	Controller *controller.Config
	Registry   *registry.Config
}

// NewConfigFromAPI creates a new Config by translating the top-level API configuration.
func NewConfigFromAPI(apiConfig *configapi.FlowControlConfig, handle plugin.Handle) (*Config, error) {
	registryConfig, err := registry.NewConfigFromAPI(apiConfig, handle)
	if err != nil {
		return nil, fmt.Errorf("failed to create registry config: %w", err)
	}
	ctrlCfg, err := controller.NewConfigFromAPI(apiConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create controller config: %w", err)
	}
	return &Config{
		Controller: ctrlCfg,
		Registry:   registryConfig,
	}, nil
}
