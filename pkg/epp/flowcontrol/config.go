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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/controller"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/registry"
)

const (
	FeatureGate = "flowControl"
)

// Config is the top-level configuration for the entire flow control module.
// It embeds the configurations for the controller and the registry, providing a single point of entry for validation
// and initialization.
type Config struct {
	Controller controller.Config
	Registry   registry.Config
}

// ValidateAndApplyDefaults checks the configuration for validity and populates any empty fields with system defaults.
// It delegates validation to the underlying controller and registry configurations.
// It returns a new, validated `Config` object and does not mutate the receiver.
func (c *Config) ValidateAndApplyDefaults() (*Config, error) {
	validatedControllerCfg, err := c.Controller.ValidateAndApplyDefaults()
	if err != nil {
		return nil, fmt.Errorf("controller config validation failed: %w", err)
	}
	validatedRegistryCfg, err := c.Registry.ValidateAndApplyDefaults()
	if err != nil {
		return nil, fmt.Errorf("registry config validation failed: %w", err)
	}
	return &Config{
		Controller: *validatedControllerCfg,
		Registry:   *validatedRegistryCfg,
	}, nil
}
