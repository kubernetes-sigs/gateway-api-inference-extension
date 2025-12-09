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

package loader

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

// validateConfig performs a deep validation of the configuration integrity.
// It checks relationships between profiles, plugins, and feature gates.
func validateConfig(cfg *configapi.EndpointPickerConfig) error {
	if err := validateFeatureGates(cfg.FeatureGates); err != nil {
		return fmt.Errorf("feature gate validation failed: %w", err)
	}
	if err := validateSchedulingProfiles(cfg); err != nil {
		return fmt.Errorf("scheduling profile validation failed: %w", err)
	}
	return nil
}

func validateSchedulingProfiles(cfg *configapi.EndpointPickerConfig) error {
	definedPlugins := sets.New[string]()
	for _, p := range cfg.Plugins {
		definedPlugins.Insert(p.Name)
	}
	seenProfileNames := sets.New[string]()

	for i, profile := range cfg.SchedulingProfiles {
		if profile.Name == "" {
			return fmt.Errorf("schedulingProfiles[%d] is missing a name", i)
		}
		if seenProfileNames.Has(profile.Name) {
			return fmt.Errorf("schedulingProfiles[%d] has duplicate name '%s'", i, profile.Name)
		}
		seenProfileNames.Insert(profile.Name)

		for j, pluginRef := range profile.Plugins {
			if pluginRef.PluginRef == "" {
				return fmt.Errorf("schedulingProfiles[%s].plugins[%d] is missing a 'pluginRef'", profile.Name, j)
			}

			if !definedPlugins.Has(pluginRef.PluginRef) {
				return fmt.Errorf("schedulingProfiles[%s] references undefined plugin '%s'",
					profile.Name, pluginRef.PluginRef)
			}
		}
	}
	return nil
}

func validateFeatureGates(gates configapi.FeatureGates) error {
	if gates == nil {
		return nil
	}

	for _, gate := range gates {
		if !registeredFeatureGates.Has(gate) {
			return fmt.Errorf("feature gate '%s' is unknown or unregistered", gate)
		}
	}

	return nil
}
