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

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
)

// DefaultScorerWeight is the weight used for scorers referenced in the configuration without explicit weights.
const DefaultScorerWeight = 1

var defaultScorerWeight = DefaultScorerWeight

// applyStaticDefaults standardizes the configuration object before plugin instantiation.
// It handles simple structural defaults that do not require knowledge of the plugin registry.
func applyStaticDefaults(cfg *configapi.EndpointPickerConfig) {
	// Infer plugin names. If a plugin has a Type but no Name, the Type becomes the Name.
	for idx, pluginConfig := range cfg.Plugins {
		if pluginConfig.Name == "" {
			cfg.Plugins[idx].Name = pluginConfig.Type
		}
	}

	// Initialize feature gates.
	if cfg.FeatureGates == nil {
		cfg.FeatureGates = configapi.FeatureGates{}
	}
}

// applySystemDefaults injects required architectural components that were omitted from the config.
// It inspects the instantiated plugins (via the handle) and ensures the system graph is complete.
func applySystemDefaults(cfg *configapi.EndpointPickerConfig, handle plugins.Handle) error {
	allPlugins := handle.GetAllPluginsWithNames()

	// Ensure the scheduling layer has profiles, pickers, and handlers.
	if err := ensureSchedulingArchitecture(cfg, handle, allPlugins); err != nil {
		return fmt.Errorf("failed to apply scheduling system defaults: %w", err)
	}

	return nil
}

// ensureSchedulingArchitecture guarantees that a valid scheduling profile exists and that all profiles have valid
// Pickers and Handlers.
func ensureSchedulingArchitecture(
	cfg *configapi.EndpointPickerConfig,
	handle plugins.Handle,
	allPlugins map[string]plugins.Plugin,
) error {
	// Ensure at least one Scheduling Profile exists.
	if len(cfg.SchedulingProfiles) == 0 {
		defaultProfile := configapi.SchedulingProfile{Name: "default"}
		// Auto-populate the default profile with all Filter, Scorer, and Picker plugins found.
		for name, p := range allPlugins {
			switch p.(type) {
			case framework.Filter, framework.Scorer, framework.Picker:
				defaultProfile.Plugins = append(defaultProfile.Plugins, configapi.SchedulingPlugin{PluginRef: name})
			}
		}
		cfg.SchedulingProfiles = []configapi.SchedulingProfile{defaultProfile}
	}

	// Ensure profile handler.
	// If there is only 1 profile and no handler is explicitly configured, use the SingleProfileHandler.
	if len(cfg.SchedulingProfiles) == 1 {
		hasHandler := false
		for _, p := range allPlugins {
			if _, ok := p.(framework.ProfileHandler); ok {
				hasHandler = true
				break
			}
		}
		if !hasHandler {
			if err := registerDefaultPlugin(cfg, handle, profile.SingleProfileHandlerType, profile.SingleProfileHandlerType); err != nil {
				return err
			}
		}
	}

	// Ensure Picker(s) and Scorer weights.
	// Find or Create a default MaxScorePicker to reuse across profiles.
	var maxScorePickerName string
	for name, p := range allPlugins {
		if _, ok := p.(framework.Picker); ok {
			maxScorePickerName = name
			break
		}
	}
	// If no Picker exists anywhere, create one.
	if maxScorePickerName == "" {
		if err := registerDefaultPlugin(cfg, handle, picker.MaxScorePickerType, picker.MaxScorePickerType); err != nil {
			return err
		}
		maxScorePickerName = picker.MaxScorePickerType
	}

	// Update profiles.
	for i, prof := range cfg.SchedulingProfiles {
		hasPicker := false
		for j, pluginRef := range prof.Plugins {
			p := handle.Plugin(pluginRef.PluginRef)

			// Default Scorer weight.
			if _, ok := p.(framework.Scorer); ok && pluginRef.Weight == nil {
				cfg.SchedulingProfiles[i].Plugins[j].Weight = &defaultScorerWeight
			}

			// Check for Picker.
			if _, ok := p.(framework.Picker); ok {
				hasPicker = true
			}
		}

		// Inject default Picker if missing.
		if !hasPicker {
			cfg.SchedulingProfiles[i].Plugins = append(
				cfg.SchedulingProfiles[i].Plugins,
				configapi.SchedulingPlugin{PluginRef: maxScorePickerName},
			)
		}
	}

	return nil
}

// registerDefaultPlugin instantiates a plugin with empty configuration (defaults) and adds it to both the handle and
// the config spec.
func registerDefaultPlugin(
	cfg *configapi.EndpointPickerConfig,
	handle plugins.Handle,
	name string,
	pluginType string,
) error {
	factory, ok := plugins.Registry[pluginType]
	if !ok {
		return fmt.Errorf("plugin type '%s' not found in registry", pluginType)
	}

	// Instantiate with nil config (factory must handle defaults).
	plugin, err := factory(name, nil, handle)
	if err != nil {
		return fmt.Errorf("failed to instantiate default plugin '%s': %w", name, err)
	}

	handle.AddPlugin(name, plugin)
	cfg.Plugins = append(cfg.Plugins, configapi.PluginSpec{
		Name: name,
		Type: pluginType,
	})

	return nil
}
