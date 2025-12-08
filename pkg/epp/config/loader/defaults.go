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

// applyStaticDefaults sanitizes the configuration object before plugin instantiation.
// It handles "Static" defaults: simple structural changes to the API object that do not require access to the plugin
// registry.
func applyStaticDefaults(cfg *configapi.EndpointPickerConfig) {
	for idx, pluginConfig := range cfg.Plugins {
		if pluginConfig.Name == "" {
			cfg.Plugins[idx].Name = pluginConfig.Type
		}
	}

	if cfg.FeatureGates == nil {
		cfg.FeatureGates = configapi.FeatureGates{}
	}
}

// applySystemDefaults injects required components that were omitted from the config.
// It handles "System" defaults: logic that requires inspecting instantiated plugins (via the handle) to ensure the
// system graph is complete.
func applySystemDefaults(cfg *configapi.EndpointPickerConfig, handle plugins.Handle) error {
	allPlugins := handle.GetAllPluginsWithNames()
	if err := ensureSchedulingLayer(cfg, handle, allPlugins); err != nil {
		return fmt.Errorf("failed to apply scheduling system defaults: %w", err)
	}
	return nil
}

// ensureSchedulingLayer guarantees that the scheduling subsystem is structurally complete.
// It ensures a valid profile exists and injects missing architectural components (like Pickers and ProfileHandlers) if
// they are not explicitly configured.
func ensureSchedulingLayer(
	cfg *configapi.EndpointPickerConfig,
	handle plugins.Handle,
	allPlugins map[string]plugins.Plugin,
) error {
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
			if err := registerDefaultPlugin(cfg, handle, profile.SingleProfileHandlerType); err != nil {
				return err
			}
		}
	}

	// Find or Create a default MaxScorePicker to reuse across profiles.
	var maxScorePickerName string
	for name, p := range allPlugins {
		if _, ok := p.(framework.Picker); ok {
			maxScorePickerName = name
			break
		}
	}

	if maxScorePickerName == "" {
		if err := registerDefaultPlugin(cfg, handle, picker.MaxScorePickerType); err != nil {
			return err
		}
		maxScorePickerName = picker.MaxScorePickerType
	}

	for i, prof := range cfg.SchedulingProfiles {
		hasPicker := false
		for j, pluginRef := range prof.Plugins {
			p := handle.Plugin(pluginRef.PluginRef)

			if _, ok := p.(framework.Scorer); ok && pluginRef.Weight == nil {
				cfg.SchedulingProfiles[i].Plugins[j].Weight = &defaultScorerWeight
			}

			if _, ok := p.(framework.Picker); ok {
				hasPicker = true
			}
		}

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
	pluginType string,
) error {
	name := pluginType
	factory, ok := plugins.Registry[pluginType]
	if !ok {
		return fmt.Errorf("plugin type '%s' not found in registry", pluginType)
	}

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
