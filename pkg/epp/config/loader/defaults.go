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
	"slices"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/registry"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	extractormetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	sourcemetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/source/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/scorer/kvcacheutilization"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/scorer/queuedepth"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/saturationdetector/framework/plugins/utilizationdetector"
)

// DefaultScorerWeight is the weight used for scorers referenced in the configuration without explicit weights.
const DefaultScorerWeight = 1.0

var defaultScorerWeight = DefaultScorerWeight

func loadDefaultConfig() *configapi.EndpointPickerConfig {
	queueScorerWeight := 2.0
	kvCacheUtilizationScorerWeight := 2.0
	prefixCacheScorerWeight := 3.0
	return &configapi.EndpointPickerConfig{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "inference.networking.x-k8s.io/v1alpha1",
			Kind:       "EndpointPickerConfig",
		},
		Plugins: []configapi.PluginSpec{
			{
				Type: queuedepth.QueueScorerType,
			},
			{
				Type: kvcacheutilization.KvCacheUtilizationScorerType,
			},
			{
				Type: prefix.PrefixCachePluginType,
			},
		},
		SchedulingProfiles: []configapi.SchedulingProfile{
			{
				Name: "default",
				Plugins: []configapi.SchedulingPlugin{
					{
						PluginRef: queuedepth.QueueScorerType,
						Weight:    &queueScorerWeight,
					},
					{
						PluginRef: kvcacheutilization.KvCacheUtilizationScorerType,
						Weight:    &kvCacheUtilizationScorerWeight,
					},
					{
						PluginRef: prefix.PrefixCachePluginType,
						Weight:    &prefixCacheScorerWeight,
					},
				},
			},
		},
	}
}

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
func applySystemDefaults(cfg *configapi.EndpointPickerConfig, handle fwkplugin.Handle) error {
	allPlugins := handle.GetAllPluginsWithNames()
	if err := ensureSchedulingLayer(cfg, handle, allPlugins); err != nil {
		return fmt.Errorf("failed to apply scheduling system defaults: %w", err)
	}
	if err := ensureFlowControlLayer(cfg, handle, allPlugins); err != nil {
		return fmt.Errorf("failed to apply flow control system defaults: %w", err)
	}
	if err := ensureParser(cfg, handle, allPlugins); err != nil {
		return fmt.Errorf("failed to apply parser defaults: %w", err)
	}
	if err := ensureSaturationDetector(cfg, handle, allPlugins); err != nil {
		return fmt.Errorf("failed to apply saturation detector defaults: %w", err)
	}
	if err := ensureDataLayer(cfg, handle, allPlugins); err != nil {
		return fmt.Errorf("failed to apply data layer defaults: %w", err)
	}
	return nil
}

// ensureSchedulingLayer guarantees that the scheduling subsystem is structurally complete.
// It ensures a valid profile exists and injects missing architectural components (like Pickers and ProfileHandlers) if
// they are not explicitly configured.
func ensureSchedulingLayer(
	cfg *configapi.EndpointPickerConfig,
	handle fwkplugin.Handle,
	allPlugins map[string]fwkplugin.Plugin,
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

// ensureFlowControlLayer guarantees that the flow control subsystem is structurally complete.
func ensureFlowControlLayer(
	cfg *configapi.EndpointPickerConfig,
	handle fwkplugin.Handle,
	allPlugins map[string]fwkplugin.Plugin,
) error {
	if _, ok := allPlugins[registry.DefaultOrderingPolicyRef]; !ok {
		if err := registerDefaultPlugin(cfg, handle, registry.DefaultOrderingPolicyRef); err != nil {
			return err
		}
	}
	if _, ok := allPlugins[registry.DefaultFairnessPolicyRef]; !ok {
		return registerDefaultPlugin(cfg, handle, registry.DefaultFairnessPolicyRef)
	}
	return nil
}

// ensureParser guarantees that parser is configured.
// If the parser is not configured, the openAI parser is configured by default.
func ensureParser(
	cfg *configapi.EndpointPickerConfig,
	handle fwkplugin.Handle,
	allPlugins map[string]fwkplugin.Plugin,
) error {
	parserConfig := cfg.Parser
	if parserConfig == nil {
		parserConfig = &configapi.ParserConfig{
			// Set default parser to openAI parser if the parser is not set in the config.
			PluginRef: openai.OpenAIParserType,
		}
		cfg.Parser = parserConfig
	}
	if _, ok := allPlugins[parserConfig.PluginRef]; !ok {
		if err := registerDefaultPlugin(cfg, handle, openai.OpenAIParserType); err != nil {
			return err
		}
	}
	return nil
}

// ensureSaturationDetector guarantees that saturation detector is configured.
// If the saturation detector is not set, the utilization detector is configured by default.
func ensureSaturationDetector(
	cfg *configapi.EndpointPickerConfig,
	handle fwkplugin.Handle,
	allPlugins map[string]fwkplugin.Plugin,
) error {
	sdConfig := cfg.SaturationDetector
	if sdConfig == nil {
		sdConfig = &configapi.SaturationDetectorConfig{
			PluginRef: utilizationdetector.UtilizationDetectorType,
		}
		cfg.SaturationDetector = sdConfig
	}
	if sdConfig.PluginRef == "" {
		sdConfig.PluginRef = utilizationdetector.UtilizationDetectorType
	}

	if sdConfig.PluginRef == utilizationdetector.UtilizationDetectorType {
		if _, ok := allPlugins[sdConfig.PluginRef]; !ok {
			if err := registerDefaultPlugin(cfg, handle, utilizationdetector.UtilizationDetectorType); err != nil {
				return err
			}
		}
	}
	return nil
}

// ensureDataLayer guarantees that the data layer is configured when the dataLayer feature gate is enabled.
// If the feature gate is enabled but no data section is provided, the default metrics-data-source and core-metrics-extractor plugins are injected.
func ensureDataLayer(
	cfg *configapi.EndpointPickerConfig,
	handle fwkplugin.Handle,
	allPlugins map[string]fwkplugin.Plugin,
) error {
	if !slices.Contains(cfg.FeatureGates, datalayer.ExperimentalDatalayerFeatureGate) {
		return nil
	}

	if cfg.Data != nil && len(cfg.Data.Sources) > 0 {
		return nil
	}

	if _, ok := allPlugins[sourcemetrics.MetricsDataSourceType]; !ok {
		if err := registerDefaultPlugin(cfg, handle, sourcemetrics.MetricsDataSourceType); err != nil {
			return err
		}
	}
	if _, ok := allPlugins[extractormetrics.MetricsExtractorType]; !ok {
		if err := registerDefaultPlugin(cfg, handle, extractormetrics.MetricsExtractorType); err != nil {
			return err
		}
	}

	cfg.Data = &configapi.DataLayerConfig{
		Sources: []configapi.DataLayerSource{{
			PluginRef: sourcemetrics.MetricsDataSourceType,
			Extractors: []configapi.DataLayerExtractor{{
				PluginRef: extractormetrics.MetricsExtractorType,
			}},
		}},
	}

	return nil
}

// registerDefaultPlugin instantiates a plugin with empty configuration (defaults) and adds it to both the handle and
// the config spec.
func registerDefaultPlugin(
	cfg *configapi.EndpointPickerConfig,
	handle fwkplugin.Handle,
	pluginType string,
) error {
	name := pluginType
	factory, ok := fwkplugin.Registry[pluginType]
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
