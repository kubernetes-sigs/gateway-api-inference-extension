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
	"errors"
	"fmt"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/saturationdetector/framework/plugins/utilizationdetector"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
)

var (
	scheme                 = runtime.NewScheme()
	registeredFeatureGates = sets.New[string]()
)

func init() {
	utilruntime.Must(configapi.Install(scheme))
}

// RegisterFeatureGate registers a feature gate name for validation purposes.
func RegisterFeatureGate(gate string) {
	registeredFeatureGates.Insert(gate)
}

// LoadRawConfig parses the raw configuration bytes, applies initial defaults, and extracts feature gates.
// It does not instantiate plugins.
func LoadRawConfig(configBytes []byte, logger logr.Logger) (*configapi.EndpointPickerConfig, map[string]bool, error) {
	rawConfig, err := decodeRawConfig(configBytes)
	if err != nil {
		return nil, nil, err
	}
	logger.Info("Loaded raw configuration", "config", rawConfig)

	applyStaticDefaults(rawConfig)

	// We validate gates early because they might dictate downstream loading logic.
	if err := validateFeatureGates(rawConfig.FeatureGates); err != nil {
		return nil, nil, fmt.Errorf("feature gate validation failed: %w", err)
	}

	featureConfig := loadFeatureConfig(rawConfig.FeatureGates)
	return rawConfig, featureConfig, nil
}

// InstantiateAndConfigure performs the heavy lifting of plugin instantiation, system architecture injection, and
// scheduler construction.
func InstantiateAndConfigure(
	rawConfig *configapi.EndpointPickerConfig,
	handle plugins.Handle,
	logger logr.Logger,
) (*config.Config, error) {

	if err := instantiatePlugins(rawConfig.Plugins, handle); err != nil {
		return nil, fmt.Errorf("plugin instantiation failed: %w", err)
	}

	if err := applySystemDefaults(rawConfig, handle); err != nil {
		return nil, fmt.Errorf("system default application failed: %w", err)
	}
	logger.Info("Effective configuration loaded", "config", rawConfig)

	if err := validateConfig(rawConfig); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}

	schedulerConfig, err := buildSchedulerConfig(rawConfig.SchedulingProfiles, handle)
	if err != nil {
		return nil, fmt.Errorf("scheduler config build failed: %w", err)
	}

	featureGates := loadFeatureConfig(rawConfig.FeatureGates)
	dataConfig, err := buildDataLayerConfig(rawConfig.Data, featureGates[datalayer.ExperimentalDatalayerFeatureGate], handle)
	if err != nil {
		return nil, fmt.Errorf("data layer config build failed: %w", err)
	}

	return &config.Config{
		SchedulerConfig:          schedulerConfig,
		SaturationDetectorConfig: buildSaturationConfig(rawConfig.SaturationDetector),
		DataConfig:               dataConfig,
	}, nil
}

func decodeRawConfig(configBytes []byte) (*configapi.EndpointPickerConfig, error) {
	cfg := &configapi.EndpointPickerConfig{}
	codecs := serializer.NewCodecFactory(scheme, serializer.EnableStrict)
	if err := runtime.DecodeInto(codecs.UniversalDecoder(), configBytes, cfg); err != nil {
		return nil, fmt.Errorf("failed to decode configuration JSON/YAML: %w", err)
	}
	return cfg, nil
}

func instantiatePlugins(configuredPlugins []configapi.PluginSpec, handle plugins.Handle) error {
	pluginNames := sets.New[string]()
	for _, spec := range configuredPlugins {
		if spec.Type == "" {
			return fmt.Errorf("plugin '%s' is missing a type", spec.Name)
		}
		if pluginNames.Has(spec.Name) {
			return fmt.Errorf("duplicate plugin name '%s'", spec.Name)
		}
		pluginNames.Insert(spec.Name)

		factory, ok := plugins.Registry[spec.Type]
		if !ok {
			return fmt.Errorf("plugin type '%s' is not registered", spec.Type)
		}

		plugin, err := factory(spec.Name, spec.Parameters, handle)
		if err != nil {
			return fmt.Errorf("failed to create plugin '%s' (type: %s): %w", spec.Name, spec.Type, err)
		}

		handle.AddPlugin(spec.Name, plugin)
	}
	return nil
}

func buildSchedulerConfig(
	configProfiles []configapi.SchedulingProfile,
	handle plugins.Handle,
) (*scheduling.SchedulerConfig, error) {

	profiles := make(map[string]*framework.SchedulerProfile)

	for _, cfgProfile := range configProfiles {
		fwProfile := framework.NewSchedulerProfile()

		for _, pluginRef := range cfgProfile.Plugins {
			plugin := handle.Plugin(pluginRef.PluginRef)
			if plugin == nil { // Should be caught by validation, but defensive check.
				return nil, fmt.Errorf(
					"plugin '%s' referenced in profile '%s' not found in handle",
					pluginRef.PluginRef, cfgProfile.Name)
			}

			// Wrap Scorers with weights.
			if scorer, ok := plugin.(framework.Scorer); ok {
				weight := DefaultScorerWeight
				if pluginRef.Weight != nil {
					weight = *pluginRef.Weight
				}
				plugin = framework.NewWeightedScorer(scorer, weight)
			}

			if err := fwProfile.AddPlugins(plugin); err != nil {
				return nil, fmt.Errorf("failed to add plugin '%s' to profile '%s': %w", pluginRef.PluginRef, cfgProfile.Name, err)
			}
		}
		profiles[cfgProfile.Name] = fwProfile
	}

	var profileHandler framework.ProfileHandler
	for name, plugin := range handle.GetAllPluginsWithNames() {
		if ph, ok := plugin.(framework.ProfileHandler); ok {
			if profileHandler != nil {
				return nil, fmt.Errorf("multiple profile handlers found ('%s', '%s'); only one is allowed",
					profileHandler.TypedName().Name, name)
			}
			profileHandler = ph
		}
	}

	if profileHandler == nil {
		return nil, errors.New("no profile handler configured")
	}

	if profileHandler.TypedName().Type == profile.SingleProfileHandlerType && len(profiles) > 1 {
		return nil, errors.New("SingleProfileHandler cannot support multiple scheduling profiles")
	}

	return scheduling.NewSchedulerConfig(profileHandler, profiles), nil
}

func loadFeatureConfig(gates configapi.FeatureGates) map[string]bool {
	config := make(map[string]bool, len(registeredFeatureGates))
	for gate := range registeredFeatureGates {
		config[gate] = false
	}
	for _, gate := range gates {
		config[gate] = true
	}
	return config
}

func buildSaturationConfig(apiConfig *configapi.SaturationDetector) *utilizationdetector.Config {
	cfg := &utilizationdetector.Config{
		QueueDepthThreshold:       utilizationdetector.DefaultQueueDepthThreshold,
		KVCacheUtilThreshold:      utilizationdetector.DefaultKVCacheUtilThreshold,
		MetricsStalenessThreshold: utilizationdetector.DefaultMetricsStalenessThreshold,
	}

	if apiConfig != nil {
		if apiConfig.QueueDepthThreshold > 0 {
			cfg.QueueDepthThreshold = apiConfig.QueueDepthThreshold
		}
		if apiConfig.KVCacheUtilThreshold > 0.0 && apiConfig.KVCacheUtilThreshold < 1.0 {
			cfg.KVCacheUtilThreshold = apiConfig.KVCacheUtilThreshold
		}
		if apiConfig.MetricsStalenessThreshold.Duration > 0 {
			cfg.MetricsStalenessThreshold = apiConfig.MetricsStalenessThreshold.Duration
		}
	}

	return cfg
}

func buildDataLayerConfig(rawDataConfig *configapi.DataLayerConfig, dataLayerEnabled bool, handle plugins.Handle) (*datalayer.Config, error) {
	if !dataLayerEnabled {
		if rawDataConfig != nil {
			return nil, errors.New("the Datalayer has not been enabled, but you specified a configuration for it")
		}
		return nil, nil
	}

	if rawDataConfig == nil {
		return nil, errors.New("the Datalayer has been enabled. You must specify the Data section in the configuration")
	}

	cfg := datalayer.Config{
		Sources: []datalayer.DataSourceConfig{},
	}
	for _, source := range rawDataConfig.Sources {
		if sourcePlugin, ok := handle.Plugin(source.PluginRef).(datalayer.DataSource); ok {
			sourceConfig := datalayer.DataSourceConfig{
				Plugin:     sourcePlugin,
				Extractors: []datalayer.Extractor{},
			}
			for _, extractor := range source.Extractors {
				if extractorPlugin, ok := handle.Plugin(extractor.PluginRef).(datalayer.Extractor); ok {
					sourceConfig.Extractors = append(sourceConfig.Extractors, extractorPlugin)
				} else {
					return nil, fmt.Errorf("the plugin %s is not a datalayer.Extractor", source.PluginRef)
				}
			}
			cfg.Sources = append(cfg.Sources, sourceConfig)
		} else {
			return nil, fmt.Errorf("the plugin %s is not a datalayer.Source", source.PluginRef)
		}
	}

	return &cfg, nil
}
