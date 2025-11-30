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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/saturationdetector"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
)

var scheme = runtime.NewScheme()

var registeredFeatureGates = sets.New[string]() // set of feature gates names, a name must be unique

func init() {
	utilruntime.Must(configapi.Install(scheme))
}

// LoadConfigPhaseOne first phase of loading configuration from supplied text that was converted to []byte
func LoadConfigPhaseOne(configBytes []byte, logger logr.Logger) (*configapi.EndpointPickerConfig, map[string]bool, error) {
	rawConfig, err := loadRawConfig(configBytes)
	if err != nil {
		return nil, nil, err
	}

	logger.Info("Loaded configuration", "config", rawConfig)

	if err = validateFeatureGates(rawConfig.FeatureGates); err != nil {
		return nil, nil, fmt.Errorf("failed to validate feature gates - %w", err)
	}

	setDefaultsPhaseOne(rawConfig)

	featureConfig := loadFeatureConfig(rawConfig.FeatureGates)

	return rawConfig, featureConfig, nil
}

// LoadConfigPhaseOne first phase of loading configuration from supplied text that was converted to []byte
func LoadConfigPhaseTwo(rawConfig *configapi.EndpointPickerConfig, handle plugins.Handle, logger logr.Logger) (*config.Config, error) {
	var err error
	// instantiate loaded plugins
	if err = instantiatePlugins(rawConfig.Plugins, handle); err != nil {
		return nil, fmt.Errorf("failed to instantiate plugins - %w", err)
	}

	setDefaultsPhaseTwo(rawConfig, handle)

	logger.Info("Configuration with defaults set", "config", rawConfig)

	if err = validateSchedulingProfiles(rawConfig); err != nil {
		return nil, fmt.Errorf("failed to validate scheduling profiles - %w", err)
	}

	config := &config.Config{}

	config.SchedulerConfig, err = loadSchedulerConfig(rawConfig.SchedulingProfiles, handle)
	if err != nil {
		return nil, err
	}
	config.SaturationDetectorConfig = loadSaturationDetectorConfig(rawConfig.SaturationDetector)

	return config, nil
}

func loadRawConfig(configBytes []byte) (*configapi.EndpointPickerConfig, error) {
	rawConfig := &configapi.EndpointPickerConfig{}

	codecs := serializer.NewCodecFactory(scheme, serializer.EnableStrict)
	err := runtime.DecodeInto(codecs.UniversalDecoder(), configBytes, rawConfig)
	if err != nil {
		return nil, fmt.Errorf("the configuration is invalid - %w", err)
	}
	return rawConfig, nil
}

func loadSchedulerConfig(configProfiles []configapi.SchedulingProfile, handle plugins.Handle) (*scheduling.SchedulerConfig, error) {
	profiles := map[string]*framework.SchedulerProfile{}
	for _, namedProfile := range configProfiles {
		profile := framework.NewSchedulerProfile()
		for _, plugin := range namedProfile.Plugins {
			referencedPlugin := handle.Plugin(plugin.PluginRef)
			if scorer, ok := referencedPlugin.(framework.Scorer); ok {
				referencedPlugin = framework.NewWeightedScorer(scorer, *plugin.Weight)
			}
			if err := profile.AddPlugins(referencedPlugin); err != nil {
				return nil, fmt.Errorf("failed to load scheduler config - %w", err)
			}
		}
		profiles[namedProfile.Name] = profile
	}

	var profileHandler framework.ProfileHandler
	for pluginName, plugin := range handle.GetAllPluginsWithNames() {
		if theProfileHandler, ok := plugin.(framework.ProfileHandler); ok {
			if profileHandler != nil {
				return nil, fmt.Errorf("only one profile handler is allowed. Both %s and %s are profile handlers", profileHandler.TypedName().Name, pluginName)
			}
			profileHandler = theProfileHandler
		}
	}
	if profileHandler == nil {
		return nil, errors.New("no profile handler was specified")
	}

	if profileHandler.TypedName().Type == profile.SingleProfileHandlerType && len(profiles) > 1 {
		return nil, errors.New("single profile handler is intended to be used with a single profile, but multiple profiles were specified")
	}

	return scheduling.NewSchedulerConfig(profileHandler, profiles), nil
}

func loadFeatureConfig(featureGates configapi.FeatureGates) map[string]bool {
	featureConfig := map[string]bool{}

	for gate := range registeredFeatureGates {
		featureConfig[gate] = false
	}

	for _, gate := range featureGates {
		featureConfig[gate] = true
	}

	return featureConfig
}

func loadSaturationDetectorConfig(sd *configapi.SaturationDetector) *saturationdetector.Config {
	sdConfig := saturationdetector.Config{}

	sdConfig.QueueDepthThreshold = sd.QueueDepthThreshold
	if sdConfig.QueueDepthThreshold <= 0 {
		sdConfig.QueueDepthThreshold = saturationdetector.DefaultQueueDepthThreshold
	}
	sdConfig.KVCacheUtilThreshold = sd.KVCacheUtilThreshold
	if sdConfig.KVCacheUtilThreshold <= 0.0 || sdConfig.KVCacheUtilThreshold >= 1.0 {
		sdConfig.KVCacheUtilThreshold = saturationdetector.DefaultKVCacheUtilThreshold
	}
	sdConfig.MetricsStalenessThreshold = sd.MetricsStalenessThreshold.Duration
	if sdConfig.MetricsStalenessThreshold <= 0.0 {
		sdConfig.MetricsStalenessThreshold = saturationdetector.DefaultMetricsStalenessThreshold
	}

	return &sdConfig
}

func instantiatePlugins(configuredPlugins []configapi.PluginSpec, handle plugins.Handle) error {
	pluginNames := sets.New[string]() // set of plugin names, a name must be unique

	for _, pluginConfig := range configuredPlugins {
		if pluginConfig.Type == "" {
			return fmt.Errorf("plugin definition for '%s' is missing a type", pluginConfig.Name)
		}

		if pluginNames.Has(pluginConfig.Name) {
			return fmt.Errorf("plugin name '%s' used more than once", pluginConfig.Name)
		}
		pluginNames.Insert(pluginConfig.Name)

		factory, ok := plugins.Registry[pluginConfig.Type]
		if !ok {
			return fmt.Errorf("plugin type '%s' is not found in registry", pluginConfig.Type)
		}

		plugin, err := factory(pluginConfig.Name, pluginConfig.Parameters, handle)
		if err != nil {
			return fmt.Errorf("failed to instantiate the plugin type '%s' - %w", pluginConfig.Type, err)
		}

		handle.AddPlugin(pluginConfig.Name, plugin)
	}

	return nil
}

// RegisterFeatureGate registers feature gate keys for validation
func RegisterFeatureGate(gate string) {
	registeredFeatureGates.Insert(gate)
}
