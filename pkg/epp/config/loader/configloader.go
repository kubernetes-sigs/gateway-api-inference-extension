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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
)

var scheme = runtime.NewScheme()

func init() {
	utilruntime.Must(configapi.Install(scheme))
}

// Load config from supplied text that was converted to []byte
func LoadConfig(configBytes []byte, handle plugins.Handle, logger logr.Logger) (*config.Config, error) {
	rawConfig, err := loadRawConfig(configBytes)
	if err != nil {
		return nil, err
	}

	logger.Info("Loaded configuration", "config", rawConfig)

	setDefaultsPhaseOne(rawConfig)

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

func validateSchedulingProfiles(config *configapi.EndpointPickerConfig) error {
	profileNames := sets.New[string]()
	for _, profile := range config.SchedulingProfiles {
		if profile.Name == "" {
			return errors.New("SchedulingProfile must have a name")
		}

		if profileNames.Has(profile.Name) {
			return fmt.Errorf("the name '%s' has been specified for more than one SchedulingProfile", profile.Name)
		}
		profileNames.Insert(profile.Name)

		for _, plugin := range profile.Plugins {
			if len(plugin.PluginRef) == 0 {
				return fmt.Errorf("SchedulingProfile '%s' plugins must have a plugin reference", profile.Name)
			}

			notFound := true
			for _, pluginConfig := range config.Plugins {
				if plugin.PluginRef == pluginConfig.Name {
					notFound = false
					break
				}
			}
			if notFound {
				return errors.New(plugin.PluginRef + " is a reference to an undefined Plugin")
			}
		}
	}
	return nil
}
