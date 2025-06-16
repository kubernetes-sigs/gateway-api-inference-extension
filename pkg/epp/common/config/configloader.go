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
	"errors"
	"fmt"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"

	configapi "sigs.k8s.io/gateway-api-inference-extension/api/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
)

var scheme = runtime.NewScheme()

func init() {
	configapi.SchemeBuilder.Register(configapi.RegisterDefaults)
	utilruntime.Must(configapi.Install(scheme))
}

// Load config either from supplied text or from a file
func LoadConfig(configText []byte, fileName string) (*configapi.EndpointPickerConfig, error) {
	var err error
	if len(configText) == 0 {
		configText, err = os.ReadFile(fileName)
		if err != nil {
			return nil, fmt.Errorf("failed to load config file. Error: %s", err)
		}
	}

	theConfig := &configapi.EndpointPickerConfig{}

	codecs := serializer.NewCodecFactory(scheme, serializer.EnableStrict)
	err = runtime.DecodeInto(codecs.UniversalDecoder(), configText, theConfig)
	if err != nil {
		return nil, fmt.Errorf("the configuration is invalid. Error: %s", err)
	}

	// Validate loaded configuration
	err = validateConfiguration(theConfig)
	if err != nil {
		return nil, fmt.Errorf("the configuration is invalid. error: %s", err)
	}
	return theConfig, nil
}

func LoadPluginReferences(thePlugins []configapi.PluginSpec, handle plugins.Handle) (map[string]plugins.Plugin, error) {
	references := map[string]plugins.Plugin{}
	for _, pluginConfig := range thePlugins {
		thePlugin, err := InstantiatePlugin(pluginConfig, handle)
		if err != nil {
			return nil, err
		}
		references[pluginConfig.Name] = thePlugin
	}
	return references, nil
}

func InstantiatePlugin(pluginSpec configapi.PluginSpec, handle plugins.Handle) (plugins.Plugin, error) {
	factory, ok := plugins.Registry[pluginSpec.PluginName]
	if !ok {
		return nil, fmt.Errorf("failed to instantiate the plugin. plugin %s not found", pluginSpec.PluginName)
	}
	thePlugin, err := factory(pluginSpec.Name, pluginSpec.Parameters, handle)
	if err != nil {
		return nil, fmt.Errorf("failed to instantiate the plugin %s. Error: %s", pluginSpec.PluginName, err)
	}
	return thePlugin, err
}

func validateConfiguration(theConfig *configapi.EndpointPickerConfig) error {
	names := make(map[string]bool)

	for _, pluginConfig := range theConfig.Plugins {
		if pluginConfig.PluginName == "" {
			return errors.New("plugin reference definition missing a plugin name")
		}

		if _, ok := names[pluginConfig.Name]; ok {
			return fmt.Errorf("the name %s has been specified for more than one plugin", pluginConfig.Name)
		}
		names[pluginConfig.Name] = true

		_, ok := plugins.Registry[pluginConfig.PluginName]
		if !ok {
			return fmt.Errorf("plugin %s is not found", pluginConfig.PluginName)
		}
	}

	if len(theConfig.SchedulingProfiles) == 0 {
		return errors.New("there must be at least one scheduling profile in the configuration")
	}

	names = map[string]bool{}
	for _, profile := range theConfig.SchedulingProfiles {
		if profile.Name == "" {
			return errors.New("SchedulingProfiles need a name")
		}

		if _, ok := names[profile.Name]; ok {
			return fmt.Errorf("the name %s has been specified for more than one SchedulingProfile", profile.Name)
		}
		names[profile.Name] = true

		if len(profile.Plugins) == 0 {
			return errors.New("SchedulingProfiles need at least one plugin")
		}
		for _, plugin := range profile.Plugins {
			if len(plugin.PluginRef) == 0 {
				return errors.New("SchedulingProfile's plugins need a plugin reference")
			}

			notFound := true
			for _, pluginConfig := range theConfig.Plugins {
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
