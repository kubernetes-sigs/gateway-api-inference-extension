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
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/go-logr/logr"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/yaml"
)

type Config struct {
	PluginDefinitions []ConfigPluginDefinition `json:"plugin_definitions"`
	ProfilePicker     BaseConfigProfilePlugin  `json:"profile_picker"`
	SchedulerProfiles []ConfigSchedulerProfile `json:"scheduler_profiles"`
}

type BaseConfigPluginDefinition struct {
	PluginName string          `json:"plugin_name"`
	Parameters json.RawMessage `json:"parameters"`
}

type ConfigPluginDefinition struct {
	BaseConfigPluginDefinition `json:",inline"`
	Name                       string `json:"name"`
}

type ConfigSchedulerProfile struct {
	Name    string                `json:"name"`
	Plugins []ConfigProfilePlugin `json:"plugins"`
}

type BaseConfigProfilePlugin struct {
	Reference *string                     `json:"reference"`
	Plugin    *BaseConfigPluginDefinition `json:"plugin"`
}

type ConfigProfilePlugin struct {
	BaseConfigProfilePlugin `json:",inline"`
	Weight                  *int `json:"weight"`
}

// Load config either from supplied text or from a file
func LoadConfig(configText []byte, fileName string, log logr.Logger) (*Config, error) {
	var err error
	if len(configText) == 0 {
		configText, err = os.ReadFile(fileName)
		if err != nil {
			log.Error(err, "failed to load config file")
			return nil, err
		}
	}

	theConfig := &Config{}
	err = yaml.Unmarshal(configText, theConfig)
	if err != nil {
		log.Error(err, "failed to parse the configuration")
		return nil, err
	}

	// Validate loaded configuration
	err = validateConfiguration(theConfig)
	if err != nil {
		log.Error(err, "the configuration is invalid")
		return nil, err
	}
	return theConfig, nil
}

func LoadPluginReferences(theConfig *Config, log logr.Logger) (map[string]framework.Plugin, error) {
	references := map[string]framework.Plugin{}
	for _, pluginReference := range theConfig.PluginDefinitions {
		thePlugin, err := InstantiatePlugin(pluginReference.BaseConfigPluginDefinition, log)
		if err != nil {
			return nil, err
		}
		references[pluginReference.Name] = thePlugin
	}
	return references, nil
}

func InstantiatePlugin(pluginDefinition BaseConfigPluginDefinition, log logr.Logger) (framework.Plugin, error) {
	factory, ok := framework.Registry[pluginDefinition.PluginName]
	if !ok {
		err := fmt.Errorf("plugin %s not found", pluginDefinition.PluginName)
		log.Error(err, "failed to instantiate plugin")
		return nil, err
	}
	thePlugin, err := factory(pluginDefinition.Parameters)
	if err != nil {
		log.Error(err, "failed to instantiate the plugin", "plugin", pluginDefinition.PluginName)
		return nil, err
	}
	return thePlugin, err
}

func validateConfiguration(theConfig *Config) error {
	for _, pluginDefinition := range theConfig.PluginDefinitions {
		if pluginDefinition.Name == "" || pluginDefinition.PluginName == "" {
			return errors.New("plugin reference definition missing name or plugin reference")
		}
		_, ok := framework.Registry[pluginDefinition.PluginName]
		if !ok {
			return fmt.Errorf("plugin %s is not found", pluginDefinition.PluginName)
		}
	}

	if len(theConfig.SchedulerProfiles) == 0 {
		return errors.New("there must be at least one scheduling profile in the configuration")
	}

	if theConfig.ProfilePicker.Reference == nil && theConfig.ProfilePicker.Plugin == nil {
		return errors.New("ProfilePicker needs either a plugin reference or definition")
	}

	for _, profile := range theConfig.SchedulerProfiles {
		if profile.Name == "" {
			return errors.New("SchedulerProfiles need a name")
		}
		if len(profile.Plugins) == 0 {
			return errors.New("SchedulingProfiles need at leas one plugin")
		}
		for _, plugin := range profile.Plugins {
			if plugin.Reference == nil && plugin.Plugin == nil {
				return errors.New("SchedulingProfile's plugins need either a plugin reference or definition")
			}
			if plugin.Reference != nil {
				notFound := true
				for _, reference := range theConfig.PluginDefinitions {
					if *plugin.Reference == reference.Name {
						notFound = false
						break
					}
				}
				if notFound {
					return errors.New(*plugin.Reference + " is a reference to an undefined PluginDefinition")
				}
			} else {
				_, ok := framework.Registry[plugin.Plugin.PluginName]
				if !ok {
					return fmt.Errorf("plugin %s is not found", plugin.Plugin.PluginName)
				}
			}
		}

	}
	return nil
}
