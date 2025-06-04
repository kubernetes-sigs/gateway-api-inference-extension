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

package scheduling

import (
	"fmt"

	"github.com/go-logr/logr"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/common/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
)

// NewSchedulerConfig creates a new SchedulerConfig object and returns its pointer.
func NewSchedulerConfig(profilePicker framework.ProfilePicker, profiles map[string]*framework.SchedulerProfile) *SchedulerConfig {
	return &SchedulerConfig{
		profilePicker: profilePicker,
		profiles:      profiles,
	}
}

// SchedulerConfig provides a configuration for the scheduler which influence routing decisions.
type SchedulerConfig struct {
	profilePicker framework.ProfilePicker
	profiles      map[string]*framework.SchedulerProfile
}

func LoadSchedulerConfig(theConfig *config.Config,
	log logr.Logger) (*SchedulerConfig, error) {
	references, err := config.LoadPluginReferences(theConfig, log)
	if err != nil {
		return nil, err
	}

	var profiles = map[string]*framework.SchedulerProfile{}

	for _, configProfile := range theConfig.SchedulerProfiles {
		profile := framework.SchedulerProfile{}

		for _, plugin := range configProfile.Plugins {
			var thePlugin framework.Plugin
			var err error
			if plugin.Reference != nil {
				thePlugin = references[*plugin.Reference]
			} else {
				thePlugin, err = config.InstantiatePlugin(*plugin.Plugin, log)
				if err != nil {
					return nil, err
				}
			}
			if theScorer, ok := thePlugin.(framework.Scorer); ok {
				if plugin.Weight == nil {
					var name string
					if plugin.Reference != nil {
						name = *plugin.Reference
					} else {
						name = plugin.Plugin.PluginName
					}
					err = fmt.Errorf("scorer %s is missing a weight", name)
					log.Error(err, "failed to instantiate scheduler profile")
					return nil, err
				}
				thePlugin = framework.NewWeightedScorer(theScorer, *plugin.Weight)
			}
			err = profile.AddPlugins(thePlugin)
			fmt.Printf("\n\n==> %s \n\n", err)
			if err != nil {
				return nil, err
			}
		}
		profiles[configProfile.Name] = &profile
	}

	var thePlugin framework.Plugin
	var pluginName string

	if theConfig.ProfilePicker.Reference != nil {
		pluginName = *theConfig.ProfilePicker.Reference
		thePlugin = references[pluginName]
	} else {
		pluginName = theConfig.ProfilePicker.Plugin.PluginName
		thePlugin, err = config.InstantiatePlugin(*theConfig.ProfilePicker.Plugin, log)
		if err != nil {
			return nil, err
		}
	}
	if profilePicker, ok := thePlugin.(framework.ProfilePicker); ok {
		return NewSchedulerConfig(profilePicker, profiles), nil
	}
	return nil, fmt.Errorf("the plugin %s is not a ProfilePicker", pluginName)
}
