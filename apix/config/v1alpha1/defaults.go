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

package v1alpha1

// SetDefaults_EndpointPickerConfig sets default values in a
// EndpointPickerConfig struct.
//
// This naming convension is required by the defalter-gen code.
func SetDefaults_EndpointPickerConfig(cfg *EndpointPickerConfig) {
	// If no name was given for the plugin, use it's type as the name
	for idx, pluginConfig := range cfg.Plugins {
		if pluginConfig.Name == "" {
			cfg.Plugins[idx].Name = pluginConfig.Type
		}
	}

	// If No SchedulerProfiles were specified in the confguration,
	// create one named default with references to all of the
	// plugins mentioned in the Plugins section of the configuration.
	if len(cfg.SchedulingProfiles) == 0 {
		cfg.SchedulingProfiles = make([]SchedulingProfile, 1)

		thePlugins := []SchedulingPlugin{}
		for _, pluginConfig := range cfg.Plugins {
			thePlugins = append(thePlugins, SchedulingPlugin{PluginRef: pluginConfig.Name})
		}

		cfg.SchedulingProfiles[0] = SchedulingProfile{
			Name:    "default",
			Plugins: thePlugins,
		}
	}
}
