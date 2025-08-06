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
	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
)

const (
	// DefaultScorerWeight is the weight used for scorers referenced in the
	// configuration without explicit weights.
	DefaultScorerWeight = 1
)

// The code below sets the defaults in the configuration. It is done in two parts:
//   1) Before the plugins are instantiated, for those defaults that can be set
//      without knowing the concrete type of plugins
//   2) After the plugins are instantiated, for things that one needs to know the
//      concrete type of the plugins

var defaultScorerWeight = DefaultScorerWeight

// setDefaultsPhaseOne Performs the first phase of setting configuration defaults.
// In particuylar it:
//  1. Sets the name of plugins, for which one wasn't specified
func setDefaultsPhaseOne(cfg *configapi.EndpointPickerConfig) {
	// If no name was given for the plugin, use it's type as the name
	for idx, pluginConfig := range cfg.Plugins {
		if pluginConfig.Name == "" {
			cfg.Plugins[idx].Name = pluginConfig.Type
		}
	}
}

// setDefaultsPhaseTwo Performs the second phase of setting configuration defaults.
// In particular it:
//  1. Adds a default SchedulingProfile if one wasn't specified.
//  2. Adds an instance of the SingleProfileHandler, if no profile handler was
//     specified and the configuration has only one SchedulingProfile
//  3. Sets a default weight for all scorers without a weight
//  4. Adds a picker (MaxScorePicker) to all SchedulingProfiles that don't have a picker
func setDefaultsPhaseTwo(cfg *configapi.EndpointPickerConfig, handle plugins.Handle) {
	allPlugins := handle.GetAllPluginsWithNames()

	// If No SchedulerProfiles were specified in the confguration,
	// create one named default with references to all of the scheduling
	// plugins mentioned in the Plugins section of the configuration.
	if len(cfg.SchedulingProfiles) == 0 {
		cfg.SchedulingProfiles = make([]configapi.SchedulingProfile, 1)

		thePlugins := []configapi.SchedulingPlugin{}
		for pluginName, plugin := range allPlugins {
			switch plugin.(type) {
			case framework.Filter, framework.Picker, framework.PostCycle, framework.Scorer:
				thePlugins = append(thePlugins, configapi.SchedulingPlugin{PluginRef: pluginName})
			}
		}

		cfg.SchedulingProfiles[0] = configapi.SchedulingProfile{
			Name:    "default",
			Plugins: thePlugins,
		}
	}

	// Add an instance of the SingleProfileHandler, if no profile handler was
	// specified and the configuration has only one SchedulingProfile
	if len(cfg.SchedulingProfiles) == 1 {
		profileHandlerFound := false
		for _, plugin := range allPlugins {
			if _, ok := plugin.(framework.ProfileHandler); ok {
				profileHandlerFound = true
				break
			}
		}
		if !profileHandlerFound {
			handle.AddPlugin(profile.SingleProfileHandlerType, profile.NewSingleProfileHandler())
			cfg.Plugins = append(cfg.Plugins,
				configapi.PluginSpec{Name: profile.SingleProfileHandlerType,
					Type: profile.SingleProfileHandlerType,
				})
		}
	}

	var maxScorePicker string
	for pluginName, plugin := range allPlugins {
		if _, ok := plugin.(framework.Picker); ok {
			maxScorePicker = pluginName
			break
		}
	}
	if maxScorePicker == "" {
		handle.AddPlugin(picker.MaxScorePickerType, picker.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
		maxScorePicker = picker.MaxScorePickerType
		cfg.Plugins = append(cfg.Plugins, configapi.PluginSpec{Name: maxScorePicker, Type: maxScorePicker})
	}

	for idx, theProfile := range cfg.SchedulingProfiles {
		hasPicker := false
		for pluginIdx, plugin := range theProfile.Plugins {
			referencedPlugin := handle.Plugin(plugin.PluginRef)
			if _, ok := referencedPlugin.(framework.Scorer); ok && plugin.Weight == nil {
				theProfile.Plugins[pluginIdx].Weight = &defaultScorerWeight
				cfg.SchedulingProfiles[idx] = theProfile
			} else if _, ok := referencedPlugin.(framework.Picker); ok {
				hasPicker = true
				break
			}
		}
		if !hasPicker {
			theProfile.Plugins =
				append(theProfile.Plugins, configapi.SchedulingPlugin{PluginRef: maxScorePicker})
			cfg.SchedulingProfiles[idx] = theProfile
		}
	}
}
