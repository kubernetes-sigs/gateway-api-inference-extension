package loader

import (
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

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

func validateFeatureGates(fg configapi.FeatureGates) error {
	if fg == nil {
		return nil
	}

	for _, gate := range fg {
		if _, ok := registeredFeatureGates[gate]; !ok {
			return errors.New(gate + " is an unregistered Feature Gate")
		}
	}

	return nil
}
