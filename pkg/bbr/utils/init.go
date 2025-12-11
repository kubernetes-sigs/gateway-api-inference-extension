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

// Initializes PluginRegistry from environment variables (as set in the helm chart)

package utils

import (
	"fmt"
	"os"
	"regexp"
	"strings"

	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	bbrplugins "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins"
)

func InitPlugins() (
	framework.PluginRegistry,
	framework.PluginsChain,
	framework.PluginsChain,
	error) {

	//The environment variables defining plugins repertoire and plugin chains are set via Helm chart
	registry := framework.NewPluginRegistry()
	requestChain := framework.NewPluginsChain()
	responseChain := framework.NewPluginsChain()

	//Define a standardized regex patterns for plugin types
	var pluginTypeRe = regexp.MustCompile(bbrplugins.PluginTypePattern)

	//Define a standardized regex pattern for plugin names
	var pluginNameRe = regexp.MustCompile(bbrplugins.PluginNamePattern)

	//helper to validate plugin name
	isValidPluginName := func(name string) bool {
		if len(name) == 0 || len(name) > bbrplugins.MaxPluginNameLength {
			return false
		}
		return pluginNameRe.MatchString(name)
	}

	//helper to validate plugin type name
	isValidPluginType := func(name string) bool {
		if len(name) == 0 || len(name) > bbrplugins.MaxPluginTypeLength {
			return false
		}
		return pluginTypeRe.MatchString(name)
	}

	//helper to process plugins
	processPlugin := func(pluginType string, chain framework.PluginsChain) error {

		//create the plugin instance
		plugin, err := registry.CreatePlugin(pluginType)
		if err != nil {
			return fmt.Errorf("failed to create an instance of %s %v", pluginType, err)
		}

		//register the plugin instance
		err = registry.RegisterPlugin(plugin)
		if err != nil {
			return fmt.Errorf("failed to register an instance of %s %v", pluginType, err)
		}

		//Add plugin type name to the pluginsChain instance
		err = chain.AddPlugin(pluginType, registry)
		if err != nil {
			return fmt.Errorf("failed to add plugin instance %s %v", pluginType, err)
		}
		return nil
	}

	// Helper to process plugin chains
	processChain := func(envVar string, chain framework.PluginsChain) error {
		envPluginsChain := os.Getenv(envVar)

		if envPluginsChain == "" {
			return nil // no plugins defined for this chain, but this is not an error
		}

		//Plugins are specified as, e.g., REQUEST_PLUGINS_CHAIN=MetaDataExtractor:simple-model-extractor, MyPluginType:my-plugin-name
		parts := strings.Split(envPluginsChain, ",")

		for i, part := range parts {
			typedname := strings.TrimSpace(part)

			subparts := strings.Split(typedname, ":")
			pluginType := subparts[0]
			pluginName := subparts[1]

			//validate plugin type naming rules
			if !isValidPluginType(pluginType) {
				return fmt.Errorf("plugin %d: invalid type %s", i, pluginType)
			}

			//validate plugin naming rules
			if !isValidPluginName(pluginName) {
				return fmt.Errorf("plugin %d: invalid type %s", i, pluginType)
			}

			//process this plugin: create an instance, register it in a registry , and add to plugin chain by type name
			if err := processPlugin(pluginType, chain); err != nil {
				return fmt.Errorf("failed to install plugin %d:  %s", i, pluginType)
			}
		}
		return nil
	}

	// Pre-register all BBRPlugin factories factories
	if err := RegisterAllFactories(registry); err != nil {
		return nil, nil, nil, err
	}

	// Process request plugins chain
	if err := processChain(framework.RequestPluginChain, requestChain); err != nil { //requestPlugins chain need not be explicitly specified if the only pluginn is the default one
		return nil, nil, nil, err
	}

	// Process response plugins chain
	if err := processChain(framework.ResponsePluginChain, responseChain); err != nil { //responsePluginsChain is currently left empty
		return nil, nil, nil, err
	}

	// If request chain is empty (i.e., it was not explicitly specified in the env via Helm), add default MetadataExtractor
	if requestChain.Length() == 0 {
		//use default plugin
		if err := processPlugin(bbrplugins.DefaultPluginType, requestChain); err != nil {
			return nil, nil, nil, fmt.Errorf("failed to create default MetaDataExtractor: %v", err)
		}
	}

	return registry, requestChain, responseChain, nil
}
