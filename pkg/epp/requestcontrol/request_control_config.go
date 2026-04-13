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

package requestcontrol

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwk "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
)

// NewConfig creates a new Config object and returns its pointer.
func NewConfig() *Config {
	return &Config{
		admissionPlugins:         []fwk.Admitter{},
		prepareDataPlugins:       []fwk.DataProducer{},
		preRequestPlugins:        []fwk.PreRequest{},
		responseReceivedPlugins:  []fwk.ResponseHeaderProcessor{},
		responseStreamingPlugins: []fwk.ResponseBodyProcessor{},
	}
}

// Config provides a configuration for the requestcontrol plugins.
type Config struct {
	admissionPlugins         []fwk.Admitter
	prepareDataPlugins       []fwk.DataProducer
	preRequestPlugins        []fwk.PreRequest
	responseReceivedPlugins  []fwk.ResponseHeaderProcessor
	responseStreamingPlugins []fwk.ResponseBodyProcessor
}

// WithPreRequestPlugins sets the given plugins as the PreRequest plugins.
// If the Config has PreRequest plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithPreRequestPlugins(plugins ...fwk.PreRequest) *Config {
	c.preRequestPlugins = plugins
	return c
}

// WithResponseReceivedPlugins sets the given plugins as the ResponseReceived plugins.
// If the Config has ResponseReceived plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithResponseReceivedPlugins(plugins ...fwk.ResponseHeaderProcessor) *Config {
	c.responseReceivedPlugins = plugins
	return c
}

// WithResponseStreamingPlugins sets the given plugins as the ResponseStreaming plugins.
// If the Config has ResponseStreaming plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithResponseStreamingPlugins(plugins ...fwk.ResponseBodyProcessor) *Config {
	c.responseStreamingPlugins = plugins
	return c
}

// WithPrepareDataPlugins sets the given plugins as the PrepareData plugins.
func (c *Config) WithPrepareDataPlugins(plugins ...fwk.DataProducer) *Config {
	c.prepareDataPlugins = plugins
	return c
}

// WithAdmissionPlugins sets the given plugins as the AdmitRequest plugins.
func (c *Config) WithAdmissionPlugins(plugins ...fwk.Admitter) *Config {
	c.admissionPlugins = plugins
	return c
}

// AddPlugins adds the given plugins to the Config.
// The type of each plugin is checked and added to the corresponding list of plugins in the Config.
// If a plugin implements multiple plugin interfaces, it will be added to each corresponding list.
func (c *Config) AddPlugins(pluginObjects ...plugin.Plugin) {
	for _, plugin := range pluginObjects {
		if preRequestPlugin, ok := plugin.(fwk.PreRequest); ok {
			c.preRequestPlugins = append(c.preRequestPlugins, preRequestPlugin)
		}
		if responseReceivedPlugin, ok := plugin.(fwk.ResponseHeaderProcessor); ok {
			c.responseReceivedPlugins = append(c.responseReceivedPlugins, responseReceivedPlugin)
		}
		if responseStreamingPlugin, ok := plugin.(fwk.ResponseBodyProcessor); ok {
			c.responseStreamingPlugins = append(c.responseStreamingPlugins, responseStreamingPlugin)
		}
		if prepareDataPlugin, ok := plugin.(fwk.DataProducer); ok {
			c.prepareDataPlugins = append(c.prepareDataPlugins, prepareDataPlugin)
		}
		if admissionPlugin, ok := plugin.(fwk.Admitter); ok {
			c.admissionPlugins = append(c.admissionPlugins, admissionPlugin)
		}
	}
}

// OrderPrepareDataPlugins reorders the prepareDataPlugins in the Config based on the given sorted plugin names.
func (c *Config) OrderPrepareDataPlugins(sortedPluginNames []string) {
	sortedPlugins := make([]fwk.DataProducer, 0, len(sortedPluginNames))
	nameToPlugin := make(map[string]fwk.DataProducer)
	for _, plugin := range c.prepareDataPlugins {
		nameToPlugin[plugin.TypedName().String()] = plugin
	}
	for _, name := range sortedPluginNames {
		if plugin, ok := nameToPlugin[name]; ok {
			sortedPlugins = append(sortedPlugins, plugin)
		}
	}
	c.prepareDataPlugins = sortedPlugins
}
