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
		admissionPlugins:         []fwk.AdmissionPlugin{},
		prepareDataPlugins:       []fwk.PrepareDataPlugin{},
		preRequestPlugins:        []fwk.PreRequest{},
		responseReceivedPlugins:  []fwk.ResponseReceived{},
		responseStreamingPlugins: []fwk.ResponseStreaming{},
		responseCompletePlugins:  []fwk.ResponseComplete{},
	}
}

// Config provides a configuration for the requestcontrol plugins.
type Config struct {
	admissionPlugins         []fwk.AdmissionPlugin
	prepareDataPlugins       []fwk.PrepareDataPlugin
	preRequestPlugins        []fwk.PreRequest
	responseReceivedPlugins  []fwk.ResponseReceived
	responseStreamingPlugins []fwk.ResponseStreaming
	responseCompletePlugins  []fwk.ResponseComplete
}

// WithPreRequestPlugins sets the given plugins as the PreRequest plugins.
// If the Config has PreRequest plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithPreRequestPlugins(plugins ...fwk.PreRequest) *Config {
	c.preRequestPlugins = plugins
	return c
}

// WithResponseReceivedPlugins sets the given plugins as the ResponseReceived plugins.
// If the Config has ResponseReceived plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithResponseReceivedPlugins(plugins ...fwk.ResponseReceived) *Config {
	c.responseReceivedPlugins = plugins
	return c
}

// WithResponseStreamingPlugins sets the given plugins as the ResponseStreaming plugins.
// If the Config has ResponseStreaming plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithResponseStreamingPlugins(plugins ...fwk.ResponseStreaming) *Config {
	c.responseStreamingPlugins = plugins
	return c
}

// WithResponseCompletePlugins sets the given plugins as the ResponseComplete plugins.
// If the Config has ResponseComplete plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithResponseCompletePlugins(plugins ...fwk.ResponseComplete) *Config {
	c.responseCompletePlugins = plugins
	return c
}

// WithPrepareDataPlugins sets the given plugins as the PrepareData plugins.
func (c *Config) WithPrepareDataPlugins(plugins ...fwk.PrepareDataPlugin) *Config {
	c.prepareDataPlugins = plugins
	return c
}

// WithAdmissionPlugins sets the given plugins as the AdmitRequest plugins.
func (c *Config) WithAdmissionPlugins(plugins ...fwk.AdmissionPlugin) *Config {
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
		if responseReceivedPlugin, ok := plugin.(fwk.ResponseReceived); ok {
			c.responseReceivedPlugins = append(c.responseReceivedPlugins, responseReceivedPlugin)
		}
		if responseStreamingPlugin, ok := plugin.(fwk.ResponseStreaming); ok {
			c.responseStreamingPlugins = append(c.responseStreamingPlugins, responseStreamingPlugin)
		}
		if responseCompletePlugin, ok := plugin.(fwk.ResponseComplete); ok {
			c.responseCompletePlugins = append(c.responseCompletePlugins, responseCompletePlugin)
		}
		if prepareDataPlugin, ok := plugin.(fwk.PrepareDataPlugin); ok {
			c.prepareDataPlugins = append(c.prepareDataPlugins, prepareDataPlugin)
		}
	}
}

// ProducerConsumerPlugins iterates through all registered plugins and returns two slices:
// one containing the names of plugins that implement the ProducerPlugin interface,
// and another for plugins that implement the ConsumerPlugin interface.
func (c *Config) ProducerConsumerPlugins() (map[string]plugin.ProducerPlugin, map[string]plugin.ConsumerPlugin) {
	producers := make(map[string]plugin.ProducerPlugin)
	consumers := make(map[string]plugin.ConsumerPlugin)
	// Collect all unique plugins from the config.
	allPlugins := make(map[string]plugin.Plugin)
	for _, p := range c.admissionPlugins {
		allPlugins[p.TypedName().String()] = p
	}
	for _, p := range c.prepareDataPlugins {
		allPlugins[p.TypedName().String()] = p
	}
	for _, p := range c.preRequestPlugins {
		allPlugins[p.TypedName().String()] = p
	}
	for _, p := range c.responseReceivedPlugins {
		allPlugins[p.TypedName().String()] = p
	}
	for _, p := range c.responseStreamingPlugins {
		allPlugins[p.TypedName().String()] = p
	}
	for _, p := range c.responseCompletePlugins {
		allPlugins[p.TypedName().String()] = p
	}

	for name, p := range allPlugins {
		if producer, ok := p.(plugin.ProducerPlugin); ok {
			producers[name] = producer
		}
		if consumer, ok := p.(plugin.ConsumerPlugin); ok {
			consumers[name] = consumer
		}
	}
	return producers, consumers
}

// ValidateDataDependencies creates a data dependency graph and sorts the plugins in topological order.
// If a cycle is detected, it returns an error.
func (c *Config) ValidateAndOrderDataDependencies() error {
	producers, consumers := c.ProducerConsumerPlugins()
	dag, err := buildDAG(producers, consumers)
	if err != nil {
		return err
	}
	plugins, err := sortPlugins(dag, c.prepareDataPlugins)
	if err != nil {
		return err
	}
	c.orderPrepareDataPlugins(plugins)

	return nil
}

// orderPrepareDataPlugins reorders the prepareDataPlugins in the Config based on the given sorted plugin names.
func (c *Config) orderPrepareDataPlugins(sortedPluginNames []string) {
	sortedPlugins := make([]fwk.PrepareDataPlugin, 0, len(sortedPluginNames))
	nameToPlugin := make(map[string]fwk.PrepareDataPlugin)
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
