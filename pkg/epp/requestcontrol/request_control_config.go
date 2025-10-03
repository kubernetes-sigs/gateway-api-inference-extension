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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
)

// NewConfig creates a new Config object and returns its pointer.
func NewConfig() *Config {
	return &Config{
		preRequestPlugins:        []PreRequest{},
		responseReceivedPlugins:  []ResponseReceived{},
		responseStreamingPlugins: []ResponseStreaming{},
		responseCompletePlugins:  []ResponseComplete{},
	}
}

// Config provides a configuration for the requestcontrol plugins.
type Config struct {
	preRequestPlugins        []PreRequest
	responseReceivedPlugins  []ResponseReceived
	responseStreamingPlugins []ResponseStreaming
	responseCompletePlugins  []ResponseComplete
}

// WithPreRequestPlugins sets the given plugins as the PreRequest plugins.
// If the Config has PreRequest plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithPreRequestPlugins(plugins ...PreRequest) *Config {
	c.preRequestPlugins = plugins
	return c
}

// WithResponseReceivedPlugins sets the given plugins as the Response plugins.
// If the Config has Response plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithResponseReceivedPlugins(plugins ...ResponseReceived) *Config {
	c.responseReceivedPlugins = plugins
	return c
}

// WithResponseStreamingPlugins sets the given plugins as the ResponseStreaming plugins.
// If the Config has ResponseStreaming plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithResponseStreamingPlugins(plugins ...ResponseStreaming) *Config {
	c.responseStreamingPlugins = plugins
	return c
}

// WithResponseCompletePlugins sets the given plugins as the ResponseComplete plugins.
// If the Config has ResponseComplete plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithResponseCompletePlugins(plugins ...ResponseComplete) *Config {
	c.responseCompletePlugins = plugins
	return c
}

// AddPlugins adds the given plugins to the Config.
// The type of each plugin is checked and added to the corresponding list of plugins in the Config.
// If a plugin implements multiple plugin interfaces, it will be added to each corresponding list.

func (c *Config) AddPlugins(pluginObjects ...plugins.Plugin) {
	for _, plugin := range pluginObjects {
		if preRequestPlugin, ok := plugin.(PreRequest); ok {
			c.preRequestPlugins = append(c.preRequestPlugins, preRequestPlugin)
		}
		if responseReceivedPlugin, ok := plugin.(ResponseReceived); ok {
			c.responseReceivedPlugins = append(c.responseReceivedPlugins, responseReceivedPlugin)
		}
		if responseStreamingPlugin, ok := plugin.(ResponseStreaming); ok {
			c.responseStreamingPlugins = append(c.responseStreamingPlugins, responseStreamingPlugin)
		}
		if responseCompletePlugin, ok := plugin.(ResponseComplete); ok {
			c.responseCompletePlugins = append(c.responseCompletePlugins, responseCompletePlugin)
		}
	}
}
