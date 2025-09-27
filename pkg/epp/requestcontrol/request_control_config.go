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
		preRequestPlugins:            []PreRequest{},
		postResponseRecievedPlugins:  []PostResponseRecieved{},
		postResponseStreamingPlugins: []PostResponseStreaming{},
		postResponseCompletePlugins:  []PostResponseComplete{},
	}
}

// Config provides a configuration for the requestcontrol plugins.
type Config struct {
	preRequestPlugins            []PreRequest
	postResponseRecievedPlugins  []PostResponseRecieved
	postResponseStreamingPlugins []PostResponseStreaming
	postResponseCompletePlugins  []PostResponseComplete
}

// WithPreRequestPlugins sets the given plugins as the PreRequest plugins.
// If the Config has PreRequest plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithPreRequestPlugins(plugins ...PreRequest) *Config {
	c.preRequestPlugins = plugins
	return c
}

// WithPostResponsePlugins sets the given plugins as the PostResponse plugins.
// If the Config has PostResponse plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithPostResponseRecievedPlugins(plugins ...PostResponseRecieved) *Config {
	c.postResponseRecievedPlugins = plugins
	return c
}

// WithPostResponseStreamingPlugins sets the given plugins as the PostResponseStreaming plugins.
// If the Config has PostResponseStreaming plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithPostResponseStreamingPlugins(plugins ...PostResponseStreaming) *Config {
	c.postResponseStreamingPlugins = plugins
	return c
}

// WithPostResponseCompletePlugins sets the given plugins as the PostResponseComplete plugins.
// If the Config has PostResponseComplete plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithPostResponseCompletePlugins(plugins ...PostResponseComplete) *Config {
	c.postResponseCompletePlugins = plugins
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
		if postResponseRecievedPlugin, ok := plugin.(PostResponseRecieved); ok {
			c.postResponseRecievedPlugins = append(c.postResponseRecievedPlugins, postResponseRecievedPlugin)
		}
		if postResponseStreamingPlugin, ok := plugin.(PostResponseStreaming); ok {
			c.postResponseStreamingPlugins = append(c.postResponseStreamingPlugins, postResponseStreamingPlugin)
		}
		if postResponseCompletePlugin, ok := plugin.(PostResponseComplete); ok {
			c.postResponseCompletePlugins = append(c.postResponseCompletePlugins, postResponseCompletePlugin)
		}
	}
}
