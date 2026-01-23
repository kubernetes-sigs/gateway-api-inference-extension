/*
Copyright 2026 The Kubernetes Authors.

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

package plugins

import "encoding/json"

// Any no-argument function that returns BBRPlugin can be assigned to this type (including a concrete factory function)
type PluginFactoryFunc func(name string, parameters json.RawMessage) (BBRPlugin, error)

// Registry is a simple mapping from the plugin name to the factory that creates this BBRPlugin
var Registry = map[string]PluginFactoryFunc{}

// Registers a plugin's factory
func Register(pluginType string, factory PluginFactoryFunc) {
	Registry[pluginType] = factory
}
