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

package plugins

import "context"

// Plugin defines the interface for a plugin.
// This interface should be embedded in all plugins across the code.
type Plugin interface {
	// Type returns the type of the plugin.
	Type() string
	// Name returns the name of this plugin instance.
	Name() string
}

// Handle provides plugins a set of standard data and tools to work with
type Handle interface {
	// Context returns a context the plugins can use, if they need one
	Context() context.Context

	// Plugins returns the sub-handle for working with instantiated plugins
	Plugins() HandlePlugins
}

// HandlePlugins defines a set of APIs to work with instantiated plugins
type HandlePlugins interface {
	// Plugin returns the named plugin instance
	Plugin(name string) Plugin

	// AddPlugin adds a plugin to the set of known plugin instances
	AddPlugin(name string, plugin Plugin)

	// GetAllPlugins returns all of the known plugins
	GetAllPlugins() []Plugin

	// GetAllPluginsWithNames returns all of the known plugins with their names
	GetAllPluginsWithNames() map[string]Plugin
}
