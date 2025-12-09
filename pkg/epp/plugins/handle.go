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

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/types"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

// Handle provides plugins with access to global EPP state, configuration blueprints, and other singleton plugin
// instances.
type Handle interface {
	// Context returns the root context for the EPP.
	Context() context.Context

	// PluginSpec returns the raw configuration blueprint for the named plugin.
	// Returns nil if no such blueprint exists.
	PluginSpec(name string) *configapi.PluginSpec

	// HandlePlugins embeds access to Singleton plugin instances.
	HandlePlugins

	// PodList returns the current snapshot of ready backend pods.
	PodList() []types.NamespacedName
}

// HandlePlugins defines the API for accessing instantiated Singleton plugins.
type HandlePlugins interface {
	// Plugin returns the named plugin instance, or nil if not found.
	Plugin(name string) Plugin

	// AddPlugin adds a plugin to the set of known plugin instances.
	// Note: This operation modifies the internal map and is not thread-safe.
	// It should generally be used only during initialization or in single-threaded contexts.
	AddPlugin(name string, plugin Plugin)

	// GetAllPlugins returns a slice of all registered Singleton plugins.
	GetAllPlugins() []Plugin

	// GetAllPluginsWithNames returns a map of all registered Singleton plugins, keyed by TypedName().Name.
	GetAllPluginsWithNames() map[string]Plugin
}

// PodListFunc is a function type that returns a filtered list of pod names.
type PodListFunc func() []types.NamespacedName

// eppHandle implements the Handle interface.
// Concurrency Note: The pluginSpecs and plugins maps are populated at startup and are strictly read-only thereafter.
// No mutexes are required for read access.
type eppHandle struct {
	ctx         context.Context
	pluginSpecs map[string]*configapi.PluginSpec
	plugins     map[string]Plugin
	podList     PodListFunc
}

// NewEppHandle creates a new, immutable handle.
func NewEppHandle(
	ctx context.Context,
	specs []configapi.PluginSpec,
	plugins map[string]Plugin,
	podList PodListFunc,
) Handle {
	// Deep copy specs into a map for O(1) lookup.
	specMap := make(map[string]*configapi.PluginSpec, len(specs))
	for i := range specs {
		s := specs[i]
		specMap[s.Name] = &s
	}

	if plugins == nil {
		plugins = make(map[string]Plugin)
	}

	return &eppHandle{
		ctx:         ctx,
		pluginSpecs: specMap,
		plugins:     plugins,
		podList:     podList,
	}
}

// Context returns the root context associated with this handle.
func (h *eppHandle) Context() context.Context {
	return h.ctx
}

// PluginSpec retrieves the configuration blueprint for a given plugin name.
func (h *eppHandle) PluginSpec(name string) *configapi.PluginSpec {
	return h.pluginSpecs[name]
}

// Plugin retrieves a singleton plugin instance by name.
func (h *eppHandle) Plugin(name string) Plugin {
	return h.plugins[name]
}

// AddPlugin registers a new singleton plugin instance.
func (h *eppHandle) AddPlugin(name string, plugin Plugin) {
	h.plugins[name] = plugin
}

// GetAllPlugins returns a list of all registered singleton plugins.
func (h *eppHandle) GetAllPlugins() []Plugin {
	result := make([]Plugin, 0, len(h.plugins))
	for _, plugin := range h.plugins {
		result = append(result, plugin)
	}
	return result
}

// GetAllPluginsWithNames returns a map of all registered Singleton plugins, keyed by TypedName().Name.
func (h *eppHandle) GetAllPluginsWithNames() map[string]Plugin {
	return h.plugins
}

// PodList returns the list of currently ready pods from the underlying watcher.
func (h *eppHandle) PodList() []types.NamespacedName {
	if h.podList == nil {
		return nil
	}
	return h.podList()
}

// PluginByType retrieves a Singleton plugin by name and casts it to the expected type T.
func PluginByType[T Plugin](h HandlePlugins, name string) (T, error) {
	var zero T

	rawPlugin := h.Plugin(name)
	if rawPlugin == nil {
		return zero, fmt.Errorf("plugin %q not found", name)
	}

	plugin, ok := rawPlugin.(T)
	if !ok {
		return zero, fmt.Errorf("plugin %q is type %T, expected %T", name, rawPlugin, zero)
	}

	return plugin, nil
}
