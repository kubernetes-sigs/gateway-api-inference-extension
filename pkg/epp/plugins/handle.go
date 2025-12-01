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
)

// Handle provides plugins a set of standard data and tools to work with
type Handle interface {
	// Context returns a context the plugins can use, if they need one
	Context() context.Context

	HandlePlugins

	// PodList lists pods.
	PodList() []types.NamespacedName
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

// PodListFunc is a function type that filters and returns a list of pod metrics
type PodListFunc func() []types.NamespacedName

// eppHandle is an implementation of the interface plugins.Handle
type eppHandle struct {
	ctx context.Context
	HandlePlugins
	podList PodListFunc
}

// Context returns a context the plugins can use, if they need one
func (h *eppHandle) Context() context.Context {
	return h.ctx
}

// eppHandlePlugins implements the set of APIs to work with instantiated plugins
type eppHandlePlugins struct {
	plugins map[string]Plugin
}

// Plugin returns the named plugin instance
func (h *eppHandlePlugins) Plugin(name string) Plugin {
	return h.plugins[name]
}

// AddPlugin adds a plugin to the set of known plugin instances
func (h *eppHandlePlugins) AddPlugin(name string, plugin Plugin) {
	h.plugins[name] = plugin
}

// GetAllPlugins returns all of the known plugins
func (h *eppHandlePlugins) GetAllPlugins() []Plugin {
	result := make([]Plugin, 0)
	for _, plugin := range h.plugins {
		result = append(result, plugin)
	}
	return result
}

// GetAllPluginsWithNames returns al of the known plugins with their names
func (h *eppHandlePlugins) GetAllPluginsWithNames() map[string]Plugin {
	return h.plugins
}

// PodList lists pods.
func (h *eppHandle) PodList() []types.NamespacedName {
	return h.podList()
}

func NewEppHandle(ctx context.Context, podList PodListFunc) Handle {
	return &eppHandle{
		ctx: ctx,
		HandlePlugins: &eppHandlePlugins{
			plugins: map[string]Plugin{},
		},
		podList: podList,
	}
}

// PluginByType retrieves the specified plugin by name and verifies its type
func PluginByType[P Plugin](handlePlugins HandlePlugins, name string) (P, error) {
	var zero P

	rawPlugin := handlePlugins.Plugin(name)
	if rawPlugin == nil {
		return zero, fmt.Errorf("there is no plugin with the name '%s' defined", name)
	}
	plugin, ok := rawPlugin.(P)
	if !ok {
		return zero, fmt.Errorf("the plugin with the name '%s' is not an instance of %T", name, zero)
	}
	return plugin, nil
}
