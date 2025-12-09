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
	"encoding/json"
	"fmt"
)

// FactoryFunc is the function signature for creating a new plugin instance.
//
// Arguments:
//   - name: The specific runtime name for this instance (e.g., "tenant-a-queue").
//   - parameters: The JSON configuration block from the blueprint.
//   - handle: The EPP handle for accessing global state/other plugins.
type FactoryFunc func(name string, parameters json.RawMessage, handle Handle) (Plugin, error)

// PluginLifecycle defines the instantiation policy for a plugin.
type PluginLifecycle int

const (
	// LifecycleSingleton indicates that a single, shared instance of the plugin should be created at startup.
	// This is the default lifecycle for all legacy plugins (Scorers, Global Controllers).
	LifecycleSingleton PluginLifecycle = iota

	// LifecycleTransient indicates that the plugin's configuration is a blueprint.
	// Independent instances are created at runtime via a PluginFactory.
	//
	// Usage: Stateful logic scoped to specific runtime entities, such as:
	//  - Inter-Flow Fairness Policies (scoped to a Priority Band)
	//  - Intra-Flow Ordering Policies (scoped to a specific Flow/Tenant)
	//  - Per-Flow Queues
	//
	// Note: Transient plugins are excluded from the global Request Control DAG.
	LifecycleTransient
)

// PluginRegistration holds the complete, self-describing metadata for a registered plugin type.
type PluginRegistration struct {
	Factory   FactoryFunc
	Lifecycle PluginLifecycle
}

// Registry is the central, global map of all known plugin types.
// It is populated via init() functions and is read-only during runtime.
var Registry = make(map[string]PluginRegistration)

// Register makes a plugin available to the system with the default Singleton lifecycle.
//
// This is preserved for backward compatibility. Plugins registered here are assumed
// to be singletons instantiated once at startup.
func Register(pluginImplType string, factory FactoryFunc) {
	RegisterWithMetadata(pluginImplType, PluginRegistration{
		Factory:   factory,
		Lifecycle: LifecycleSingleton,
	})
}

// RegisterWithMetadata makes a plugin available to the system with explicit lifecycle definitions.
//
// This function must only be called during package initialization (init()).
// Panics if a plugin with the same type is already registered.
func RegisterWithMetadata(pluginImplType string, reg PluginRegistration) {
	if _, exists := Registry[pluginImplType]; exists {
		panic(fmt.Sprintf("plugin type %q is already registered", pluginImplType))
	}
	Registry[pluginImplType] = reg
}

// ValidatePluginRef checks if a plugin reference is valid based on its type and lifecycle.
// It ensures that the configuration does not attempt to use a Transient plugin where a Singleton is expected.
func ValidatePluginRef(pluginType string, expectedLifecycle PluginLifecycle) error {
	reg, ok := Registry[pluginType]
	if !ok {
		return fmt.Errorf("plugin type %q not found in registry", pluginType)
	}

	if reg.Lifecycle != expectedLifecycle {
		return fmt.Errorf("plugin type %q has lifecycle %v, but expected %v", pluginType, reg.Lifecycle, expectedLifecycle)
	}

	return nil
}
