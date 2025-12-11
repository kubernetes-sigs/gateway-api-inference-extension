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

package framework

import (
	"fmt"

	bbrplugins "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins"
)

// -------------------- INTERFACES -----------------------------------------------------------------------
// Interfaces are defined in "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework/interfaces.go"

// --------------------- PluginRegistry implementation ---------------------------------------------------

// pluginRegistry implements PluginRegistry
type pluginRegistry struct {
	pluginsFactory map[string]PluginFactoryFunc    //constructors
	plugins        map[string]bbrplugins.BBRPlugin // instances
}

// NewPluginRegistry creates a new instance of pluginRegistry
func NewPluginRegistry() PluginRegistry {
	return &pluginRegistry{
		pluginsFactory: make(map[string]PluginFactoryFunc),
		plugins:        make(map[string]bbrplugins.BBRPlugin),
	}
}

// Register a plugin factory by type key (e.g., "ModelSelector", "MetadataExtractor")
func (r *pluginRegistry) RegisterFactory(typeKey string, factory PluginFactoryFunc) error {
	//validate whether already registered
	alreadyRegistered := r.ContainsFactory(typeKey)
	if alreadyRegistered {
		err := fmt.Errorf("factory fot plugin interface type %s is already registered", typeKey)
		return err
	}
	r.pluginsFactory[typeKey] = factory

	return nil
}

// Register a plugin instance (created through the appropriate factory)
func (r *pluginRegistry) RegisterPlugin(plugin bbrplugins.BBRPlugin) error {
	//validate whether this interface is supported
	alreadyRegistered := r.ContainsPlugin(plugin.TypedName().Type)

	if alreadyRegistered {
		err := fmt.Errorf("plugin implementing interface type %s is already registered", plugin.TypedName().Type)
		return err
	}

	// validate that the factory for this plugin is registered: always register factory before the plugin
	if _, ok := r.pluginsFactory[plugin.TypedName().Type]; !ok {
		err := fmt.Errorf("no plugin factory registered for plugin interface type %s", plugin.TypedName().Type)
		return err
	}
	r.plugins[plugin.TypedName().Type] = plugin

	return nil
}

// Retrieves a plugin factory by type key
func (r *pluginRegistry) GetFactory(typeKey string) (PluginFactoryFunc, error) {
	if pluginFactory, ok := r.pluginsFactory[typeKey]; ok {
		return pluginFactory, nil
	}
	return nil, fmt.Errorf("plugin type %s not found", typeKey)
}

// Retrieves a plugin instance by type key
func (r *pluginRegistry) GetPlugin(typeKey string) (bbrplugins.BBRPlugin, error) {
	if plugin, ok := r.plugins[typeKey]; ok {
		return plugin, nil
	}
	return nil, fmt.Errorf("plugin type %s not found", typeKey)
}

// Constructs a new plugin (a caller can perform either type  assertion of a concrete implementation of the BBR plugin)
func (r *pluginRegistry) CreatePlugin(typeKey string) (bbrplugins.BBRPlugin, error) {
	if factory, ok := r.pluginsFactory[typeKey]; ok {
		plugin := factory()
		return plugin, nil
	}
	return nil, fmt.Errorf("plugin %s not registered", typeKey)
}

// Removes a plugin factory by type key
func (r *pluginRegistry) UnregisterFactory(typeKey string) error {
	if _, ok := r.pluginsFactory[typeKey]; ok {
		delete(r.pluginsFactory, typeKey)
		return nil
	}
	return fmt.Errorf("plugin (%s) not found", typeKey)
}

// ListPlugins lists all registered plugins
func (r *pluginRegistry) ListPlugins() []string {
	typeKeys := make([]string, 0, len(r.plugins))
	for k := range r.plugins {
		typeKeys = append(typeKeys, k)
	}
	return typeKeys
}

// ListPlugins lists all registered plugins; this functionis not really needed. Just for sanity checks and tests
func (r *pluginRegistry) ListFactories() []string {
	typeKeys := make([]string, 0, len(r.pluginsFactory))
	for k := range r.pluginsFactory {
		typeKeys = append(typeKeys, k)
	}
	return typeKeys
}

// Get factories
func (r *pluginRegistry) GetFactories() map[string]PluginFactoryFunc {
	return r.pluginsFactory
}

// Get plugins
func (r *pluginRegistry) GetPlugins() map[string]bbrplugins.BBRPlugin {
	return r.plugins
}

// Checks for presense of a factory in this registry
func (r *pluginRegistry) ContainsFactory(typeKey string) bool {
	_, exists := r.pluginsFactory[typeKey]
	return exists
}

// Helper: Checks for presense of a plugin in this registry
func (r *pluginRegistry) ContainsPlugin(typeKey string) bool {
	_, exists := r.plugins[typeKey]
	return exists
}

func (r *pluginRegistry) String() string {
	return fmt.Sprintf("{plugins=%v}{pluginsFactory=%v}", r.plugins, r.pluginsFactory)
}

//-------------------------- PluginsChain implementation --------------------------

// PluginsChain is a sequence of plugins to be executed in order inside the ext_proc server
type pluginsChain struct {
	plugins []string
}

// NewPluginsChain creates a new PluginsChain instance
func NewPluginsChain() PluginsChain {
	return &pluginsChain{
		plugins: []string{},
	}
}

// AddPlugin adds a plugin to the chain
func (pc *pluginsChain) AddPlugin(typeKey string, r PluginRegistry) error {
	// check whether this plugin was registered in the registry (i.e., the factory for the plugin exist and an instance was created)
	if ok := r.ContainsPlugin(typeKey); !ok {
		err := fmt.Errorf("plugin type %s not found", typeKey)
		return err
	}
	pc.plugins = append(pc.plugins, typeKey)

	return nil
}

// GetPlugin retrieves the next plugin in the chain by index
func (pc *pluginsChain) GetPlugin(index int, r PluginRegistry) (bbrplugins.BBRPlugin, error) {
	if index < 0 || index >= len(pc.plugins) {
		return nil, fmt.Errorf("plugin index %d out of range", index)
	}
	plugins := r.GetPlugins()
	plugin, ok := plugins[pc.plugins[index]]
	if !ok {
		return nil, fmt.Errorf("plugin index %d is not found in the registry", index)
	}
	return plugin, nil
}

// Length returns the number of plugins in the chain
func (pc *pluginsChain) Length() int {
	return len(pc.plugins)
}

// AddPluginInOrder inserts a plugin into the chain in the specified index
func (pc *pluginsChain) AddPluginAtInd(typeKey string, i int, r PluginRegistry) error {
	if i < 0 || i > len(pc.plugins) {
		return fmt.Errorf("index %d is out of range", i)
	}
	// validate that the plugin is registered
	plugins := r.GetPlugins()
	if _, ok := plugins[pc.plugins[i]]; !ok {
		return fmt.Errorf("plugin index %d is not found in the registry", i)
	}
	pc.plugins = append(pc.plugins[:i], append([]string{typeKey}, pc.plugins[i:]...)...)
	return nil
}

func (pc *pluginsChain) GetPlugins() []string {
	return pc.plugins
}

// MergeMaps copies all key/value pairs from src into dst and returns dst.
// If dst is nil a new map is allocated.
// Existing keys in dst are not overwritten.
// This is a helper function used to merge headers from multiple plugins safely.
func MergeMaps(dst map[string]string, src map[string]string) map[string]string {
	if src == nil {
		if dst == nil {
			return map[string]string{}
		}
		return dst
	}
	if dst == nil {
		dst = make(map[string]string, len(src))
	}

	for k, v := range src {
		if _, exists := dst[k]; !exists {
			dst[k] = v
		}
	}

	return dst
}

func (pc *pluginsChain) Run(
	bodyBytes []byte,
	r PluginRegistry,
) (headers map[string]string, mutateBodyBytes []byte, err error) {

	allHeaders := make(map[string]string)
	mutatedBodyBytes := bodyBytes

	for i := range pc.Length() {
		plugin, _ := pc.GetPlugin(i, r)
		pluginType := plugin.TypedName().Type

		metExtPlugin, err := r.GetPlugin(pluginType)

		if err != nil {
			return allHeaders, bodyBytes, err
		}

		// The plugin i in the chain receives the (potentially mutated) body and headers from plugin i-1 in the chain
		headers, mutatedBodyBytes, err := metExtPlugin.Execute(mutatedBodyBytes)

		if err != nil {
			return headers, mutatedBodyBytes, err
		}

		//note that the existing overlapping keys are NOT over-written by merge
		MergeMaps(allHeaders, headers)
	}
	return allHeaders, mutatedBodyBytes, nil
}

func (pc *pluginsChain) String() string {
	return fmt.Sprintf("PluginsChain{plugins=%v}", pc.plugins)
}

// -------------------------- End of PluginsChain implementation --------------------------
