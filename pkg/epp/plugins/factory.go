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
	"fmt"
)

// PluginFactory abstracts the creation of transient plugin instances.
type PluginFactory interface {
	// NewPlugin creates a fresh instance of a plugin based on a configuration blueprint.
	//
	// Arguments:
	//   - blueprintName: The name of the PluginSpec in the configuration (e.g., "standard-fairness-policy").
	//   - instanceAlias: An optional runtime name for this specific instance (e.g., "tenant-a").
	//                    If empty, the blueprintName is used.
	NewPlugin(blueprintName string, instanceAlias string) (Plugin, error)
}

// EPPPluginFactory is the concrete implementation of PluginFactory.
// It ties together the configuration (Handle) and the implementation (Registry).
type EPPPluginFactory struct {
	handle Handle
}

// NewEPPPluginFactory returns a new factory instance.
func NewEPPPluginFactory(handle Handle) *EPPPluginFactory {
	return &EPPPluginFactory{handle: handle}
}

// NewPlugin implements PluginFactory.
func (f *EPPPluginFactory) NewPlugin(blueprintName string, instanceAlias string) (Plugin, error) {
	spec := f.handle.PluginSpec(blueprintName)
	if spec == nil {
		return nil, fmt.Errorf("plugin blueprint %q not found in configuration", blueprintName)
	}

	reg, ok := Registry[spec.Type]
	if !ok {
		return nil, fmt.Errorf("plugin type %q (referenced by blueprint %q) is not registered", spec.Type, blueprintName)
	}

	// Determine runtime identity. This ensures that structured logs and internal maps can distinguish between different
	// instances of the same plugin type.
	finalName := spec.Name
	if instanceAlias != "" {
		finalName = instanceAlias
	}

	plugin, err := reg.Factory(finalName, spec.Parameters, f.handle)
	if err != nil {
		return nil, fmt.Errorf("failed to instantiate plugin %q (type %s): %w", finalName, spec.Type, err)
	}

	return plugin, nil
}

// NewPluginByType is a helper to create a plugin and assert its type in one step.
func NewPluginByType[T Plugin](factory PluginFactory, blueprintName string, instanceAlias string) (T, error) {
	var zero T

	rawPlugin, err := factory.NewPlugin(blueprintName, instanceAlias)
	if err != nil {
		return zero, err
	}

	plugin, ok := rawPlugin.(T)
	if !ok {
		return zero, fmt.Errorf(
			"plugin created from blueprint %q is type %T, but expected %T",
			blueprintName, rawPlugin, zero,
		)
	}

	return plugin, nil
}
