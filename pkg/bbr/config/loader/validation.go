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

package loader

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

// validateConfig performs validation of the BBR configuration integrity.
// It checks plugin definitions and pipeline references.
func validateConfig(cfg *configapi.BodyBasedRoutingConfig) error {
	definedPlugins, err := validatePlugins(cfg.Plugins)
	if err != nil {
		return fmt.Errorf("plugin validation failed: %w", err)
	}
	if err := validatePipeline("request", cfg.Request, definedPlugins); err != nil {
		return fmt.Errorf("request pipeline validation failed: %w", err)
	}
	if err := validatePipeline("response", cfg.Response, definedPlugins); err != nil {
		return fmt.Errorf("response pipeline validation failed: %w", err)
	}
	return nil
}

// validatePlugins checks that all plugin specs have a non-empty type and unique names.
// It returns the set of defined plugin names for use in pipeline validation.
func validatePlugins(plugins []configapi.PluginSpec) (sets.Set[string], error) {
	pluginNames := sets.New[string]()
	for i, p := range plugins {
		if p.Type == "" {
			return nil, fmt.Errorf("plugins[%d] (name: %q) is missing a type", i, p.Name)
		}
		if pluginNames.Has(p.Name) {
			return nil, fmt.Errorf("plugins[%d] has duplicate name %q", i, p.Name)
		}
		pluginNames.Insert(p.Name)
	}
	return pluginNames, nil
}

// validatePipeline checks that all pluginRef entries in a pipeline reference
// plugins defined in the Plugins section.
func validatePipeline(pipelineName string, refs []configapi.BBRPluginRef, definedPlugins sets.Set[string]) error {
	for i, ref := range refs {
		if ref.PluginRef == "" {
			return fmt.Errorf("%s[%d] is missing a pluginRef", pipelineName, i)
		}
		if !definedPlugins.Has(ref.PluginRef) {
			return fmt.Errorf("%s[%d] references undefined plugin %q", pipelineName, i, ref.PluginRef)
		}
	}
	return nil
}
