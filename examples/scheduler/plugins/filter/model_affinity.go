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

// Package filter demonstrates how to implement a custom scheduling Filter plugin.
//
// A Filter removes candidate endpoints that should not receive the current request.
// The scheduling framework calls every registered Filter in sequence; each Filter
// receives the survivors of the previous one.
//
// To implement your own Filter:
//
//  1. Define a struct that holds any configuration your filter needs.
//  2. Implement plugin.Plugin  — return a TypedName (type + instance name).
//  3. Implement scheduling.Filter — the Filter method receives the LLMRequest and
//     the current candidate list, and returns the subset that should proceed to scoring.
//  4. Provide a FactoryFunc so the config loader can instantiate the plugin from YAML.
//  5. Register the factory in RegisterAllPlugins() (see plugins/register.go).
package filter

import (
	"context"
	"encoding/json"
	"fmt"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const PluginType = "model-affinity-filter"

// compile-time interface check
var _ fwksched.Filter = &ModelAffinityFilter{}

// ModelAffinityFilter drops any endpoint whose ActiveModels map does not
// contain the request's TargetModel. This ensures only pods already serving
// the requested model are considered for scheduling.
type ModelAffinityFilter struct {
	typedName fwkplugin.TypedName
}

// New creates a ModelAffinityFilter with default type and name.
func New() *ModelAffinityFilter {
	return &ModelAffinityFilter{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: PluginType},
	}
}

func (f *ModelAffinityFilter) TypedName() fwkplugin.TypedName { return f.typedName }

// Filter keeps only the endpoints that have request.TargetModel loaded.
func (f *ModelAffinityFilter) Filter(
	_ context.Context,
	_ *fwksched.CycleState,
	request *fwksched.LLMRequest,
	endpoints []fwksched.Endpoint,
) []fwksched.Endpoint {
	filtered := make([]fwksched.Endpoint, 0, len(endpoints))
	for _, ep := range endpoints {
		if _, ok := ep.GetMetrics().ActiveModels[request.TargetModel]; ok {
			filtered = append(filtered, ep)
		}
	}
	fmt.Printf("  [filter/%s] %d → %d endpoints (kept those with model %q)\n",
		PluginType, len(endpoints), len(filtered), request.TargetModel)
	return filtered
}

// Factory is the FactoryFunc that the plugin registry uses to create instances
// from an EndpointPickerConfig YAML. The `name` parameter allows multiple
// instances of the same plugin type with different names.
func Factory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	f := New()
	f.typedName.Name = name
	return f, nil
}
