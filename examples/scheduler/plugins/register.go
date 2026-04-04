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

// Package plugins provides the registration of all custom scheduler plugins.
//
// Call RegisterAllPlugins() from your main.go before starting the Runner.
// This follows the same pattern as llm-d-inference-scheduler:
// https://github.com/llm-d/llm-d-inference-scheduler/blob/main/pkg/plugins/register.go
package plugins

import (
	"sigs.k8s.io/gateway-api-inference-extension/examples/scheduler/plugins/filter"
	"sigs.k8s.io/gateway-api-inference-extension/examples/scheduler/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/examples/scheduler/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/examples/scheduler/plugins/scorer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// RegisterAllPlugins registers the factory functions of all custom plugins
// in this package. It must be called before the Runner processes configuration,
// so that the config loader can find these plugin types by name.
func RegisterAllPlugins() {
	// Filters
	plugin.Register(filter.PluginType, filter.Factory)
	// Scorers
	plugin.Register(scorer.PluginType, scorer.Factory)
	// Pickers
	plugin.Register(picker.PluginType, picker.Factory)
	// ProfileHandlers
	plugin.Register(profile.PluginType, profile.Factory)
}
