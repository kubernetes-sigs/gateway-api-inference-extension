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

import (
	"context"
	"encoding/json"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	DefaultPluginType = "no-op-plugin"
)

// compile-time type validation
var _ framework.PayloadProcessor = &DefaultPlugin{}

type DefaultPlugin struct {
	typedName plugin.TypedName
}

// DefaultPluginFactory defines the factory function for DefaultPlugin.
// The name and rawParameters are ignored as the plugin uses the default configuration.
func DefaultPluginFactory(_ string, _ json.RawMessage) (framework.PayloadProcessor, error) {
	return NewDefaultPlugin(), nil
}

// NewDefaultPlugin returns a concrete *DefaultPlugin.
func NewDefaultPlugin() *DefaultPlugin {
	return &DefaultPlugin{
		typedName: plugin.TypedName{Type: DefaultPluginType, Name: DefaultPluginType},
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *DefaultPlugin) TypedName() plugin.TypedName {
	return p.typedName
}

// WithName sets the name of the default BBR plugin
func (p *DefaultPlugin) WithName(name string) *DefaultPlugin {
	p.typedName.Name = name
	return p
}

// The current default plugin is a no-op and returns the request body unchanged.
// After integration, "no-op-plugin" will be replaced by "default-plugin",
// which will extract the model name, map it to a base model, and set the
// necessary request headers.
func (p *DefaultPlugin) Execute(ctx context.Context, headers map[string]string, body map[string]any) (map[string]string, map[string]any, error) {
	return headers, body, nil
}
