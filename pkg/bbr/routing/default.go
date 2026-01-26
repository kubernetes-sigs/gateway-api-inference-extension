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

package routing

import (
	"encoding/json"

	bbr "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins"
	epp "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	DefaultPluginType = "default-bbr"
)

// compile-time type validation
var _ bbr.BBRPlugin = &DefaultPlugin{}

type DefaultPlugin struct {
	typedName epp.TypedName
}

// DefaultPluginFactory defines the factory function for DefaultPlugin.
// The name and rawParameters are ignored as the plugin uses the default configuration.
func DefaultPluginFactory(_ string, _ json.RawMessage) (bbr.BBRPlugin, error) {
	return NewDefaultPlugin(), nil
}

// / NewDefaultPlugin returns a concrete *DefaultPlugin.
func NewDefaultPlugin() *DefaultPlugin {
	return &DefaultPlugin{
		typedName: epp.TypedName{Type: DefaultPluginType, Name: DefaultPluginType},
	}
}

func (p *DefaultPlugin) Execute(requestBodyBytes []byte) ([]byte, map[string][]string, error) {
	// No-op BBR plugin to be replaced by the actual logic currently running in the
	return requestBodyBytes, nil, nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *DefaultPlugin) TypedName() epp.TypedName {
	return p.typedName
}

// WithName sets the name of the default BBR plugin
func (p *DefaultPlugin) WithName(name string) *DefaultPlugin {
	p.typedName.Name = name
	return p
}
