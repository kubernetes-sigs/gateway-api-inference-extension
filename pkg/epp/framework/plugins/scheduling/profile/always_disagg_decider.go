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

package profile

import (
	"context"
	"encoding/json"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	// AlwaysDisaggDeciderPluginType is the type-name of the alwaysDisaggPDDecider plugin.
	AlwaysDisaggDeciderPluginType = "always-disagg-pd-decider"
)

// compile-time type assertion
var _ pdDeciderPlugin = &AlwaysDisaggPDDecider{}

// AlwaysDisaggPDDecider is a PD decider plugin which always decide to disaggregate PD
type AlwaysDisaggPDDecider struct {
	typedName plugin.TypedName
}

// AlwaysDisaggPDDeciderPluginFactory defines the factory function for creating
// a new instance of the AlwaysDisaggPDDecider.
func AlwaysDisaggPDDeciderPluginFactory(name string, _ json.RawMessage,
	_ plugin.Handle) (plugin.Plugin, error) {
	return NewAlwaysDisaggPDDecider().WithName(name), nil
}

func NewAlwaysDisaggPDDecider() *AlwaysDisaggPDDecider {
	return &AlwaysDisaggPDDecider{
		typedName: plugin.TypedName{Type: AlwaysDisaggDeciderPluginType},
	}
}

// TypedName returns the typed name of the plugin.
func (d *AlwaysDisaggPDDecider) TypedName() plugin.TypedName {
	return d.typedName
}

// WithName sets the name of the plugin.
func (d *AlwaysDisaggPDDecider) WithName(name string) *AlwaysDisaggPDDecider {
	d.typedName.Name = name
	return d
}

func (d *AlwaysDisaggPDDecider) disaggregate(ctx context.Context, inputTokens int, endpoint scheduling.Endpoint) bool {
	return true
}
