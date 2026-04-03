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

package v1alpha1

import (
	"encoding/json"
	"fmt"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +kubebuilder:object:root=true

// BodyBasedRoutingConfig is the Schema for the bodybasedroutingconfigs API.
// It defines the plugins and request/response pipelines for the BBR component.
type BodyBasedRoutingConfig struct {
	metav1.TypeMeta `json:",inline"`

	// +required
	// +kubebuilder:validation:Required
	// Plugins is the list of plugins that will be instantiated.
	Plugins []PluginSpec `json:"plugins"`

	// +optional
	// Request is the ordered list of plugins to execute on the request path.
	// Plugins are executed in the order they appear in this list.
	// Each referenced plugin must implement the RequestProcessor interface.
	Request []BBRPluginRef `json:"request,omitempty"`

	// +optional
	// Response is the ordered list of plugins to execute on the response path.
	// Plugins are executed in the order they appear in this list.
	// Each referenced plugin must implement the ResponseProcessor interface.
	Response []BBRPluginRef `json:"response,omitempty"`
}

func (cfg BodyBasedRoutingConfig) String() string {
	var parts []string
	if len(cfg.Plugins) > 0 {
		parts = append(parts, fmt.Sprintf("Plugins: %v", cfg.Plugins))
	}
	if len(cfg.Request) > 0 {
		parts = append(parts, fmt.Sprintf("Request: %v", cfg.Request))
	}
	if len(cfg.Response) > 0 {
		parts = append(parts, fmt.Sprintf("Response: %v", cfg.Response))
	}
	return "{" + strings.Join(parts, ", ") + "}"
}

// BBRPluginRef references a plugin defined in the Plugins section
// for use in request or response pipelines.
type BBRPluginRef struct {
	// +required
	// +kubebuilder:validation:Required
	// PluginRef specifies a particular Plugin instance to be used in this
	// pipeline step. The reference is to the name of an entry of the Plugins
	// defined in the configuration's Plugins section.
	PluginRef string `json:"pluginRef"`

	// +optional
	// Parameters are optional runtime parameters for this pipeline step.
	// The plugin is responsible for parsing these parameters.
	Parameters json.RawMessage `json:"parameters,omitempty"`
}

func (ref BBRPluginRef) String() string {
	var parts []string
	parts = append(parts, "PluginRef: "+ref.PluginRef)
	if len(ref.Parameters) > 0 {
		parts = append(parts, "Parameters: "+string(ref.Parameters))
	}
	return "{" + strings.Join(parts, ", ") + "}"
}
