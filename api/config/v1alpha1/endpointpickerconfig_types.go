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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:defaulter-gen=true
// +kubebuilder:object:root=true

// EndpointPickerConfig is the Schema for the endpointpickerconfigs API
type EndpointPickerConfig struct {
	metav1.TypeMeta `json:",inline"`

	// +required
	// +kubebuilder:validation:Required
	Plugins []PluginSpec `json:"plugins"`

	// +required
	// +kubebuilder:validation:Required
	SchedulingProfiles []SchedulingProfile `json:"schedulingProfiles"`
}
type PluginSpec struct {
	// +optional
	Name string `json:"name"`

	// +required
	// +kubebuilder:validation:Required
	PluginName string `json:"pluginName"`

	// +optional
	Parameters json.RawMessage `json:"parameters"`
}

type SchedulingProfile struct {
	// +optional
	Name string `json:"name"`

	// +required
	// +kubebuilder:validation:Required
	Plugins []SchedulingProfilePlugin `json:"plugins"`
}

type SchedulingProfilePlugin struct {
	// +required
	// +kubebuilder:validation:Required
	PluginRef string `json:"pluginRef"`

	// +optional
	Weight *int `json:"weight"`
}
