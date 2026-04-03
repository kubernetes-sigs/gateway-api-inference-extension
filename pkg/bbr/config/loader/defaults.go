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
	"encoding/json"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins/basemodelextractor"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins/bodyfieldtoheader"
)

// defaultModelField is the body field name used in the default configuration.
// This mirrors the modelField constant in cmd/bbr/runner/runner.go.
const defaultModelField = "model"

// loadDefaultConfig returns the default BBR configuration, equivalent to the
// default behavior when no --plugin flags are specified in runner.go:177-196.
func loadDefaultConfig() *configapi.BodyBasedRoutingConfig {
	return &configapi.BodyBasedRoutingConfig{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "inference.networking.x-k8s.io/v1alpha1",
			Kind:       "BodyBasedRoutingConfig",
		},
		Plugins: []configapi.PluginSpec{
			{
				Type: bodyfieldtoheader.BodyFieldToHeaderPluginType,
				Parameters: json.RawMessage(fmt.Sprintf(
					`{"field_name":%q,"header_name":%q}`,
					defaultModelField,
					bodyfieldtoheader.ModelHeader,
				)),
			},
			{
				Type: basemodelextractor.BaseModelToHeaderPluginType,
			},
		},
		Request: []configapi.BBRPluginRef{
			{PluginRef: bodyfieldtoheader.BodyFieldToHeaderPluginType},
			{PluginRef: basemodelextractor.BaseModelToHeaderPluginType},
		},
		Response: []configapi.BBRPluginRef{},
	}
}

// applyStaticDefaults sanitizes the configuration object before plugin instantiation.
// It sets the plugin name to the plugin type when name is omitted.
func applyStaticDefaults(cfg *configapi.BodyBasedRoutingConfig) {
	for idx, pluginConfig := range cfg.Plugins {
		if pluginConfig.Name == "" {
			cfg.Plugins[idx].Name = pluginConfig.Type
		}
	}
}
