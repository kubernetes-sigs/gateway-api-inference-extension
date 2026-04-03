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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins/basemodelextractor"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins/bodyfieldtoheader"
)

func TestLoadDefaultConfig(t *testing.T) {
	cfg := loadDefaultConfig()

	require.NotNil(t, cfg)
	assert.Equal(t, "inference.networking.x-k8s.io/v1alpha1", cfg.APIVersion)
	assert.Equal(t, "BodyBasedRoutingConfig", cfg.Kind)

	// Should have 2 default plugins.
	require.Len(t, cfg.Plugins, 2)
	assert.Equal(t, bodyfieldtoheader.BodyFieldToHeaderPluginType, cfg.Plugins[0].Type)
	assert.Equal(t, basemodelextractor.BaseModelToHeaderPluginType, cfg.Plugins[1].Type)

	// body-field-to-header should have parameters with model field and header.
	assert.Contains(t, string(cfg.Plugins[0].Parameters), `"field_name":"model"`)
	assert.Contains(t, string(cfg.Plugins[0].Parameters), `"header_name":"`+bodyfieldtoheader.ModelHeader+`"`)

	// base-model-to-header should have no parameters.
	assert.Nil(t, cfg.Plugins[1].Parameters)

	// Request pipeline should reference both plugins in order.
	require.Len(t, cfg.Request, 2)
	assert.Equal(t, bodyfieldtoheader.BodyFieldToHeaderPluginType, cfg.Request[0].PluginRef)
	assert.Equal(t, basemodelextractor.BaseModelToHeaderPluginType, cfg.Request[1].PluginRef)

	// Response pipeline should be empty.
	assert.Empty(t, cfg.Response)
}

func TestApplyStaticDefaults(t *testing.T) {
	tests := []struct {
		name     string
		cfg      *configapi.BodyBasedRoutingConfig
		wantName string
	}{
		{
			name: "name defaults to type when empty",
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Type: "my-plugin"},
				},
			},
			wantName: "my-plugin",
		},
		{
			name: "explicit name is preserved",
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Type: "my-plugin", Name: "custom-name"},
				},
			},
			wantName: "custom-name",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			applyStaticDefaults(tc.cfg)
			assert.Equal(t, tc.wantName, tc.cfg.Plugins[0].Name)
		})
	}
}
