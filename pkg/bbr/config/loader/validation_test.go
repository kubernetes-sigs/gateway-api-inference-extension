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

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

func TestValidateConfig(t *testing.T) {
	tests := []struct {
		name    string
		cfg     *configapi.BodyBasedRoutingConfig
		wantErr string
	}{
		{
			name: "valid config",
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Name: "plugin-a", Type: "type-a"},
					{Name: "plugin-b", Type: "type-b"},
				},
				Request:  []configapi.BBRPluginRef{{PluginRef: "plugin-a"}},
				Response: []configapi.BBRPluginRef{{PluginRef: "plugin-b"}},
			},
		},
		{
			name: "valid config with empty pipelines",
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Name: "plugin-a", Type: "type-a"},
				},
			},
		},
		{
			name: "plugin missing type",
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Name: "plugin-a", Type: ""},
				},
			},
			wantErr: `plugin validation failed: plugins[0] (name: "plugin-a") is missing a type`,
		},
		{
			name: "duplicate plugin name",
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Name: "plugin-a", Type: "type-a"},
					{Name: "plugin-a", Type: "type-b"},
				},
			},
			wantErr: `plugin validation failed: plugins[1] has duplicate name "plugin-a"`,
		},
		{
			name: "request references undefined plugin",
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Name: "plugin-a", Type: "type-a"},
				},
				Request: []configapi.BBRPluginRef{{PluginRef: "nonexistent"}},
			},
			wantErr: `request pipeline validation failed: request[0] references undefined plugin "nonexistent"`,
		},
		{
			name: "response references undefined plugin",
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Name: "plugin-a", Type: "type-a"},
				},
				Response: []configapi.BBRPluginRef{{PluginRef: "nonexistent"}},
			},
			wantErr: `response pipeline validation failed: response[0] references undefined plugin "nonexistent"`,
		},
		{
			name: "request with empty pluginRef",
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Name: "plugin-a", Type: "type-a"},
				},
				Request: []configapi.BBRPluginRef{{PluginRef: ""}},
			},
			wantErr: `request pipeline validation failed: request[0] is missing a pluginRef`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateConfig(tc.cfg)
			if tc.wantErr != "" {
				assert.EqualError(t, err, tc.wantErr)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
