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
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
)

func TestLoadRawConfig(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		configText string
		want       *configapi.BodyBasedRoutingConfig
		wantErr    bool
	}{
		{
			name: "full valid config",
			configText: `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: BodyBasedRoutingConfig
plugins:
  - type: body-field-to-header
    name: model-extractor
    parameters:
      field_name: model
      header_name: X-Gateway-Model-Name
  - type: base-model-to-header
request:
  - pluginRef: model-extractor
  - pluginRef: base-model-to-header
response: []
`,
			want: &configapi.BodyBasedRoutingConfig{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "inference.networking.x-k8s.io/v1alpha1",
					Kind:       "BodyBasedRoutingConfig",
				},
				Plugins: []configapi.PluginSpec{
					{
						Name:       "model-extractor",
						Type:       "body-field-to-header",
						Parameters: json.RawMessage(`{"field_name":"model","header_name":"X-Gateway-Model-Name"}`),
					},
					{
						Name: "base-model-to-header",
						Type: "base-model-to-header",
					},
				},
				Request: []configapi.BBRPluginRef{
					{PluginRef: "model-extractor"},
					{PluginRef: "base-model-to-header"},
				},
				Response: []configapi.BBRPluginRef{},
			},
		},
		{
			name: "plugin name defaults to type via static defaults",
			configText: `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: BodyBasedRoutingConfig
plugins:
  - type: body-field-to-header
request:
  - pluginRef: body-field-to-header
`,
			want: &configapi.BodyBasedRoutingConfig{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "inference.networking.x-k8s.io/v1alpha1",
					Kind:       "BodyBasedRoutingConfig",
				},
				Plugins: []configapi.PluginSpec{
					{
						Name: "body-field-to-header",
						Type: "body-field-to-header",
					},
				},
				Request: []configapi.BBRPluginRef{
					{PluginRef: "body-field-to-header"},
				},
			},
		},
		{
			name: "empty config uses defaults with static defaults applied",
			want: func() *configapi.BodyBasedRoutingConfig {
				cfg := loadDefaultConfig()
				applyStaticDefaults(cfg)
				return cfg
			}(),
		},
		{
			name:       "invalid YAML",
			configText: `not: valid: yaml: [`,
			wantErr:    true,
		},
		{
			name: "wrong kind",
			configText: `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins: []
`,
			wantErr: true,
		},
		{
			name: "unknown fields are rejected (strict mode)",
			configText: `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: BodyBasedRoutingConfig
plugins: []
unknownField: value
`,
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			logger := logging.NewTestLogger()

			got, err := LoadRawConfig([]byte(tc.configText), logger)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error but got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("LoadRawConfig() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
