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
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// --- mock plugins ---

type mockRequestPlugin struct {
	typedName plugin.TypedName
}

func (m *mockRequestPlugin) TypedName() plugin.TypedName { return m.typedName }
func (m *mockRequestPlugin) ProcessRequest(_ context.Context, _ *framework.CycleState, _ *framework.InferenceRequest) error {
	return nil
}

type mockResponsePlugin struct {
	typedName plugin.TypedName
}

func (m *mockResponsePlugin) TypedName() plugin.TypedName { return m.typedName }
func (m *mockResponsePlugin) ProcessResponse(_ context.Context, _ *framework.CycleState, _ *framework.InferenceResponse) error {
	return nil
}

// mockBothPlugin implements both RequestProcessor and ResponseProcessor.
type mockBothPlugin struct {
	typedName plugin.TypedName
}

func (m *mockBothPlugin) TypedName() plugin.TypedName { return m.typedName }
func (m *mockBothPlugin) ProcessRequest(_ context.Context, _ *framework.CycleState, _ *framework.InferenceRequest) error {
	return nil
}
func (m *mockBothPlugin) ProcessResponse(_ context.Context, _ *framework.CycleState, _ *framework.InferenceResponse) error {
	return nil
}

// mockPlainPlugin implements only BBRPlugin (no processor interface).
type mockPlainPlugin struct {
	typedName plugin.TypedName
}

func (m *mockPlainPlugin) TypedName() plugin.TypedName { return m.typedName }

// --- helpers ---

func registerMockFactory(pluginType string, factory framework.FactoryFunc) {
	framework.Registry[pluginType] = factory
}

func cleanupRegistry(types ...string) {
	for _, t := range types {
		delete(framework.Registry, t)
	}
}

func requestFactory(name string, _ json.RawMessage, _ framework.Handle) (framework.BBRPlugin, error) {
	return &mockRequestPlugin{typedName: plugin.TypedName{Type: "mock-request", Name: name}}, nil
}

func responseFactory(name string, _ json.RawMessage, _ framework.Handle) (framework.BBRPlugin, error) {
	return &mockResponsePlugin{typedName: plugin.TypedName{Type: "mock-response", Name: name}}, nil
}

func bothFactory(name string, _ json.RawMessage, _ framework.Handle) (framework.BBRPlugin, error) {
	return &mockBothPlugin{typedName: plugin.TypedName{Type: "mock-both", Name: name}}, nil
}

func plainFactory(name string, _ json.RawMessage, _ framework.Handle) (framework.BBRPlugin, error) {
	return &mockPlainPlugin{typedName: plugin.TypedName{Type: "mock-plain", Name: name}}, nil
}

func failingFactory(_ string, _ json.RawMessage, _ framework.Handle) (framework.BBRPlugin, error) {
	return nil, errors.New("factory error")
}

// --- TestLoadRawConfig ---

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

// --- TestInstantiateAndConfigure ---

func TestInstantiateAndConfigure(t *testing.T) {
	tests := []struct {
		name          string
		setup         func()
		teardown      func()
		cfg           *configapi.BodyBasedRoutingConfig
		wantReqCount  int
		wantRespCount int
		wantReqOrder  []string
		wantRespOrder []string
		wantErr       string
	}{
		{
			name: "request and response plugins assembled in order",
			setup: func() {
				registerMockFactory("mock-request", requestFactory)
				registerMockFactory("mock-response", responseFactory)
			},
			teardown: func() { cleanupRegistry("mock-request", "mock-response") },
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Name: "req-a", Type: "mock-request"},
					{Name: "req-b", Type: "mock-request"},
					{Name: "resp-a", Type: "mock-response"},
				},
				Request: []configapi.BBRPluginRef{
					{PluginRef: "req-b"},
					{PluginRef: "req-a"},
				},
				Response: []configapi.BBRPluginRef{
					{PluginRef: "resp-a"},
				},
			},
			wantReqCount:  2,
			wantRespCount: 1,
			wantReqOrder:  []string{"req-b", "req-a"},
			wantRespOrder: []string{"resp-a"},
		},
		{
			name: "plugin in both request and response pipelines",
			setup: func() {
				registerMockFactory("mock-both", bothFactory)
			},
			teardown: func() { cleanupRegistry("mock-both") },
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Name: "dual", Type: "mock-both"},
				},
				Request:  []configapi.BBRPluginRef{{PluginRef: "dual"}},
				Response: []configapi.BBRPluginRef{{PluginRef: "dual"}},
			},
			wantReqCount:  1,
			wantRespCount: 1,
			wantReqOrder:  []string{"dual"},
			wantRespOrder: []string{"dual"},
		},
		{
			name: "empty pipelines",
			setup: func() {
				registerMockFactory("mock-request", requestFactory)
			},
			teardown: func() { cleanupRegistry("mock-request") },
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Name: "unused", Type: "mock-request"},
				},
			},
			wantReqCount:  0,
			wantRespCount: 0,
		},
		{
			name:     "unregistered plugin type",
			setup:    func() {},
			teardown: func() {},
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Name: "foo", Type: "nonexistent-type"},
				},
			},
			wantErr: `plugin type "nonexistent-type" is not registered`,
		},
		{
			name: "factory returns error",
			setup: func() {
				registerMockFactory("mock-fail", failingFactory)
			},
			teardown: func() { cleanupRegistry("mock-fail") },
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Name: "bad", Type: "mock-fail"},
				},
			},
			wantErr: `failed to create plugin "bad" (type: mock-fail): factory error`,
		},
		{
			name: "request pipeline references plugin that does not implement RequestProcessor",
			setup: func() {
				registerMockFactory("mock-response", responseFactory)
			},
			teardown: func() { cleanupRegistry("mock-response") },
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Name: "resp-only", Type: "mock-response"},
				},
				Request: []configapi.BBRPluginRef{{PluginRef: "resp-only"}},
			},
			wantErr: `request[0]: plugin "resp-only" does not implement RequestProcessor`,
		},
		{
			name: "response pipeline references plugin that does not implement ResponseProcessor",
			setup: func() {
				registerMockFactory("mock-request", requestFactory)
			},
			teardown: func() { cleanupRegistry("mock-request") },
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Name: "req-only", Type: "mock-request"},
				},
				Response: []configapi.BBRPluginRef{{PluginRef: "req-only"}},
			},
			wantErr: `response[0]: plugin "req-only" does not implement ResponseProcessor`,
		},
		{
			name: "plain plugin fails interface check in request",
			setup: func() {
				registerMockFactory("mock-plain", plainFactory)
			},
			teardown: func() { cleanupRegistry("mock-plain") },
			cfg: &configapi.BodyBasedRoutingConfig{
				Plugins: []configapi.PluginSpec{
					{Name: "plain", Type: "mock-plain"},
				},
				Request: []configapi.BBRPluginRef{{PluginRef: "plain"}},
			},
			wantErr: `request[0]: plugin "plain" does not implement RequestProcessor`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.setup()
			defer tc.teardown()

			reqPlugins, respPlugins, err := InstantiateAndConfigure(tc.cfg, nil)
			if tc.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Len(t, reqPlugins, tc.wantReqCount)
			assert.Len(t, respPlugins, tc.wantRespCount)

			// Verify order.
			if tc.wantReqOrder != nil {
				for i, name := range tc.wantReqOrder {
					assert.Equal(t, name, reqPlugins[i].TypedName().Name, "request plugin order mismatch at index %d", i)
				}
			}
			if tc.wantRespOrder != nil {
				for i, name := range tc.wantRespOrder {
					assert.Equal(t, name, respPlugins[i].TypedName().Name, "response plugin order mismatch at index %d", i)
				}
			}
		})
	}
}
