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

package plugins

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

// transientPlugin is a mock implementation that captures how it was created.
type transientPlugin struct {
	Name       string
	Parameters string
}

func (p *transientPlugin) TypedName() TypedName {
	return TypedName{Type: "transientPlugin", Name: p.Name}
}

// transientFactory creates transientPlugin instances.
func transientFactory(name string, parameters json.RawMessage, _ Handle) (Plugin, error) {
	if string(parameters) == `"fail-me"` {
		return nil, errors.New("intentional factory failure")
	}
	return &transientPlugin{
		Name:       name,
		Parameters: string(parameters),
	}, nil
}

func TestEPPPluginFactory_NewPlugin(t *testing.T) {
	// Register a known type for testing factories.
	// We use a unique name to ensure parallel safety if other tests use the registry.
	const factoryType = "test-transient-factory-type"
	RegisterWithMetadata(factoryType, PluginRegistration{
		Factory:   transientFactory,
		Lifecycle: LifecycleTransient,
	})

	t.Parallel()

	specs := []configapi.PluginSpec{
		{
			Name:       "blueprint-default",
			Type:       factoryType,
			Parameters: json.RawMessage(`{"key": "val"}`),
		},
		{
			Name:       "blueprint-broken-factory",
			Type:       factoryType,
			Parameters: json.RawMessage(`"fail-me"`),
		},
		{
			Name: "blueprint-missing-type",
			Type: "unknown-type",
		},
	}

	handle := NewEppHandle(context.Background(), specs, nil, nil)
	factory := NewEPPPluginFactory(handle)

	tests := []struct {
		name           string
		blueprintName  string
		instanceAlias  string
		expectErr      bool
		errorContains  string
		expectedName   string // The name the plugin instance should think it has.
		expectedParams string
	}{
		{
			name:           "success_standard_creation",
			blueprintName:  "blueprint-default",
			instanceAlias:  "", // No alias
			expectErr:      false,
			expectedName:   "blueprint-default", // Should default to blueprint name.
			expectedParams: `{"key": "val"}`,
		},
		{
			name:           "success_with_instance_alias",
			blueprintName:  "blueprint-default",
			instanceAlias:  "tenant-a-queue",
			expectErr:      false,
			expectedName:   "tenant-a-queue", // Should take the alias.
			expectedParams: `{"key": "val"}`,
		},
		{
			name:          "fail_blueprint_not_found",
			blueprintName: "non-existent-blueprint",
			expectErr:     true,
			errorContains: "blueprint \"non-existent-blueprint\" not found",
		},
		{
			name:          "fail_plugin_type_not_registered",
			blueprintName: "blueprint-missing-type",
			expectErr:     true,
			errorContains: "plugin type \"unknown-type\" (referenced by blueprint \"blueprint-missing-type\") is not registered",
		},
		{
			name:          "fail_factory_returns_error",
			blueprintName: "blueprint-broken-factory",
			expectErr:     true,
			errorContains: "failed to instantiate plugin",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			plugin, err := factory.NewPlugin(tc.blueprintName, tc.instanceAlias)

			if tc.expectErr {
				require.Error(t, err)
				assert.Nil(t, plugin)
				assert.Contains(t, err.Error(), tc.errorContains)
			} else {
				require.NoError(t, err)
				require.NotNil(t, plugin)

				// Cast to verify internal state
				p, ok := plugin.(*transientPlugin)
				require.True(t, ok, "plugin should be of type *transientPlugin")
				assert.Equal(t, tc.expectedName, p.Name, "plugin name should match expected identity")
				assert.JSONEq(t, tc.expectedParams, p.Parameters, "parameters should be passed through correctly")
			}
		})
	}
}

func TestNewPluginByType_GenericHelper(t *testing.T) {
	const helperType = "test-helper-type"
	RegisterWithMetadata(helperType, PluginRegistration{
		Factory:   transientFactory,
		Lifecycle: LifecycleTransient,
	})

	t.Parallel()

	specs := []configapi.PluginSpec{{Name: "bp", Type: helperType}}
	handle := NewEppHandle(context.Background(), specs, nil, nil)
	factory := NewEPPPluginFactory(handle)

	t.Run("success_cast", func(t *testing.T) {
		p, err := NewPluginByType[*transientPlugin](factory, "bp", "alias")
		require.NoError(t, err)
		assert.Equal(t, "alias", p.Name)
	})

	t.Run("fail_cast_mismatch", func(t *testing.T) {
		// Try to cast the transientPlugin to mockPluginImpl (which it is not).
		_, err := NewPluginByType[*mockPluginImpl](factory, "bp", "alias")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "is type *plugins.transientPlugin, but expected *plugins.mockPluginImpl")
	})
}
