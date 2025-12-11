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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

// mockPluginImpl is a simple struct to verify casting/retrieval.
type mockPluginImpl struct{ name string }

func (m *mockPluginImpl) TypedName() TypedName { return TypedName{Type: "mock-plugin", Name: m.name} }

func TestNewEppHandle(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	specs := []configapi.PluginSpec{
		{Name: "spec-1", Type: "type-a"},
	}
	instance := &mockPluginImpl{name: "instance-1"}
	pluginMap := map[string]Plugin{
		"instance-1": instance,
	}

	handle := NewEppHandle(ctx, specs, pluginMap, nil)

	assert.Equal(t, ctx, handle.Context(), "handle should return the provided context")

	retrievedSpec := handle.PluginSpec("spec-1")
	require.NotNil(t, retrievedSpec, "expected to retrieve configured spec")
	assert.Equal(t, "type-a", retrievedSpec.Type, "spec data should match input")

	retrievedPlugin := handle.Plugin("instance-1")
	assert.Equal(t, instance, retrievedPlugin, "expected to retrieve configured plugin instance")
}

func TestEppHandle_Immutability(t *testing.T) {
	t.Parallel()

	mutableSpecs := []configapi.PluginSpec{
		{Name: "original", Type: "original-type"},
	}
	handle := NewEppHandle(context.Background(), mutableSpecs, nil, nil)

	mutableSpecs[0].Type = "modified-type"

	spec := handle.PluginSpec("original")
	require.NotNil(t, spec, "spec should exist")
	assert.Equal(t, "original-type", spec.Type, "handle should hold a deep copy of the specs, not a reference")
}

func TestEppHandle_AddPlugin(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	handle := NewEppHandle(ctx, nil, nil, nil)
	newPlugin := &mockPluginImpl{name: "dynamic-addition"}

	handle.AddPlugin("dynamic-plugin", newPlugin)

	retrieved := handle.Plugin("dynamic-plugin")
	require.NotNil(t, retrieved, "expected to retrieve added plugin")
	assert.Equal(t, newPlugin, retrieved, "retrieved plugin should match the added instance")

	all := handle.GetAllPluginsWithNames()
	assert.Contains(t, all, "dynamic-plugin")
}

func TestPluginByType(t *testing.T) {
	t.Parallel()

	// We embed mockPluginImpl to automatically satisfy the Plugin interface, but it is a distinct type
	// (*wrongPluginImpl != *mockPluginImpl).
	type wrongPluginImpl struct {
		mockPluginImpl
	}

	correctInstance := &mockPluginImpl{name: "correct"}
	wrongInstance := &wrongPluginImpl{mockPluginImpl{name: "wrong"}}

	pluginMap := map[string]Plugin{
		"valid":   correctInstance,
		"invalid": wrongInstance,
	}
	handle := NewEppHandle(context.Background(), nil, pluginMap, nil)

	tests := []struct {
		name          string
		pluginName    string
		expectErr     bool
		errorContains string
	}{
		{
			name:       "Successful Retrieval",
			pluginName: "valid",
			expectErr:  false,
		},
		{
			name:          "Plugin not Found",
			pluginName:    "missing",
			expectErr:     true,
			errorContains: "not found",
		},
		{
			name:          "Type Mismatch",
			pluginName:    "invalid",
			expectErr:     true,
			errorContains: "expected *plugins.mockPluginImpl",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := PluginByType[*mockPluginImpl](handle, tc.pluginName)

			if tc.expectErr {
				require.Error(t, err)
				assert.Nil(t, result)
				assert.Contains(t, err.Error(), tc.errorContains)
			} else {
				require.NoError(t, err)
				assert.Equal(t, correctInstance, result)
			}
		})
	}
}
