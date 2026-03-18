/*
Copyright 2026 The Kubernetes Authors.

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

package basemodelextractor

import (
	"context"
	"testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// TestBaseModelToHeaderPlugin_TypedName tests the TypedName method returns correct type and name.
func TestBaseModelToHeaderPlugin_TypedName(t *testing.T) {
	p := &BaseModelToHeaderPlugin{
		typedName:     plugin.TypedName{Type: BaseModelToHeaderPluginType, Name: "test-plugin"},
		adaptersStore: newAdaptersStore(),
	}

	got := p.TypedName()
	if got.Type != BaseModelToHeaderPluginType {
		t.Errorf("TypedName().Type = %q, want %q", got.Type, BaseModelToHeaderPluginType)
	}
	if got.Name != "test-plugin" {
		t.Errorf("TypedName().Name = %q, want %q", got.Name, "test-plugin")
	}
}

// TestBaseModelToHeaderPlugin_WithName tests that WithName correctly updates the plugin name.
func TestBaseModelToHeaderPlugin_WithName(t *testing.T) {
	p := &BaseModelToHeaderPlugin{
		typedName:     plugin.TypedName{Type: BaseModelToHeaderPluginType, Name: "original"},
		adaptersStore: newAdaptersStore(),
	}

	p = p.WithName("new-name")

	got := p.TypedName()
	if got.Name != "new-name" {
		t.Errorf("TypedName().Name = %q, want %q", got.Name, "new-name")
	}
	if got.Type != BaseModelToHeaderPluginType {
		t.Errorf("TypedName().Type = %q, want %q", got.Type, BaseModelToHeaderPluginType)
	}
}

// TestBaseModelToHeaderPlugin_ProcessRequest_EdgeCases tests edge cases that are difficult
// to trigger in integration tests: nil request, nil headers, nil body.
func TestBaseModelToHeaderPlugin_ProcessRequest_EdgeCases(t *testing.T) {
	p := &BaseModelToHeaderPlugin{
		typedName:     plugin.TypedName{Type: BaseModelToHeaderPluginType, Name: "test"},
		adaptersStore: newAdaptersStore(),
	}

	tests := []struct {
		name string
		req  *framework.InferenceRequest
	}{
		{
			name: "nil request",
			req:  nil,
		},
		{
			name: "nil headers",
			req: &framework.InferenceRequest{
				InferenceMessage: framework.InferenceMessage{
					Body: map[string]any{"model": "test"},
				},
			},
		},
		{
			name: "nil body",
			req:  framework.NewInferenceRequest(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := p.ProcessRequest(context.Background(), nil, tt.req)
			if err != nil {
				t.Errorf("ProcessRequest() error = %v, want nil (should handle gracefully)", err)
			}
		})
	}
}

// TestBaseModelToHeaderPlugin_GetReconciler tests that GetReconciler returns a properly
// configured reconciler with the same adaptersStore.
func TestBaseModelToHeaderPlugin_GetReconciler(t *testing.T) {
	store := newAdaptersStore()
	p := &BaseModelToHeaderPlugin{
		typedName:     plugin.TypedName{Type: BaseModelToHeaderPluginType, Name: "test"},
		adaptersStore: store,
	}

	reconciler := p.GetReconciler()
	if reconciler == nil {
		t.Fatal("GetReconciler() returned nil")
	}

	if reconciler.adaptersStore != store {
		t.Error("GetReconciler() returned reconciler with different adaptersStore")
	}
}
