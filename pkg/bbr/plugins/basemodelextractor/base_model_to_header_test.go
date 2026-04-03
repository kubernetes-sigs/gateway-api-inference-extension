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
	"encoding/json"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	crconfig "sigs.k8s.io/controller-runtime/pkg/config"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	testBaseModel = "qwen3-8b"
	testAdapter   = "finance-adapter"
)

// TestBaseModelToHeaderPlugin_TypedName tests the TypedName method returns correct type and name.
func TestBaseModelToHeaderPlugin_TypedName(t *testing.T) {
	p := &BaseModelToHeaderPlugin{
		typedName:     plugin.TypedName{Type: BaseModelToHeaderPluginType, Name: "test-plugin"},
		AdaptersStore: NewAdaptersStore(),
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
		AdaptersStore: NewAdaptersStore(),
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

// TestBaseModelToHeaderPluginFactory tests the factory function with various configurations.
func TestBaseModelToHeaderPluginFactory(t *testing.T) {
	tests := []struct {
		name       string
		pluginName string
		rawParams  json.RawMessage
		wantErr    bool
		wantName   string
	}{
		{
			name:       "valid empty config",
			pluginName: "my-base-model-plugin",
			rawParams:  json.RawMessage(`{}`),
			wantName:   "my-base-model-plugin",
		},
		{
			name:       "null parameters",
			pluginName: "my-plugin",
			rawParams:  nil,
			wantName:   "my-plugin",
		},
		{
			name:       "JSON null",
			pluginName: "my-plugin",
			rawParams:  json.RawMessage(`null`),
			wantName:   "my-plugin",
		},
		{
			name:       "empty parameters",
			pluginName: "my-plugin",
			rawParams:  json.RawMessage(``),
			wantName:   "my-plugin",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a lightweight manager for testing
			skipValidation := true
			mgr, err := ctrl.NewManager(&rest.Config{Host: "http://dummy:0"}, ctrl.Options{
				Metrics:    metricsserver.Options{BindAddress: "0"},
				Controller: crconfig.Controller{SkipNameValidation: &skipValidation},
			})
			if err != nil {
				t.Fatalf("failed to create test manager: %v", err)
			}

			// Create a handle using the test manager
			handle := framework.NewBbrHandle(context.Background(), mgr)

			p, err := BaseModelToHeaderPluginFactory(tt.pluginName, tt.rawParams, handle)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got := p.TypedName().Name; got != tt.wantName {
				t.Errorf("Name = %q, want %q", got, tt.wantName)
			}
			if got := p.TypedName().Type; got != BaseModelToHeaderPluginType {
				t.Errorf("Type = %q, want %q", got, BaseModelToHeaderPluginType)
			}
		})
	}
}

// TestBaseModelToHeaderPlugin_ProcessRequest tests the ProcessRequest functionality
// with various scenarios specific to base model extraction.
func TestBaseModelToHeaderPlugin_ProcessRequest(t *testing.T) {
	// Setup adaptersStore with test data
	store := NewAdaptersStore()
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cm",
			Namespace: "default",
		},
		Data: map[string]string{
			"baseModel": testBaseModel,
			"adapters":  "- " + testAdapter,
		},
	}
	if err := store.configMapUpdateOrAddIfNotExist(cm); err != nil {
		t.Fatalf("failed to setup adapters store: %v", err)
	}

	p := &BaseModelToHeaderPlugin{
		typedName:     plugin.TypedName{Type: BaseModelToHeaderPluginType, Name: "test"},
		AdaptersStore: store,
	}

	tests := []struct {
		name       string
		request    *framework.InferenceRequest
		wantErr    bool
		wantHeader string
	}{
		{
			name: "base model maps to itself",
			request: func() *framework.InferenceRequest {
				r := framework.NewInferenceRequest()
				r.Body["model"] = testBaseModel
				return r
			}(),
			wantHeader: testBaseModel,
		},
		{
			name: "lora adapter maps to base model",
			request: func() *framework.InferenceRequest {
				r := framework.NewInferenceRequest()
				r.Body["model"] = testAdapter
				return r
			}(),
			wantHeader: testBaseModel,
		},
		{
			name: "unknown model returns empty string",
			request: func() *framework.InferenceRequest {
				r := framework.NewInferenceRequest()
				r.Body["model"] = "unknown-model"
				return r
			}(),
			wantHeader: "",
		},
		{
			name: "missing model field returns empty string",
			request: func() *framework.InferenceRequest {
				r := framework.NewInferenceRequest()
				r.Body["prompt"] = "test prompt"
				return r
			}(),
			wantHeader: "",
		},
		{
			name: "empty string model returns empty string",
			request: func() *framework.InferenceRequest {
				r := framework.NewInferenceRequest()
				r.Body["model"] = ""
				return r
			}(),
			wantHeader: "",
		},
		{
			name: "integer model value",
			request: func() *framework.InferenceRequest {
				r := framework.NewInferenceRequest()
				r.Body["model"] = 123
				return r
			}(),
			wantHeader: "",
		},
		{
			name: "boolean model value",
			request: func() *framework.InferenceRequest {
				r := framework.NewInferenceRequest()
				r.Body["model"] = true
				return r
			}(),
			wantHeader: "",
		},
		{
			name: "nil model value",
			request: func() *framework.InferenceRequest {
				r := framework.NewInferenceRequest()
				r.Body["model"] = nil
				return r
			}(),
			wantHeader: "",
		},
		{
			name:    "nil request",
			request: nil,
		},
		{
			name: "nil headers",
			request: &framework.InferenceRequest{
				InferenceMessage: framework.InferenceMessage{
					Headers: nil,
					Body:    map[string]any{"model": testBaseModel},
				},
			},
		},
		{
			name: "nil body",
			request: &framework.InferenceRequest{
				InferenceMessage: framework.InferenceMessage{
					Headers: map[string]string{},
					Body:    nil,
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := p.ProcessRequest(context.Background(), nil, tt.request)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.wantHeader != "" && tt.request != nil && tt.request.Headers != nil {
				if got := tt.request.Headers[BaseModelHeader]; got != tt.wantHeader {
					t.Errorf("Headers[%q] = %q, want %q", BaseModelHeader, got, tt.wantHeader)
				}
			}
		})
	}
}

// TestBaseModelToHeaderPlugin_ProcessRequest_MutatedHeaders tests that mutated headers
// are properly tracked in the mutated headers map.
func TestBaseModelToHeaderPlugin_ProcessRequest_MutatedHeaders(t *testing.T) {
	// Setup adaptersStore with test data
	store := NewAdaptersStore()
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cm",
			Namespace: "default",
		},
		Data: map[string]string{
			"baseModel": testBaseModel,
			"adapters":  "- " + testAdapter,
		},
	}
	if err := store.configMapUpdateOrAddIfNotExist(cm); err != nil {
		t.Fatalf("failed to setup adapters store: %v", err)
	}

	p := &BaseModelToHeaderPlugin{
		typedName:     plugin.TypedName{Type: BaseModelToHeaderPluginType, Name: "test"},
		AdaptersStore: store,
	}

	request := framework.NewInferenceRequest()
	request.Body["model"] = testAdapter

	if err := p.ProcessRequest(context.Background(), nil, request); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	mutated := request.MutatedHeaders()
	if got, ok := mutated[BaseModelHeader]; !ok || got != testBaseModel {
		t.Errorf("MutatedHeaders()[%q] = %q, %v; want %q, true", BaseModelHeader, got, ok, testBaseModel)
	}
}
