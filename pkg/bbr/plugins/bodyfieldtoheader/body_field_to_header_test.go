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

package bodyfieldtoheader

import (
	"context"
	"encoding/json"
	"testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
)

const testModelValue = "llama-3"

func TestNewBodyFieldToHeaderPlugin(t *testing.T) {
	tests := []struct {
		name       string
		fieldName  string
		headerName string
		wantErr    bool
	}{
		{
			name:       "valid config",
			fieldName:  "model",
			headerName: "X-Gateway-Model",
		},
		{
			name:       "empty field name",
			fieldName:  "",
			headerName: "X-Gateway-Model",
			wantErr:    true,
		},
		{
			name:       "empty header name",
			fieldName:  "model",
			headerName: "",
			wantErr:    true,
		},
		{
			name:       "both empty",
			fieldName:  "",
			headerName: "",
			wantErr:    true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := NewBodyFieldToHeaderPlugin(tt.fieldName, tt.headerName)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if p.TypedName().Type != BodyFieldToHeaderPluginType {
				t.Errorf("Type = %q, want %q", p.TypedName().Type, BodyFieldToHeaderPluginType)
			}
			if p.TypedName().Name != BodyFieldToHeaderPluginType {
				t.Errorf("Name = %q, want %q", p.TypedName().Name, BodyFieldToHeaderPluginType)
			}
		})
	}
}

func TestBodyFieldToHeaderPlugin_WithName(t *testing.T) {
	p, err := NewBodyFieldToHeaderPlugin("model", "X-Gateway-Model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	p.WithName("custom-name")

	if got := p.TypedName().Name; got != "custom-name" {
		t.Errorf("Name after WithName = %q, want %q", got, "custom-name")
	}
	if got := p.TypedName().Type; got != BodyFieldToHeaderPluginType {
		t.Errorf("Type should be unchanged = %q, want %q", got, BodyFieldToHeaderPluginType)
	}
}

func TestBodyFieldToHeaderPluginFactory(t *testing.T) {
	tests := []struct {
		name       string
		pluginName string
		rawParams  json.RawMessage
		wantErr    bool
		wantName   string
	}{
		{
			name:       "valid config",
			pluginName: "my-plugin",
			rawParams:  json.RawMessage(`{"field_name":"model","header_name":"X-Gateway-Model"}`),
			wantName:   "my-plugin",
		},
		{
			name:       "invalid JSON",
			pluginName: "my-plugin",
			rawParams:  json.RawMessage(`{invalid`),
			wantErr:    true,
		},
		{
			name:       "missing field_name",
			pluginName: "my-plugin",
			rawParams:  json.RawMessage(`{"header_name":"X-Gateway-Model"}`),
			wantErr:    true,
		},
		{
			name:       "missing header_name",
			pluginName: "my-plugin",
			rawParams:  json.RawMessage(`{"field_name":"model"}`),
			wantErr:    true,
		},
		{
			name:       "empty parameters",
			pluginName: "my-plugin",
			rawParams:  json.RawMessage(``),
			wantErr:    true,
		},
		{
			name:       "null parameters",
			pluginName: "my-plugin",
			rawParams:  nil,
			wantErr:    true,
		},
		{
			name:       "JSON null",
			pluginName: "my-plugin",
			rawParams:  json.RawMessage(`null`),
			wantErr:    true,
		},
		{
			name:       "empty JSON object",
			pluginName: "my-plugin",
			rawParams:  json.RawMessage(`{}`),
			wantErr:    true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := BodyFieldToHeaderPluginFactory(tt.pluginName, tt.rawParams, nil)
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
			if got := p.TypedName().Type; got != BodyFieldToHeaderPluginType {
				t.Errorf("Type = %q, want %q", got, BodyFieldToHeaderPluginType)
			}
		})
	}
}

func TestBodyFieldToHeaderPlugin_ProcessRequest(t *testing.T) {
	tests := []struct {
		name       string
		fieldName  string
		headerName string
		request    *framework.InferenceRequest
		wantErr    bool
		wantHeader string
	}{
		{
			name:       "string field value",
			fieldName:  "model",
			headerName: "X-Gateway-Model",
			request: func() *framework.InferenceRequest {
				r := framework.NewInferenceRequest()
				r.Body["model"] = testModelValue
				return r
			}(),
			wantHeader: testModelValue,
		},
		{
			name:       "integer field value",
			fieldName:  "count",
			headerName: "X-Gateway-Count",
			request: func() *framework.InferenceRequest {
				r := framework.NewInferenceRequest()
				r.Body["count"] = 42
				return r
			}(),
			wantHeader: "42",
		},
		{
			name:       "float field value",
			fieldName:  "temperature",
			headerName: "X-Gateway-Temp",
			request: func() *framework.InferenceRequest {
				r := framework.NewInferenceRequest()
				r.Body["temperature"] = 0.7
				return r
			}(),
			wantHeader: "0.7",
		},
		{
			name:       "boolean field value",
			fieldName:  "stream",
			headerName: "X-Gateway-Stream",
			request: func() *framework.InferenceRequest {
				r := framework.NewInferenceRequest()
				r.Body["stream"] = true
				return r
			}(),
			wantHeader: "true",
		},
		{
			name:       "field not found - skips gracefully",
			fieldName:  "missing",
			headerName: "X-Gateway-Missing",
			request: func() *framework.InferenceRequest {
				r := framework.NewInferenceRequest()
				r.Body["other"] = "value"
				return r
			}(),
		},
		{
			name:       "field is empty string - skips gracefully",
			fieldName:  "model",
			headerName: "X-Gateway-Model",
			request: func() *framework.InferenceRequest {
				r := framework.NewInferenceRequest()
				r.Body["model"] = ""
				return r
			}(),
		},
		{
			name:       "nil field value",
			fieldName:  "model",
			headerName: "X-Gateway-Model",
			request: func() *framework.InferenceRequest {
				r := framework.NewInferenceRequest()
				r.Body["model"] = nil
				return r
			}(),
			wantHeader: "<nil>",
		},
		{
			name:       "nil request",
			fieldName:  "model",
			headerName: "X-Gateway-Model",
			request:    nil,
		},
		{
			name:       "nil headers",
			fieldName:  "model",
			headerName: "X-Gateway-Model",
			request: &framework.InferenceRequest{
				InferenceMessage: framework.InferenceMessage{
					Headers: nil,
					Body:    map[string]any{"model": "llama"},
				},
			},
		},
		{
			name:       "nil body",
			fieldName:  "model",
			headerName: "X-Gateway-Model",
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
			p, err := NewBodyFieldToHeaderPlugin(tt.fieldName, tt.headerName)
			if err != nil {
				t.Fatalf("failed to create plugin: %v", err)
			}

			err = p.ProcessRequest(context.Background(), nil, tt.request)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.wantHeader != "" {
				if got := tt.request.Headers[tt.headerName]; got != tt.wantHeader {
					t.Errorf("Headers[%q] = %q, want %q", tt.headerName, got, tt.wantHeader)
				}
			}
		})
	}
}

func TestBodyFieldToHeaderPlugin_ProcessRequest_MutatedHeaders(t *testing.T) {
	p, err := NewBodyFieldToHeaderPlugin("model", "X-Gateway-Model")
	if err != nil {
		t.Fatalf("failed to create plugin: %v", err)
	}

	request := framework.NewInferenceRequest()
	request.Body["model"] = testModelValue

	if err := p.ProcessRequest(context.Background(), nil, request); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	mutated := request.MutatedHeaders()
	if got, ok := mutated["X-Gateway-Model"]; !ok || got != testModelValue {
		t.Errorf("MutatedHeaders[\"X-Gateway-Model\"] = %q, %v; want %q, true", got, ok, testModelValue)
	}
}
