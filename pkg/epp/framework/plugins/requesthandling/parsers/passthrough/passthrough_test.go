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

package passthrough

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestPassthroughParser_ParseRequest(t *testing.T) {
	parser := NewPassthroughParser()
	ctx := context.Background()

	tests := []struct {
		name     string
		body     []byte
		headers  map[string]string
		wantBody *scheduling.LLMRequestBody
	}{
		{
			name:    "empty body",
			body:    []byte{},
			headers: map[string]string{},
			wantBody: &scheduling.LLMRequestBody{
				Payload: scheduling.RawPayload([]byte{}),
			},
		},
		{
			name:    "non-empty body",
			body:    []byte("hello world"),
			headers: map[string]string{},
			wantBody: &scheduling.LLMRequestBody{
				Payload: scheduling.RawPayload([]byte("hello world")),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parser.ParseRequest(ctx, tt.body, tt.headers)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tt.wantBody, got); diff != "" {
				t.Errorf("Unexpected body (-want +got):\n%s", diff)
			}
		})
	}
}

func TestPassthroughParser_ParseResponse(t *testing.T) {
	parser := NewPassthroughParser()
	ctx := context.Background()

	got, err := parser.ParseResponse(ctx, []byte("hello"), map[string]string{}, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != nil {
		t.Errorf("expected nil result, got %v", got)
	}
}

func TestPassthroughParser_SupportedAppProtocols(t *testing.T) {
	parser := NewPassthroughParser()
	supported := parser.SupportedAppProtocols()
	if len(supported) != 0 {
		t.Errorf("SupportedAppProtocols() = %v, want empty non-nil list", supported)
	}
}
