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

package request

import (
	"strings"
	"testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
)

func TestIsSystemOwnedHeader(t *testing.T) {
	tests := []struct {
		name     string
		key      string
		expected bool
	}{
		// InputControlHeaders — exact case (already lowercase)
		{
			name:     "FlowFairnessIDKey exact",
			key:      metadata.FlowFairnessIDKey,
			expected: true,
		},
		{
			name:     "ObjectiveKey exact",
			key:      metadata.ObjectiveKey,
			expected: true,
		},
		{
			name:     "ModelNameRewriteKey exact",
			key:      metadata.ModelNameRewriteKey,
			expected: true,
		},
		{
			name:     "SubsetFilterKey exact",
			key:      metadata.SubsetFilterKey,
			expected: true,
		},
		// InputControlHeaders — uppercase input (case-insensitive matching)
		{
			name:     "FlowFairnessIDKey uppercase",
			key:      strings.ToUpper(metadata.FlowFairnessIDKey),
			expected: true,
		},
		{
			name:     "ObjectiveKey uppercase",
			key:      strings.ToUpper(metadata.ObjectiveKey),
			expected: true,
		},
		// OutputInjectionHeaders — exact case
		{
			name:     "DestinationEndpointKey exact",
			key:      metadata.DestinationEndpointKey,
			expected: true,
		},
		{
			name:     "DestinationEndpointServedKey exact",
			key:      metadata.DestinationEndpointServedKey,
			expected: true,
		},
		// OutputInjectionHeaders — uppercase input
		{
			name:     "DestinationEndpointKey uppercase",
			key:      strings.ToUpper(metadata.DestinationEndpointKey),
			expected: true,
		},
		{
			name:     "DestinationEndpointServedKey uppercase",
			key:      strings.ToUpper(metadata.DestinationEndpointServedKey),
			expected: true,
		},
		// ProtocolHeaders
		{
			name:     "content-length lowercase",
			key:      "content-length",
			expected: true,
		},
		{
			name:     "Content-Length canonical HTTP form",
			key:      "Content-Length",
			expected: true,
		},
		{
			name:     "CONTENT-LENGTH uppercase",
			key:      "CONTENT-LENGTH",
			expected: true,
		},
		// Non-system headers
		{
			name:     "arbitrary custom header",
			key:      "x-custom-header",
			expected: false,
		},
		{
			name:     "authorization header",
			key:      "authorization",
			expected: false,
		},
		{
			name:     "content-type header",
			key:      "content-type",
			expected: false,
		},
		{
			name:     "empty string",
			key:      "",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsSystemOwnedHeader(tt.key)
			if result != tt.expected {
				t.Errorf("IsSystemOwnedHeader(%q) = %v, want %v", tt.key, result, tt.expected)
			}
		})
	}
}

func TestInputControlHeadersContents(t *testing.T) {
	expected := []string{
		strings.ToLower(metadata.FlowFairnessIDKey),
		strings.ToLower(metadata.ObjectiveKey),
		strings.ToLower(metadata.ModelNameRewriteKey),
		strings.ToLower(metadata.SubsetFilterKey),
	}
	for _, key := range expected {
		if !InputControlHeaders.Has(key) {
			t.Errorf("InputControlHeaders missing expected key %q", key)
		}
	}
	if got, want := InputControlHeaders.Len(), len(expected); got != want {
		t.Errorf("InputControlHeaders length = %d, want %d", got, want)
	}
}

func TestOutputInjectionHeadersContents(t *testing.T) {
	expected := []string{
		strings.ToLower(metadata.DestinationEndpointKey),
		strings.ToLower(metadata.DestinationEndpointServedKey),
	}
	for _, key := range expected {
		if !OutputInjectionHeaders.Has(key) {
			t.Errorf("OutputInjectionHeaders missing expected key %q", key)
		}
	}
	if got, want := OutputInjectionHeaders.Len(), len(expected); got != want {
		t.Errorf("OutputInjectionHeaders length = %d, want %d", got, want)
	}
}

func TestProtocolHeadersContents(t *testing.T) {
	expected := []string{"content-length"}
	for _, key := range expected {
		if !ProtocolHeaders.Has(key) {
			t.Errorf("ProtocolHeaders missing expected key %q", key)
		}
	}
	if got, want := ProtocolHeaders.Len(), len(expected); got != want {
		t.Errorf("ProtocolHeaders length = %d, want %d", got, want)
	}
}
