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
	"testing"

	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

func TestExtractHeaderValue(t *testing.T) {
	tests := []struct {
		name     string
		headers  []*corev3.HeaderValue
		key      string
		expected string
	}{
		{
			name: "Exact match",
			headers: []*corev3.HeaderValue{
				{Key: "x-request-id", RawValue: []byte("123")},
			},
			key:      "x-request-id",
			expected: "123",
		},
		{
			name: "Case-insensitive match",
			headers: []*corev3.HeaderValue{
				{Key: "X-Request-ID", RawValue: []byte("456")},
			},
			key:      "x-request-id",
			expected: "456",
		},
		{
			name: "Non-existent key",
			headers: []*corev3.HeaderValue{
				{Key: "other-header", RawValue: []byte("abc")},
			},
			key:      "x-request-id",
			expected: "",
		},
		{
			name: "String value fallback",
			headers: []*corev3.HeaderValue{
				{Key: "fallback-test", Value: "only-string"},
			},
			key:      "fallback-test",
			expected: "only-string",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &extProcPb.ProcessingRequest_RequestHeaders{
				RequestHeaders: &extProcPb.HttpHeaders{
					Headers: &corev3.HeaderMap{
						Headers: tt.headers,
					},
				},
			}

			result := ExtractHeaderValue(req, tt.key)
			if result != tt.expected {
				t.Errorf("ExtractHeaderValue() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestGetHeaderValue(t *testing.T) {
	tests := []struct {
		name     string
		header   *corev3.HeaderValue
		expected string
	}{
		{
			name: "Prefers RawValue when present",
			header: &corev3.HeaderValue{
				Key:      "x-test",
				RawValue: []byte("raw-content"),
				Value:    "string-content", // Should be ignored
			},
			expected: "raw-content",
		},
		{
			name: "Falls back to Value when RawValue is nil",
			header: &corev3.HeaderValue{
				Key:      "x-test",
				RawValue: nil,
				Value:    "string-content",
			},
			expected: "string-content",
		},
		{
			name: "Falls back to Value when RawValue is empty slice",
			header: &corev3.HeaderValue{
				Key:      "x-test",
				RawValue: []byte{},
				Value:    "string-content",
			},
			expected: "string-content",
		},
		{
			name: "Returns empty if both are empty",
			header: &corev3.HeaderValue{
				Key:      "x-test",
				RawValue: []byte{},
				Value:    "",
			},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetHeaderValue(tt.header)
			if result != tt.expected {
				t.Errorf("GetHeaderValue() = %v, want %v", result, tt.expected)
			}
		})
	}
}
