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

	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
)

const (
	RequestIdHeaderKey = "x-request-id"
)

var (
	// InputControlHeaders are sent by the Gateway/User to control EPP behavior.
	// We must extract these, then strip them so they don't leak to the backend.
	InputControlHeaders = map[string]bool{
		strings.ToLower(metadata.FlowFairnessIDKey):   true,
		strings.ToLower(metadata.ObjectiveKey):        true,
		strings.ToLower(metadata.ModelNameRewriteKey): true,
		strings.ToLower(metadata.SubsetFilterKey):     true,
	}

	// OutputInjectionHeaders are headers EPP injects for the backend.
	// If the user sends these, they must be stripped to prevent ambiguity.
	OutputInjectionHeaders = map[string]bool{
		strings.ToLower(metadata.DestinationEndpointKey):       true,
		strings.ToLower(metadata.DestinationEndpointServedKey): true,
	}

	// ProtocolHeaders are managed by the proxy layer (Envoy/EPP).
	ProtocolHeaders = map[string]bool{
		"content-length": true,
	}
)

func IsSystemOwnedHeader(key string) bool {
	k := strings.ToLower(key)
	return InputControlHeaders[k] || OutputInjectionHeaders[k] || ProtocolHeaders[k]
}

// GetHeaderValue safely extracts the string value from an Envoy HeaderValue field.
func GetHeaderValue(header *corev3.HeaderValue) string {
	if len(header.RawValue) > 0 {
		return string(header.RawValue)
	}
	return header.Value
}

// ExtractHeaderValue searches for a specific header key in the processing request and returns its value.
// The lookup is case-insensitive.
// Returns an empty string if the header is missing or if the request structure is nil.
func ExtractHeaderValue(req *extProcPb.ProcessingRequest_RequestHeaders, headerKey string) string {
	headerKeyInLower := strings.ToLower(headerKey)
	if req != nil && req.RequestHeaders != nil && req.RequestHeaders.Headers != nil {
		for _, headerKv := range req.RequestHeaders.Headers.Headers {
			if strings.ToLower(headerKv.Key) == headerKeyInLower {
				return GetHeaderValue(headerKv)
			}
		}
	}
	return ""
}
