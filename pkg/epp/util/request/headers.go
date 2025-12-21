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
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
)

const (
	RequestIdHeaderKey = "x-request-id"
)

var (
	// InputControlHeaders are sent by the Gateway/User to control EPP behavior.
	// We must extract these, then strip them so they don't leak to the backend.
	InputControlHeaders = sets.New(
		strings.ToLower(metadata.FlowFairnessIDKey),
		strings.ToLower(metadata.ObjectiveKey),
		strings.ToLower(metadata.ModelNameRewriteKey),
		strings.ToLower(metadata.SubsetFilterKey),
	)

	// OutputInjectionHeaders are headers EPP injects for the backend.
	// If the user sends these, they must be stripped to prevent ambiguity.
	OutputInjectionHeaders = sets.New(
		strings.ToLower(metadata.DestinationEndpointKey),
		strings.ToLower(metadata.DestinationEndpointServedKey),
	)

	// ProtocolHeaders are managed by the proxy layer (Envoy/EPP).
	ProtocolHeaders = sets.New("content-length")
)

func IsSystemOwnedHeader(key string) bool {
	k := strings.ToLower(key)
	return InputControlHeaders.Has(k) || OutputInjectionHeaders.Has(k) || ProtocolHeaders.Has(k)
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
