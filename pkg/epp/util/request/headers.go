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

	"k8s.io/apimachinery/pkg/util/sets"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
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
