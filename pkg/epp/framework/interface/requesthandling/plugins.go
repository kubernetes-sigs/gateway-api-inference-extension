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

package requesthandling

import (
	"context"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// Parser defines the interface for parsing payload(requests and responses).
type Parser interface {
	fwkplugin.Plugin
	// ParseRequest parses the request body and headers and returns a map representation.
	ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*RequestBody, error)

	// ParseResponse parses the response payload.
	// For streaming responses , this method is invoked multiple times (once per chunk),
	// where 'endOfStream' is set to true only for the final chunk.
	// For non-streaming responses, this method is invoked exactly once with the full
	// buffered response body and 'endOfStream' set to true.
	ParseResponse(ctx context.Context, body []byte, headers map[string]string, endofStream bool) (*ParsedResponse, error)
}

type ParsedResponse struct {
	// Usage is only populate when the raw response has usage.
	Usage *Usage
}
