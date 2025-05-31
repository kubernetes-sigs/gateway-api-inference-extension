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

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

const (
	RequestIdHeaderKey = "x-request-id"
)

func ExtractHeaderValue(req *extProcPb.ProcessingRequest_RequestHeaders, headerKey string) string {
	// header key should be case insensitive
	headerKeyInLower := strings.ToLower(headerKey)
	if req != nil && req.RequestHeaders != nil && req.RequestHeaders.Headers != nil {
		for _, headerKv := range req.RequestHeaders.Headers.Headers {
			if strings.ToLower(headerKv.Key) == headerKeyInLower {
				return string(headerKv.RawValue)
			}
		}
	}
	return ""
}
