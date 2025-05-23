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

// Package traffic contains helper functions specifically for generating,
// sending, and validating network traffic related to inference workloads
// within the Gateway API Inference Extension conformance tests.
package traffic

import (
	"net/http"

	gwhttp "sigs.k8s.io/gateway-api/conformance/utils/http"
)

// BuildExpectedHTTPResponse constructs a gwhttp.ExpectedResponse for common test scenarios.
// For 200 OK responses, it sets up ExpectedRequest to check Host and Path.
// For other status codes (like 404), ExpectedRequest is nil as detailed backend checks are usually skipped by CompareRequest.
func BuildExpectedHTTPResponse(
	requestHost string,
	requestPath string,
	expectedStatusCode int,
	backendName string,
	backendNamespace string,
) gwhttp.ExpectedResponse {
	resp := gwhttp.ExpectedResponse{
		Request: gwhttp.Request{
			Host:   requestHost,
			Path:   requestPath,
			Method: "GET",
		},
		Response: gwhttp.Response{
			StatusCode: expectedStatusCode,
		},
		Backend:   backendName,
		Namespace: backendNamespace,
	}

	if expectedStatusCode == http.StatusOK {
		resp.ExpectedRequest = &gwhttp.ExpectedRequest{
			Request: gwhttp.Request{
				Host:   requestHost,
				Path:   requestPath,
				Method: "GET",
			},
		}
	}
	return resp
}
