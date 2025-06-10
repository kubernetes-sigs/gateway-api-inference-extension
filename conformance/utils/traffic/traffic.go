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

package traffic

import (
	"net/http"
	"testing"

	gwconfig "sigs.k8s.io/gateway-api/conformance/utils/config"
	gwhttp "sigs.k8s.io/gateway-api/conformance/utils/http"
	"sigs.k8s.io/gateway-api/conformance/utils/roundtripper"
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
	return BuildExpectedHTTPResponseWithHeaders(requestHost, requestPath, expectedStatusCode, backendName, backendNamespace, nil)
}

func BuildExpectedHTTPResponseWithHeaders(
	requestHost string,
	requestPath string,
	expectedStatusCode int,
	backendName string,
	backendNamespace string,
	headers map[string]string,
) gwhttp.ExpectedResponse {
	resp := gwhttp.ExpectedResponse{
		Request: gwhttp.Request{
			Host:    requestHost,
			Path:    requestPath,
			Method:  "GET",
			Headers: headers,
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
				Host:    requestHost,
				Path:    requestPath,
				Headers: headers,
				Method:  "GET",
			},
		}
	}
	return resp
}

// MakeRequestAndExpectSuccess is a helper function that builds an expected success (200 OK) response
// and then calls MakeRequestAndExpectEventuallyConsistentResponse.
func MakeRequestAndExpectSuccess(
	t *testing.T,
	r roundtripper.RoundTripper,
	timeoutConfig gwconfig.TimeoutConfig,
	gatewayAddress string,
	requestHost string,
	requestPath string,
	backendName string,
	backendNamespace string,
) {
	t.Helper()
	expectedResponse := BuildExpectedHTTPResponse(
		requestHost,
		requestPath,
		http.StatusOK,
		backendName,
		backendNamespace,
	)
	gwhttp.MakeRequestAndExpectEventuallyConsistentResponse(t, r, timeoutConfig, gatewayAddress, expectedResponse)
}

// MakeRequestAndExpectSuccess is a helper function that builds an expected success (200 OK) response
// and then calls MakeRequestAndExpectEventuallyConsistentResponse.
func MakeRequestAndExpectSuccessV2(
	t *testing.T,
	r roundtripper.RoundTripper,
	timeoutConfig gwconfig.TimeoutConfig,
	gatewayAddress string,
	requestHost string,
	requestPath string,
	backendName string,
	backendNamespace string,
	headers map[string]string,
) {
	t.Helper()
	expectedResponse := BuildExpectedHTTPResponseWithHeaders(
		requestHost,
		requestPath,
		http.StatusOK,
		backendName,
		backendNamespace,
		headers,
	)
	expectedResponse.BackendSetResponseHeaders = map[string]string{"Hello": "world!"}
	gwhttp.MakeRequestAndExpectEventuallyConsistentResponse(t, r, timeoutConfig, gatewayAddress, expectedResponse)
}

// MakeRequestAndExpectNotFound is a helper function that builds an expected not found (404) response
// and then calls MakeRequestAndExpectEventuallyConsistentResponse.
func MakeRequestAndExpectNotFound(
	t *testing.T,
	r roundtripper.RoundTripper,
	timeoutConfig gwconfig.TimeoutConfig,
	gatewayAddress string,
	requestHost string,
	requestPath string,
) {
	t.Helper()
	expectedResponse := BuildExpectedHTTPResponse(
		requestHost,
		requestPath,
		http.StatusNotFound,
		"", // Backend name not relevant for 404
		"", // Backend namespace not relevant for 404
	)
	gwhttp.MakeRequestAndExpectEventuallyConsistentResponse(t, r, timeoutConfig, gatewayAddress, expectedResponse)
}
