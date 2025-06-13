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

// Request defines the parameters for a single HTTP test request and its expected outcome.
type Request struct {
	// Host is the hostname to use in the HTTP request.
	Host string
	// Path is the path to request.
	Path string
	// Method is the HTTP method to use. Defaults to "GET" if empty.
	Method string
	// Headers are the HTTP headers to include in the request.
	Headers map[string]string
	// Body is the request body.
	Body string

	// ExpectedStatusCode is the HTTP status code expected in the response.
	ExpectedStatusCode int
	// Backend is the name of the backend service expected to handle the request.
	// This is not checked for non-200 responses.
	Backend string
	// Namespace is the namespace of the backend service.
	Namespace string
}

// Deprecated: this will be replaced by letting caller to construct the Request type.
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

// Deprecated: please use MakeRequestAndExpectSuccessV2 instead.
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

// Deprecated: please use MakeRequestAndExpectEventuallyConsistentResponse instead and specify the ExpectedStatusCode in Request.
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

// MakeRequestAndExpectEventuallyConsistentResponse makes a request using the parameters
// from the Request struct and waits for the response to consistently match the expectations.
func MakeRequestAndExpectEventuallyConsistentResponse(
	t *testing.T,
	r roundtripper.RoundTripper,
	timeoutConfig gwconfig.TimeoutConfig,
	gatewayAddress string,
	req Request,
) {
	t.Helper()

	method := http.MethodGet
	if req.Method != "" {
		method = req.Method
	}

	expectedResponse := gwhttp.ExpectedResponse{
		Request: gwhttp.Request{
			Host:    req.Host,
			Path:    req.Path,
			Method:  method,
			Headers: req.Headers,
			Body:    req.Body,
		},
		Response: gwhttp.Response{
			StatusCode: req.ExpectedStatusCode,
		},
		Backend:   req.Backend,
		Namespace: req.Namespace,
	}

	// For successful responses (200 OK), we also verify that the backend
	// received the request with the correct details (Host, Path, etc.).
	// For other statuses (e.g., 404), this check is skipped.
	if req.ExpectedStatusCode == http.StatusOK {
		expectedResponse.ExpectedRequest = &gwhttp.ExpectedRequest{
			Request: gwhttp.Request{
				Host:    req.Host,
				Path:    req.Path,
				Headers: req.Headers,
				Method:  method,
			},
		}
	}
	gwhttp.MakeRequestAndExpectEventuallyConsistentResponse(t, r, timeoutConfig, gatewayAddress, expectedResponse)
}

// MakeRequestAndExpectSuccessV2 is a convenience wrapper for requests that are
// expected to succeed with a 200 OK status.
func MakeRequestAndExpectSuccessV2(
	t *testing.T,
	r roundtripper.RoundTripper,
	timeoutConfig gwconfig.TimeoutConfig,
	gatewayAddress string,
	req Request,
) {
	t.Helper()
	req.ExpectedStatusCode = http.StatusOK
	MakeRequestAndExpectEventuallyConsistentResponse(t, r, timeoutConfig, gatewayAddress, req)
}
