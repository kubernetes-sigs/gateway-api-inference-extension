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
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"regexp"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	gwconfig "sigs.k8s.io/gateway-api/conformance/utils/config"
	gwhttp "sigs.k8s.io/gateway-api/conformance/utils/http"
	"sigs.k8s.io/gateway-api/conformance/utils/roundtripper"
	"sigs.k8s.io/gateway-api/conformance/utils/tlog"
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

// MakeRequestAndExpectSuccess is a convenience wrapper for requests that are
// expected to succeed with a 200 OK status.
func MakeRequestAndExpectSuccess(
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

	expectedResponse := makeExpectedResponse(t, req)
	waitForConvergeToExpected(t, r, timeoutConfig, gatewayAddress, req.Body, expectedResponse)
}

// MakeRequestAndExpectResponseFromPod sends a request to the specified path by IP address and
// uses a special "test-epp-endpoint-selection" header to target a specific backend Pod.
// It then verifies that the response was served by that Pod.
func MakeRequestAndExpectResponseFromPod(t *testing.T, r roundtripper.RoundTripper, timeoutConfig gwconfig.TimeoutConfig, gwAddr, path string, targetPod *corev1.Pod) {
	t.Helper()

	const (
		eppSelectionHeader = "test-epp-endpoint-selection"
		backendPort        = 3000
	)

	expectedResponse := gwhttp.ExpectedResponse{
		Request: gwhttp.Request{
			Path: path,
			Headers: map[string]string{
				eppSelectionHeader: fmt.Sprintf("%s:%d", targetPod.Status.PodIP, backendPort),
			},
		},
		Backend:   targetPod.Name,
		Namespace: targetPod.Namespace,
	}

	gwhttp.MakeRequestAndExpectEventuallyConsistentResponse(t, r, timeoutConfig, gwAddr, expectedResponse)
}

func makeExpectedResponse(t *testing.T, req Request) gwhttp.ExpectedResponse {
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
	return expectedResponse
}

// TODO: https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/1031
// replace the following method when sigs.k8s.io/gateway-api/conformance/utils/roundtripper is able to send request with body.
func waitForConvergeToExpected(
	t *testing.T,
	r roundtripper.RoundTripper,
	timeoutConfig gwconfig.TimeoutConfig,
	gatewayAddress string,
	requestBody string,
	expectedResponse gwhttp.ExpectedResponse,
) {
	gwhttp.AwaitConvergence(t, timeoutConfig.RequiredConsecutiveSuccesses, timeoutConfig.MaxTimeToConsistency, func(elapsed time.Duration) bool {
		req := gwhttp.MakeRequest(t, &expectedResponse, gatewayAddress, "HTTP", "http")
		request := &RequestWithBody{Request: req}
		if requestBody != "" {
			request = &RequestWithBody{Request: req, Body: strings.NewReader(requestBody)}
		}
		cReq, cRes, err := MakeCallRoundTripper(t, r, request)
		if err != nil {
			tlog.Logf(t, "Request failed, not ready yet: %v (after %v)", err.Error(), elapsed)
			return false
		}

		if err := gwhttp.CompareRequest(t, &request.Request, cReq, cRes, expectedResponse); err != nil {
			tlog.Logf(t, "Response expectation failed for request: %+v  not ready yet: %v (after %v)", request.Request, err, elapsed)
			return false
		}

		return true
	})
	tlog.Logf(t, "Request passed")
}

// TODO: https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/1031
// remove this when sigs.k8s.io/gateway-api/conformance/utils/roundtripper is able to send request with body.
// RequestWithBody extends roundtripper.Request to include a request body.
type RequestWithBody struct {
	roundtripper.Request
	Body io.Reader
}

// TODO: https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/1031
// remove this when sigs.k8s.io/gateway-api/conformance/utils/roundtripper is able to send request with body.
// MakeCallRoundTripper executes an HTTP request using the provided RoundTripper and captures the request and response.
func MakeCallRoundTripper(t *testing.T, r roundtripper.RoundTripper, request *RequestWithBody) (*roundtripper.CapturedRequest, *roundtripper.CapturedResponse, error) {
	client := &http.Client{}

	defaultRoundTripper, ok := r.(*roundtripper.DefaultRoundTripper)
	if !ok {
		t.Fatalf("Unsupported RoundTripper type: %T", r)
	}
	rt := defaultRoundTripper
	if request.UnfollowRedirect {
		client.CheckRedirect = func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		}
	}

	client.Transport = &http.Transport{
		DialContext: rt.CustomDialContext,
		// We disable keep-alives so that we don't leak established TCP connections.
		// Leaking TCP connections is bad because we could eventually hit the
		// threshold of maximum number of open TCP connections to a specific
		// destination. Keep-alives are not presently utilized so disabling this has
		// no adverse affect.
		//
		// Ref. https://github.com/kubernetes-sigs/gateway-api/issues/2357
		DisableKeepAlives: true,
	}

	method := "GET"
	if request.Method != "" {
		method = request.Method
	}
	ctx, cancel := context.WithTimeout(context.Background(), rt.TimeoutConfig.RequestTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, method, request.URL.String(), request.Body)
	if err != nil {
		return nil, nil, err
	}

	if request.Host != "" {
		req.Host = request.Host
	}

	if request.Headers != nil {
		for name, value := range request.Headers {
			req.Header.Set(name, value[0])
		}
	}

	if rt.Debug {
		var dump []byte
		dump, err = httputil.DumpRequestOut(req, true)
		if err != nil {
			return nil, nil, err
		}

		tlog.Logf(request.T, "Sending Request:\n%s\n\n", formatDump(dump, "< "))
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	if rt.Debug {
		var dump []byte
		dump, err = httputil.DumpResponse(resp, true)
		if err != nil {
			return nil, nil, err
		}

		tlog.Logf(request.T, "Received Response:\n%s\n\n", formatDump(dump, "< "))
	}

	cReq := &roundtripper.CapturedRequest{}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, err
	}

	// we cannot assume the response is JSON
	if resp.Header.Get("Content-type") == "application/json" {
		err = json.Unmarshal(body, cReq)
		if err != nil {
			return nil, nil, fmt.Errorf("unexpected error reading response: %w", err)
		}
	} else {
		cReq.Method = method // assume it made the right request if the service being called isn't echoing
	}

	cRes := &roundtripper.CapturedResponse{
		StatusCode:    resp.StatusCode,
		ContentLength: resp.ContentLength,
		Protocol:      resp.Proto,
		Headers:       resp.Header,
	}

	if resp.TLS != nil {
		cRes.PeerCertificates = resp.TLS.PeerCertificates
	}

	if roundtripper.IsRedirect(resp.StatusCode) {
		redirectURL, err := resp.Location()
		if err != nil {
			return nil, nil, err
		}
		cRes.RedirectRequest = &roundtripper.RedirectRequest{
			Scheme: redirectURL.Scheme,
			Host:   redirectURL.Hostname(),
			Port:   redirectURL.Port(),
			Path:   redirectURL.Path,
		}
	}

	return cReq, cRes, nil
}

var startLineRegex = regexp.MustCompile(`(?m)^`)

func formatDump(data []byte, prefix string) string {
	data = startLineRegex.ReplaceAllLiteral(data, []byte(prefix))
	return string(data)
}
