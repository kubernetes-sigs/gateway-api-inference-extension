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

package handlers

import (
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

func TestHandleRequestHeaders(t *testing.T) {
	t.Parallel()

	// Setup a mock server and request context
	server := &StreamingServer{
		fairnessIDHeaderKey: "test-fairness-id",
	}

	reqCtx := &RequestContext{
		Request: &Request{
			Headers: make(map[string]string),
		},
	}

	req := &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &configPb.HeaderMap{
				Headers: []*configPb.HeaderValue{
					{
						Key:   "x-test-header",
						Value: "test-value",
					},
					{
						Key:   "test-fairness-id",
						Value: "test-fairness-id-value",
					},
				},
			},
			EndOfStream: false,
		},
	}

	err := server.HandleRequestHeaders(reqCtx, req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if reqCtx.FairnessID != "test-fairness-id-value" {
		t.Errorf("expected fairness ID to be 'test-fairness-id-value', got %s", reqCtx.FairnessID)
	}
	if reqCtx.Request.Headers["test-fairness-id"] == "test-fairness-id-value" {
		t.Errorf("expected fairness ID header to be removed from request headers, but it was not")
	}
}
