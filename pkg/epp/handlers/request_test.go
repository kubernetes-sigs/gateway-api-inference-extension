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
	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
)

func TestHandleRequestHeaders(t *testing.T) {
	t.Parallel()

	// Setup a mock server and request context
	server := &StreamingServer{}

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
						Key:   metadata.FlowFairnessIDKey,
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
	if reqCtx.Request.Headers[metadata.FlowFairnessIDKey] == "test-fairness-id-value" {
		t.Errorf("expected fairness ID header to be removed from request headers, but it was not")
	}
}

func TestHandleRequestHeaders_DefaultFairnessID(t *testing.T) {
	t.Parallel()

	server := &StreamingServer{}
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
				},
			},
			EndOfStream: false,
		},
	}

	err := server.HandleRequestHeaders(reqCtx, req)
	assert.NoError(t, err, "expected no error")
	assert.Equal(t, defaultFairnessID, reqCtx.FairnessID, "expected fairness ID to be defaulted")
}
