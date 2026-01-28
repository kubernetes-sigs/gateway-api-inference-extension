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
	"context"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/types/known/structpb"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
)

func TestHandleRequestHeaders(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		headers        []*configPb.HeaderValue
		wantHeaders    map[string]string
		wantFairnessID string
	}{
		{
			name: "Extracts Fairness ID and Removes Header",
			headers: []*configPb.HeaderValue{
				{Key: "x-test", Value: "val"},
				{Key: metadata.FlowFairnessIDKey, Value: "user-123"},
			},
			wantHeaders:    map[string]string{"x-test": "val"},
			wantFairnessID: "user-123",
		},
		{
			name: "Prefers RawValue over Value",
			headers: []*configPb.HeaderValue{
				{Key: metadata.FlowFairnessIDKey, RawValue: []byte("binary-id"), Value: "wrong-id"},
			},
			wantFairnessID: "binary-id",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := &StreamingServer{}
			reqCtx := &RequestContext{
				Request: &Request{Headers: make(map[string]string)},
			}
			req := &extProcPb.ProcessingRequest_RequestHeaders{
				RequestHeaders: &extProcPb.HttpHeaders{
					Headers: &configPb.HeaderMap{Headers: tc.headers},
				},
			}

			err := server.HandleRequestHeaders(context.Background(), reqCtx, req)
			assert.NoError(t, err, "HandleRequestHeaders should not return an error")

			assert.Equal(t, tc.wantFairnessID, reqCtx.FairnessID, "FairnessID should match expected value")

			if tc.wantHeaders != nil {
				for k, v := range tc.wantHeaders {
					assert.Equal(t, v, reqCtx.Request.Headers[k], "Header %q should match expected value", k)
				}
			}
		})
	}
}

func TestGenerateHeaders_Sanitization(t *testing.T) {
	server := &StreamingServer{}
	reqCtx := &RequestContext{
		TargetEndpoint: "1.2.3.4:8080",
		RequestSize:    123,
		Request: &Request{
			Headers: map[string]string{
				"x-user-data":                   "important",              // should passthrough
				metadata.ObjectiveKey:           "sensitive-objective-id", // should be stripped
				metadata.DestinationEndpointKey: "1.1.1.1:666",            // should be stripped
				"content-length":                "99999",                  // should be stripped (re-added by logic)
			},
		},
	}

	results := server.generateHeaders(context.Background(), reqCtx)

	gotHeaders := make(map[string]string)
	for _, h := range results {
		gotHeaders[h.Header.Key] = string(h.Header.RawValue)
	}

	assert.Contains(t, gotHeaders, "x-user-data")
	assert.NotContains(t, gotHeaders, metadata.ObjectiveKey)
	assert.Equal(t, "1.2.3.4:8080", gotHeaders[metadata.DestinationEndpointKey])
	assert.Equal(t, "123", gotHeaders["Content-Length"])
}

func TestGenerateRequestHeaderResponse_MergeMetadata(t *testing.T) {
	t.Parallel()

	server := &StreamingServer{}
	reqCtx := &RequestContext{
		TargetEndpoint: "1.2.3.4:8080",
		Request: &Request{
			Headers: make(map[string]string),
		},
		Response: &Response{
			DynamicMetadata: &structpb.Struct{
				Fields: map[string]*structpb.Value{
					"existing_namespace": {
						Kind: &structpb.Value_StructValue{
							StructValue: &structpb.Struct{
								Fields: map[string]*structpb.Value{
									"existing_key": {Kind: &structpb.Value_StringValue{StringValue: "existing_value"}},
								},
							},
						},
					},
				},
			},
		},
	}

	resp := server.generateRequestHeaderResponse(context.Background(), reqCtx)

	// Check that the existing metadata is preserved
	existingNamespace, ok := resp.DynamicMetadata.Fields["existing_namespace"]
	assert.True(t, ok, "Expected existing_namespace to be in DynamicMetadata")
	existingKey, ok := existingNamespace.GetStructValue().Fields["existing_key"]
	assert.True(t, ok, "Expected existing_key to be in existing_namespace")
	assert.Equal(t, "existing_value", existingKey.GetStringValue(), "Unexpected value for existing_key")

	// Check that the new metadata is added
	endpointNamespace, ok := resp.DynamicMetadata.Fields[metadata.DestinationEndpointNamespace]
	assert.True(t, ok, "Expected DestinationEndpointNamespace to be in DynamicMetadata")
	endpointKey, ok := endpointNamespace.GetStructValue().Fields[metadata.DestinationEndpointKey]
	assert.True(t, ok, "Expected DestinationEndpointKey to be in DestinationEndpointNamespace")
	assert.Equal(t, "1.2.3.4:8080", endpointKey.GetStringValue(), "Unexpected value for DestinationEndpointKey")
}
