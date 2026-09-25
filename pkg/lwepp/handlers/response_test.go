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

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/types/known/structpb"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/metadata"
)

const (
	testPoolName      = "pool"
	testPoolNamespace = "test"
)

func TestHandleResponseHeaders_MissingEnvoyLBMetadata(t *testing.T) {
	server := NewStreamingServer(nil, types.NamespacedName{Namespace: testPoolNamespace, Name: testPoolName})

	resp := server.handleResponseHeaders(t.Context(), nil, nil)

	setHeaders := resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders()
	assert.Len(t, setHeaders, 3)
	assert.Equal(t, metadata.ConformanceTestResultHeader, setHeaders[0].GetHeader().GetKey())
	assert.Equal(t, "fail: missing envoy lb metadata", string(setHeaders[0].GetHeader().GetRawValue()))
	assert.Equal(t, "x-went-into-resp-headers", setHeaders[1].GetHeader().GetKey())
	assert.Equal(t, "true", string(setHeaders[1].GetHeader().GetRawValue()))
}

func TestHandleResponseHeaders_MissingDestinationEndpointServedKey(t *testing.T) {
	server := NewStreamingServer(nil, types.NamespacedName{Namespace: testPoolNamespace, Name: testPoolName})

	fullReq := &extProcPb.ProcessingRequest{
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: map[string]*structpb.Struct{
				metadata.DestinationEndpointNamespace: {
					Fields: map[string]*structpb.Value{
						"some-other-key": structpb.NewStringValue("value"),
					},
				},
			},
		},
	}

	resp := server.handleResponseHeaders(t.Context(), nil, fullReq)

	setHeaders := resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders()
	assert.Len(t, setHeaders, 3)
	assert.Equal(t, metadata.ConformanceTestResultHeader, setHeaders[0].GetHeader().GetKey())
	assert.Equal(t, "fail: missing destination endpoint served metadata", string(setHeaders[0].GetHeader().GetRawValue()))
	assert.Equal(t, "x-went-into-resp-headers", setHeaders[1].GetHeader().GetKey())
	assert.Equal(t, "true", string(setHeaders[1].GetHeader().GetRawValue()))
}

func TestHandleResponseHeaders_UsesServedEndpointFromMetadata(t *testing.T) {
	server := NewStreamingServer(nil, types.NamespacedName{Namespace: testPoolNamespace, Name: testPoolName})

	fullReq := &extProcPb.ProcessingRequest{
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: map[string]*structpb.Struct{
				metadata.DestinationEndpointNamespace: {
					Fields: map[string]*structpb.Value{
						metadata.DestinationEndpointServedKey: structpb.NewStringValue("10.0.0.2:3000"),
					},
				},
			},
		},
	}

	resp := server.handleResponseHeaders(t.Context(), nil, fullReq)

	setHeaders := resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders()
	assert.Len(t, setHeaders, 3)
	assert.Equal(t, metadata.ConformanceTestResultHeader, setHeaders[0].GetHeader().GetKey())
	assert.Equal(t, "10.0.0.2:3000", string(setHeaders[0].GetHeader().GetRawValue()))
	assert.Equal(t, "x-went-into-resp-headers", setHeaders[1].GetHeader().GetKey())
	assert.Equal(t, "true", string(setHeaders[1].GetHeader().GetRawValue()))
}

func TestHandleResponseHeaders_DoesNotOverwriteReportsWithBackendHeaders(t *testing.T) {
	server := NewStreamingServer(nil, types.NamespacedName{Namespace: testPoolNamespace, Name: testPoolName})
	req := &extProcPb.ProcessingRequest{
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: map[string]*structpb.Struct{
				metadata.DestinationEndpointNamespace: {
					Fields: map[string]*structpb.Value{
						metadata.DestinationEndpointServedKey: structpb.NewStringValue("10.0.0.2:3000"),
					},
				},
			},
		},
		Request: &extProcPb.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{
					Headers: []*envoyCorev3.HeaderValue{
						{Key: metadata.ConformanceTestResultHeader, RawValue: []byte("10.0.0.1:3000")},
						{Key: metadata.ConformanceTestSelectedHeader, RawValue: []byte("10.0.0.3:3000")},
						{Key: metadata.ConformanceTestPoolHeader, RawValue: []byte("other/pool")},
						{Key: "x-custom-header", RawValue: []byte("custom-value")},
					},
				},
			},
		},
	}
	resp := server.handleResponseHeaders(t.Context(), &RequestContext{TargetEndpoint: "10.0.0.1:3000"}, req)

	got := map[string]string{}
	for _, h := range resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders() {
		key := h.GetHeader().GetKey()
		assert.NotContains(t, got, key, "each report must be set only once")
		got[key] = string(h.GetHeader().GetRawValue())
	}
	assert.Equal(t, "10.0.0.2:3000", got[metadata.ConformanceTestResultHeader])
	assert.Equal(t, "10.0.0.1:3000", got[metadata.ConformanceTestSelectedHeader])
	assert.Equal(t, "test/pool", got[metadata.ConformanceTestPoolHeader])
	assert.NotContains(t, got, "x-custom-header", "ordinary backend headers need no mutation")
}

func TestResponseHeadersCarryConfiguredPoolBeforeSync(t *testing.T) {
	for _, namespace := range []string{"primary", "secondary"} {
		t.Run(namespace, func(t *testing.T) {
			// No datastore is needed to report the pool supplied at startup.
			server := NewStreamingServer(nil, types.NamespacedName{Namespace: namespace, Name: testPoolName})
			resp := server.handleResponseHeaders(t.Context(), nil, nil)
			got := map[string]string{}
			for _, h := range resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders() {
				got[h.GetHeader().GetKey()] = string(h.GetHeader().GetRawValue())
			}
			assert.Equal(t, namespace+"/pool", got[metadata.ConformanceTestPoolHeader])
			assert.NotContains(t, got, metadata.ConformanceTestSelectedHeader)
		})
	}
}

func TestResponseHeadersWithoutSelection(t *testing.T) {
	server := NewStreamingServer(nil, types.NamespacedName{Namespace: testPoolNamespace, Name: testPoolName})
	resp := server.handleResponseHeaders(t.Context(), &RequestContext{}, nil)
	for _, h := range resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders() {
		assert.NotEqual(t, metadata.ConformanceTestSelectedHeader, h.GetHeader().GetKey())
	}
}
