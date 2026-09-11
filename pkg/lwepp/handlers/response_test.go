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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/metadata"
)

func TestHandleResponseHeaders_MissingEnvoyLBMetadata(t *testing.T) {
	server := &StreamingServer{}

	resp := server.handleResponseHeaders(t.Context(), nil, nil, &extProcPb.ProcessingRequest_ResponseHeaders{})

	setHeaders := resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders()
	assert.Len(t, setHeaders, 2)
	assert.Equal(t, metadata.ConformanceTestResultHeader, setHeaders[0].GetHeader().GetKey())
	assert.Equal(t, "fail: missing envoy lb metadata", string(setHeaders[0].GetHeader().GetRawValue()))
	assert.Equal(t, "x-went-into-resp-headers", setHeaders[1].GetHeader().GetKey())
	assert.Equal(t, "true", string(setHeaders[1].GetHeader().GetRawValue()))
}

func TestHandleResponseHeaders_MissingDestinationEndpointServedKey(t *testing.T) {
	server := &StreamingServer{}

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

	resp := server.handleResponseHeaders(t.Context(), nil, fullReq, &extProcPb.ProcessingRequest_ResponseHeaders{})

	setHeaders := resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders()
	assert.Len(t, setHeaders, 2)
	assert.Equal(t, metadata.ConformanceTestResultHeader, setHeaders[0].GetHeader().GetKey())
	assert.Equal(t, "fail: missing destination endpoint served metadata", string(setHeaders[0].GetHeader().GetRawValue()))
	assert.Equal(t, "x-went-into-resp-headers", setHeaders[1].GetHeader().GetKey())
	assert.Equal(t, "true", string(setHeaders[1].GetHeader().GetRawValue()))
}

func TestHandleResponseHeaders_UsesServedEndpointFromMetadata(t *testing.T) {
	server := &StreamingServer{}

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

	resp := server.handleResponseHeaders(t.Context(), nil, fullReq, &extProcPb.ProcessingRequest_ResponseHeaders{})

	setHeaders := resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders()
	assert.Len(t, setHeaders, 2)
	assert.Equal(t, metadata.ConformanceTestResultHeader, setHeaders[0].GetHeader().GetKey())
	assert.Equal(t, "10.0.0.2:3000", string(setHeaders[0].GetHeader().GetRawValue()))
	assert.Equal(t, "x-went-into-resp-headers", setHeaders[1].GetHeader().GetKey())
	assert.Equal(t, "true", string(setHeaders[1].GetHeader().GetRawValue()))
}

func TestHandleResponseHeaders_ForwardsOriginalHeaders(t *testing.T) {
	server := &StreamingServer{}

	originalHeaders := &extProcPb.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &extProcPb.HttpHeaders{
			Headers: &envoyCorev3.HeaderMap{
				Headers: []*envoyCorev3.HeaderValue{
					{
						Key:   "x-custom-header",
						Value: "custom-value",
					},
				},
			},
		},
	}

	resp := server.handleResponseHeaders(t.Context(), nil, nil, originalHeaders)

	setHeaders := resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders()
	assert.Len(t, setHeaders, 3)
	assert.Equal(t, metadata.ConformanceTestResultHeader, setHeaders[0].GetHeader().GetKey())
	assert.Equal(t, "x-went-into-resp-headers", setHeaders[1].GetHeader().GetKey())
	assert.Equal(t, "x-custom-header", setHeaders[2].GetHeader().GetKey())
	assert.Equal(t, "custom-value", string(setHeaders[2].GetHeader().GetRawValue()))
}

func TestResponseHeadersCarryThePoolThatPicked(t *testing.T) {
	// given a synced pool, the response names the pool this picker speaks for, so a
	// test spanning several pools can tell which picker answered rather than
	// inferring it from which pool owns the served endpoint
	server := NewStreamingServer(&mockDatastore{pool: &datastore.EndpointPool{Name: "primary-inference-pool"}})
	resp := server.handleResponseHeaders(t.Context(), nil, &extProcPb.ProcessingRequest{}, nil)

	var got string
	for _, h := range resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders() {
		if h.GetHeader().GetKey() == metadata.ConformanceTestPoolHeader {
			got = string(h.GetHeader().GetRawValue())
		}
	}
	assert.Equal(t, "primary-inference-pool", got)

	// before the datastore syncs a pool there is nothing to name, so the header is
	// absent rather than empty and consumers must treat it as optional
	resp = NewStreamingServer(&mockDatastore{}).handleResponseHeaders(t.Context(), nil, &extProcPb.ProcessingRequest{}, nil)
	for _, h := range resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders() {
		assert.NotEqual(t, metadata.ConformanceTestPoolHeader, h.GetHeader().GetKey())
	}
}

func TestResponseHeadersPromoteTheSelectedEndpoint(t *testing.T) {
	// given a picker that chose an endpoint, the choice is reported as a header of
	// its own rather than only through the echo server, so a test can compare it
	// against the endpoint that served without relying on the backend to reflect it
	server := NewStreamingServer(&mockDatastore{pool: &datastore.EndpointPool{Name: "primary-inference-pool"}})
	reqCtx := &RequestContext{TargetEndpoint: "10.0.0.1:8080"}
	resp := server.handleResponseHeaders(t.Context(), reqCtx, &extProcPb.ProcessingRequest{}, nil)

	got := map[string]string{}
	for _, h := range resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders() {
		got[h.GetHeader().GetKey()] = string(h.GetHeader().GetRawValue())
	}
	assert.Equal(t, "10.0.0.1:8080", got[metadata.ConformanceTestSelectedHeader])
	assert.Equal(t, "primary-inference-pool", got[metadata.ConformanceTestPoolHeader])

	// nothing was picked, so there is no selection to report
	resp = server.handleResponseHeaders(t.Context(), &RequestContext{}, &extProcPb.ProcessingRequest{}, nil)
	for _, h := range resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders() {
		assert.NotEqual(t, metadata.ConformanceTestSelectedHeader, h.GetHeader().GetKey())
	}
}
