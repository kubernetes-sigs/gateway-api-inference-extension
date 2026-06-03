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

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/types/known/structpb"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/metadata"
)

type mockDatastore struct {
	pods []*datastore.Endpoint
}

func (m *mockDatastore) PoolGet() (*datastore.EndpointPool, error) {
	return nil, nil
}

func (m *mockDatastore) PodList(predicate func(*datastore.Endpoint) bool) []*datastore.Endpoint {
	var res []*datastore.Endpoint
	for _, p := range m.pods {
		if predicate(p) {
			res = append(res, p)
		}
	}
	return res
}

func TestHandleRequestHeaders_RoundRobin(t *testing.T) {
	pods := []*datastore.Endpoint{
		{Address: "10.0.0.1", Port: "8080"},
		{Address: "10.0.0.2", Port: "8080"},
	}
	ds := &mockDatastore{pods: pods}
	server := NewStreamingServer(ds)

	req := &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &envoyCorev3.HeaderMap{},
		},
	}

	// First request
	reqCtx1 := &RequestContext{}
	err := server.handleRequestHeaders(context.Background(), reqCtx1, nil, req)
	assert.NoError(t, err)
	err = server.pickEndpoint(context.Background(), reqCtx1, nil)
	assert.NoError(t, err)

	// Second request
	reqCtx2 := &RequestContext{}
	err = server.handleRequestHeaders(context.Background(), reqCtx2, nil, req)
	assert.NoError(t, err)
	err = server.pickEndpoint(context.Background(), reqCtx2, nil)
	assert.NoError(t, err)

	// They should be different pods (round-robin)
	assert.NotEqual(t, reqCtx1.SelectedPodIP, reqCtx2.SelectedPodIP)

	// Third request should wrap around
	reqCtx3 := &RequestContext{}
	err = server.handleRequestHeaders(context.Background(), reqCtx3, nil, req)
	assert.NoError(t, err)
	err = server.pickEndpoint(context.Background(), reqCtx3, nil)
	assert.NoError(t, err)
	assert.Equal(t, reqCtx1.SelectedPodIP, reqCtx3.SelectedPodIP)
}

func TestHandleRequestHeaders_FilteringViaHeader(t *testing.T) {
	pods := []*datastore.Endpoint{
		{Address: "10.0.0.1", Port: "8080"},
		{Address: "10.0.0.2", Port: "8080"},
		{Address: "10.0.0.3", Port: "8080"},
	}
	ds := &mockDatastore{pods: pods}
	server := NewStreamingServer(ds)

	req := &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &envoyCorev3.HeaderMap{
				Headers: []*envoyCorev3.HeaderValue{
					{Key: "test-epp-endpoint-selection", Value: "10.0.0.2"},
				},
			},
		},
	}

	reqCtx := &RequestContext{}
	err := server.handleRequestHeaders(context.Background(), reqCtx, nil, req)
	assert.NoError(t, err)
	err = server.pickEndpoint(context.Background(), reqCtx, nil)
	assert.NoError(t, err)
	assert.Equal(t, "10.0.0.2", reqCtx.SelectedPodIP)
}

func TestHandleRequestHeaders_NoPods(t *testing.T) {
	ds := &mockDatastore{pods: []*datastore.Endpoint{}}
	server := NewStreamingServer(ds)

	req := &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &envoyCorev3.HeaderMap{},
		},
	}

	reqCtx := &RequestContext{}
	err := server.handleRequestHeaders(context.Background(), reqCtx, nil, req)
	assert.Error(t, err)
}

func TestHandleRequestHeaders_FilteringViaFilterMetadata(t *testing.T) {
	pods := []*datastore.Endpoint{
		{Address: "10.0.0.1", Port: "8080"},
		{Address: "10.0.0.2", Port: "8080"},
		{Address: "10.0.0.3", Port: "8080"},
	}
	ds := &mockDatastore{pods: pods}
	server := NewStreamingServer(ds)

	fullReq := &extProcPb.ProcessingRequest{
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: map[string]*structpb.Struct{
				metadata.SubsetFilterNamespace: {
					Fields: map[string]*structpb.Value{
						metadata.SubsetFilterKey: structpb.NewListValue(&structpb.ListValue{
							Values: []*structpb.Value{
								structpb.NewStringValue("10.0.0.3:8080"),
							},
						}),
					},
				},
			},
		},
	}
	req := &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &envoyCorev3.HeaderMap{},
		},
	}

	reqCtx := &RequestContext{}
	err := server.handleRequestHeaders(context.Background(), reqCtx, fullReq, req)
	assert.NoError(t, err)
	err = server.pickEndpoint(context.Background(), reqCtx, nil)
	assert.NoError(t, err)
	assert.Equal(t, "10.0.0.3", reqCtx.SelectedPodIP)
}

func TestHandleRequestHeaders_FilteringViaFilterMetadata_StringValue(t *testing.T) {
	pods := []*datastore.Endpoint{
		{Address: "10.0.0.1", Port: "8080"},
		{Address: "10.0.0.2", Port: "8080"},
		{Address: "10.0.0.3", Port: "8080"},
	}
	ds := &mockDatastore{pods: pods}
	server := NewStreamingServer(ds)

	fullReq := &extProcPb.ProcessingRequest{
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: map[string]*structpb.Struct{
				metadata.SubsetFilterNamespace: {
					Fields: map[string]*structpb.Value{
						metadata.SubsetFilterKey: structpb.NewStringValue("10.0.0.2:8080,10.0.0.3:8080"),
					},
				},
			},
		},
	}
	req := &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &envoyCorev3.HeaderMap{},
		},
	}

	reqCtx := &RequestContext{}
	err := server.handleRequestHeaders(context.Background(), reqCtx, fullReq, req)
	assert.NoError(t, err)

	// Candidates resolved should contain only 10.0.0.2 and 10.0.0.3
	assert.Len(t, reqCtx.Candidates, 2)
	assert.ElementsMatch(t, []string{"10.0.0.2", "10.0.0.3"}, []string{reqCtx.Candidates[0].Address, reqCtx.Candidates[1].Address})
}

func TestHandleRequestHeaders_HeaderTakesPrecedenceOverMetadata(t *testing.T) {
	pods := []*datastore.Endpoint{
		{Address: "10.0.0.2", Port: "8080"},
		{Address: "10.0.0.3", Port: "8080"},
	}
	ds := &mockDatastore{pods: pods}
	server := NewStreamingServer(ds)

	fullReq := &extProcPb.ProcessingRequest{
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: map[string]*structpb.Struct{
				metadata.SubsetFilterNamespace: {
					Fields: map[string]*structpb.Value{
						metadata.SubsetFilterKey: structpb.NewListValue(&structpb.ListValue{
							Values: []*structpb.Value{
								structpb.NewStringValue("10.0.0.2:8080"),
								structpb.NewStringValue("10.0.0.3:8080"),
							},
						}),
					},
				},
			},
		},
	}
	req := &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &envoyCorev3.HeaderMap{
				Headers: []*envoyCorev3.HeaderValue{
					{Key: "test-epp-endpoint-selection", Value: "10.0.0.2"},
				},
			},
		},
	}

	reqCtx := &RequestContext{}
	err := server.handleRequestHeaders(context.Background(), reqCtx, fullReq, req)
	assert.NoError(t, err)
	err = server.pickEndpoint(context.Background(), reqCtx, nil)
	assert.NoError(t, err)
	// The test header should override metadata-based filtering.
	assert.Equal(t, "10.0.0.2", reqCtx.SelectedPodIP)
}

func TestHandleRequestHeaders_DuplicateHeadersArePreserved(t *testing.T) {
	pods := []*datastore.Endpoint{
		{Address: "10.0.0.1", Port: "8080"},
	}
	ds := &mockDatastore{pods: pods}
	server := NewStreamingServer(ds)

	req := &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &envoyCorev3.HeaderMap{
				Headers: []*envoyCorev3.HeaderValue{
					{Key: "X-Forwarded-For", Value: "1.1.1.1"},
					{Key: "X-Forwarded-For", Value: "2.2.2.2"},
					{Key: "Cookie", Value: "session=abc"},
					{Key: "Cookie", Value: "user=xyz"},
				},
			},
		},
	}

	reqCtx := &RequestContext{}
	err := server.handleRequestHeaders(context.Background(), reqCtx, nil, req)
	assert.NoError(t, err)

	// Duplicate header keys must be preserved in sequence arrays instead of getting overwritten.
	assert.Equal(t, []string{"1.1.1.1", "2.2.2.2"}, reqCtx.Headers["X-Forwarded-For"])
	assert.Equal(t, []string{"session=abc", "user=xyz"}, reqCtx.Headers["Cookie"])
}

func TestHandleRequestHeaders_NoSubsetMetadataReturnsAllPods(t *testing.T) {
	pods := []*datastore.Endpoint{
		{Address: "10.0.0.1", Port: "8080"},
		{Address: "10.0.0.2", Port: "8080"},
	}
	ds := &mockDatastore{pods: pods}
	server := NewStreamingServer(ds)

	req := &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &envoyCorev3.HeaderMap{},
		},
	}

	reqCtx := &RequestContext{}
	err := server.handleRequestHeaders(context.Background(), reqCtx, nil, req)
	assert.NoError(t, err)

	// If no metadata is present on fullReq, all pods should be candidates
	assert.Len(t, reqCtx.Candidates, 2)
	assert.ElementsMatch(t, []string{"10.0.0.1", "10.0.0.2"}, []string{reqCtx.Candidates[0].Address, reqCtx.Candidates[1].Address})
}

func TestHandleRequestHeaders_MetadataNamespaceEmptyReturnsAllPods(t *testing.T) {
	pods := []*datastore.Endpoint{
		{Address: "10.0.0.1", Port: "8080"},
		{Address: "10.0.0.2", Port: "8080"},
	}
	ds := &mockDatastore{pods: pods}
	server := NewStreamingServer(ds)

	fullReq := &extProcPb.ProcessingRequest{
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: map[string]*structpb.Struct{
				metadata.SubsetFilterNamespace: {
					Fields: map[string]*structpb.Value{
						"some-other-unrelated-key": structpb.NewStringValue("val"),
					},
				},
			},
		},
	}

	req := &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &envoyCorev3.HeaderMap{},
		},
	}

	reqCtx := &RequestContext{}
	err := server.handleRequestHeaders(context.Background(), reqCtx, fullReq, req)
	assert.NoError(t, err)

	// If the subset key is missing from the namespace struct, all pods should be candidates
	assert.Len(t, reqCtx.Candidates, 2)
	assert.ElementsMatch(t, []string{"10.0.0.1", "10.0.0.2"}, []string{reqCtx.Candidates[0].Address, reqCtx.Candidates[1].Address})
}

func TestHandleRequestHeaders_SubsetFilterEmptyReturnsNoPods(t *testing.T) {
	pods := []*datastore.Endpoint{
		{Address: "10.0.0.1", Port: "8080"},
		{Address: "10.0.0.2", Port: "8080"},
	}
	ds := &mockDatastore{pods: pods}
	server := NewStreamingServer(ds)

	fullReq := &extProcPb.ProcessingRequest{
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: map[string]*structpb.Struct{
				metadata.SubsetFilterNamespace: {
					Fields: map[string]*structpb.Value{
						metadata.SubsetFilterKey: structpb.NewListValue(&structpb.ListValue{
							Values: []*structpb.Value{},
						}),
					},
				},
			},
		},
	}

	req := &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &envoyCorev3.HeaderMap{},
		},
	}

	reqCtx := &RequestContext{}
	err := server.handleRequestHeaders(context.Background(), reqCtx, fullReq, req)
	assert.NoError(t, err)

	// If the subset list is present but empty, it must strictly restrict candidates to empty (fail closed to 503)
	assert.Empty(t, reqCtx.Candidates)
}

func TestHandleRequestHeaders_StringMetadataMalformedWhitespaceTrimming(t *testing.T) {
	pods := []*datastore.Endpoint{
		{Address: "10.0.0.1", Port: "8080"},
		{Address: "10.0.0.2", Port: "8080"},
		{Address: "10.0.0.3", Port: "8080"},
	}
	ds := &mockDatastore{pods: pods}
	server := NewStreamingServer(ds)

	fullReq := &extProcPb.ProcessingRequest{
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: map[string]*structpb.Struct{
				metadata.SubsetFilterNamespace: {
					Fields: map[string]*structpb.Value{
						metadata.SubsetFilterKey: structpb.NewStringValue("  10.0.0.2:8080 ,  ,  10.0.0.3:8080  "),
					},
				},
			},
		},
	}

	req := &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &envoyCorev3.HeaderMap{},
		},
	}

	reqCtx := &RequestContext{}
	err := server.handleRequestHeaders(context.Background(), reqCtx, fullReq, req)
	assert.NoError(t, err)

	// Trimmed string should result only in pods 2 and 3
	assert.Len(t, reqCtx.Candidates, 2)
	assert.ElementsMatch(t, []string{"10.0.0.2", "10.0.0.3"}, []string{reqCtx.Candidates[0].Address, reqCtx.Candidates[1].Address})
}

func TestHandleRequestHeaders_StringMetadataNoMatchesReturnsNoPods(t *testing.T) {
	pods := []*datastore.Endpoint{
		{Address: "10.0.0.1", Port: "8080"},
		{Address: "10.0.0.2", Port: "8080"},
	}
	ds := &mockDatastore{pods: pods}
	server := NewStreamingServer(ds)

	fullReq := &extProcPb.ProcessingRequest{
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: map[string]*structpb.Struct{
				metadata.SubsetFilterNamespace: {
					Fields: map[string]*structpb.Value{
						metadata.SubsetFilterKey: structpb.NewStringValue("192.168.1.1:8080"), // No match
					},
				},
			},
		},
	}

	req := &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &envoyCorev3.HeaderMap{},
		},
	}

	reqCtx := &RequestContext{}
	err := server.handleRequestHeaders(context.Background(), reqCtx, fullReq, req)
	assert.NoError(t, err)

	// If no pods match a subset filter, it must strictly return zero candidates (fail closed to 503)
	assert.Empty(t, reqCtx.Candidates)
}

func TestHandleRequestHeaders_MixedArrayAndCommaStringElements(t *testing.T) {
	pods := []*datastore.Endpoint{
		{Address: "10.0.0.1", Port: "8080"},
		{Address: "10.0.0.2", Port: "8080"},
		{Address: "10.0.0.3", Port: "8080"},
	}
	ds := &mockDatastore{pods: pods}
	server := NewStreamingServer(ds)

	fullReq := &extProcPb.ProcessingRequest{
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: map[string]*structpb.Struct{
				metadata.SubsetFilterNamespace: {
					Fields: map[string]*structpb.Value{
						metadata.SubsetFilterKey: structpb.NewListValue(&structpb.ListValue{
							Values: []*structpb.Value{
								structpb.NewStringValue("10.0.0.2:8080, 10.0.0.3:8080"), // Mixed format inside list element
							},
						}),
					},
				},
			},
		},
	}

	req := &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &envoyCorev3.HeaderMap{},
		},
	}

	reqCtx := &RequestContext{}
	err := server.handleRequestHeaders(context.Background(), reqCtx, fullReq, req)
	assert.NoError(t, err)

	// Mixed comma element inside list should be correctly resolved to pods 2 and 3
	assert.Len(t, reqCtx.Candidates, 2)
	assert.ElementsMatch(t, []string{"10.0.0.2", "10.0.0.3"}, []string{reqCtx.Candidates[0].Address, reqCtx.Candidates[1].Address})
}
