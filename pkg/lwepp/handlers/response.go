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

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"

	envoy "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/metadata"
)

func (s *StreamingServer) handleResponseHeaders(ctx context.Context, reqCtx *RequestContext, fullReq *extProcPb.ProcessingRequest, respHeaders *extProcPb.ProcessingRequest_ResponseHeaders) *extProcPb.ProcessingResponse {
	logger := log.FromContext(ctx)
	logger.Info("Handling response headers")

	// Read the endpoint that actually served the request from Envoy's response metadata.
	// GKE Gateway sets envoy.lb.x-gateway-destination-endpoint-served after routing to the backend.
	var servedEndpoint string
	respMetadata := envoy.ExtractMetadataValues(fullReq)
	logger.Info("Extracted metadata in response headers", "metadata", respMetadata)
	lbMetadata, ok := respMetadata[metadata.DestinationEndpointNamespace].(map[string]any)
	if !ok {
		servedEndpoint = "fail: missing envoy lb metadata"
	} else if served, ok := lbMetadata[metadata.DestinationEndpointServedKey].(string); !ok {
		servedEndpoint = "fail: missing destination endpoint served metadata"
	} else {
		servedEndpoint = served
	}

	logger.Info("Setting conformance test result header", "header", metadata.ConformanceTestResultHeader, "value", servedEndpoint)

	headers := []*configPb.HeaderValueOption{
		{
			Header: &configPb.HeaderValue{
				Key:      metadata.ConformanceTestResultHeader,
				RawValue: []byte(servedEndpoint),
			},
		},
		{
			Header: &configPb.HeaderValue{
				// This is for debugging purpose only.
				Key:      "x-went-into-resp-headers",
				RawValue: []byte("true"),
			},
		},
	}

	// Promote the endpoint this picker selected to a response header of its own.
	// It is already reflected through the echo server's X-Echo-Set-Header, which
	// stays as it is, but that path needs a backend which implements it. Reading
	// the selection and what actually served as separate headers is what lets a
	// test see a picker whose choice was discarded.
	if reqCtx != nil && reqCtx.TargetEndpoint != "" {
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      metadata.ConformanceTestSelectedHeader,
				RawValue: []byte(reqCtx.TargetEndpoint),
			},
		})
	}

	// Name the pool this picker fronts. A rule that weights traffic across several
	// pools must consult each pool's own picker, and the served endpoint alone
	// cannot show which picker answered: a test would have to infer it from pool
	// membership, which assumes every endpoint belongs to exactly one pool.
	if poolName := s.poolName(); poolName != "" {
		logger.Info("Setting conformance test pool header", "header", metadata.ConformanceTestPoolHeader, "value", poolName)
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      metadata.ConformanceTestPoolHeader,
				RawValue: []byte(poolName),
			},
		})
	}

	// Include any non-system-owned headers from the original response.
	if respHeaders != nil && respHeaders.ResponseHeaders != nil && respHeaders.ResponseHeaders.Headers != nil {
		for _, header := range respHeaders.ResponseHeaders.Headers.Headers {
			key := header.Key
			headers = append(headers, &configPb.HeaderValueOption{
				Header: &configPb.HeaderValue{
					Key:      key,
					RawValue: []byte(envoy.GetHeaderValue(header)),
				},
			})
		}
	}

	resp := &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseHeaders{
			ResponseHeaders: &extProcPb.HeadersResponse{
				Response: &extProcPb.CommonResponse{
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: headers,
					},
				},
			},
		},
	}

	return resp
}

// poolName is the name of the pool this picker fronts, or empty when there is none
// to report: no datastore, a pool that has yet to sync, or one carrying no name.
// Callers omit the header rather than send an empty one, so every one of those
// cases is the same answer.
func (s *StreamingServer) poolName() string {
	if s.datastore == nil {
		return ""
	}
	pool, err := s.datastore.PoolGet()
	if err != nil || pool == nil {
		return ""
	}
	return pool.Name
}
