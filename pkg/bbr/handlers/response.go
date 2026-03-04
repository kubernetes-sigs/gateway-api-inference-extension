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
	"encoding/json"
	"fmt"
	"strconv"

	basepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	eppb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/common"
	reqenvoy "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/request"
)

// HandleResponseHeaders extracts response headers into reqCtx and returns
// the ext-proc header response.
func (s *Server) HandleResponseHeaders(reqCtx *RequestContext, headers *eppb.HttpHeaders) ([]*eppb.ProcessingResponse, error) {
	if headers != nil && headers.Headers != nil {
		for _, header := range headers.Headers.Headers {
			reqCtx.Response.Headers[header.Key] = reqenvoy.GetHeaderValue(header)
		}
	}

	return []*eppb.ProcessingResponse{
		{
			Response: &eppb.ProcessingResponse_ResponseHeaders{
				ResponseHeaders: &eppb.HeadersResponse{},
			},
		},
	}, nil
}

// HandleResponseBody handles response bodies by executing response plugins in order.
func (s *Server) HandleResponseBody(ctx context.Context, reqCtx *RequestContext, responseBodyBytes []byte) ([]*eppb.ProcessingResponse, error) {
	logger := log.FromContext(ctx)
	if len(s.responsePlugins) == 0 {
		return []*eppb.ProcessingResponse{
			{
				Response: &eppb.ProcessingResponse_ResponseBody{
					ResponseBody: &eppb.BodyResponse{},
				},
			},
		}, nil
	}

	var responseBody map[string]any
	if err := json.Unmarshal(responseBodyBytes, &responseBody); err != nil {
		logger.Error(err, "Failed to parse response body as JSON, skipping response plugins")
		return []*eppb.ProcessingResponse{
			{
				Response: &eppb.ProcessingResponse_ResponseBody{
					ResponseBody: &eppb.BodyResponse{},
				},
			},
		}, nil
	}

	updatedHeaders, updatedBody, err := s.executePlugins(ctx, reqCtx.Response.Headers, responseBody, s.responsePlugins)
	if err != nil {
		logger.Error(err, "Response plugin execution failed")
		return nil, fmt.Errorf("failed to execute response plugins - %w", err)
	}

	mutatedBytes, err := json.Marshal(updatedBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal mutated response body - %w", err)
	}

	headerMutation := buildResponseHeaderMutation(updatedHeaders, len(mutatedBytes))

	if s.streaming {
		return addStreamedResponseBodyResponse(mutatedBytes, headerMutation), nil
	}

	return []*eppb.ProcessingResponse{
		{
			Response: &eppb.ProcessingResponse_ResponseBody{
				ResponseBody: &eppb.BodyResponse{
					Response: &eppb.CommonResponse{
						HeaderMutation: headerMutation,
						BodyMutation: &eppb.BodyMutation{
							Mutation: &eppb.BodyMutation_Body{
								Body: mutatedBytes,
							},
						},
					},
				},
			},
		},
	}, nil
}

// buildResponseHeaderMutation creates a HeaderMutation from the plugin-updated
// headers and sets Content-Length to match the new body size.
func buildResponseHeaderMutation(headers map[string]string, bodyLen int) *eppb.HeaderMutation {
	setHeaders := make([]*basepb.HeaderValueOption, 0, len(headers)+1)
	for k, v := range headers {
		setHeaders = append(setHeaders, &basepb.HeaderValueOption{
			Header: &basepb.HeaderValue{
				Key:      k,
				RawValue: []byte(v),
			},
		})
	}
	setHeaders = append(setHeaders, &basepb.HeaderValueOption{
		Header: &basepb.HeaderValue{
			Key:      "content-length",
			RawValue: []byte(strconv.Itoa(bodyLen)),
		},
	})
	return &eppb.HeaderMutation{SetHeaders: setHeaders}
}

// addStreamedResponseBodyResponse builds chunked streaming responses for
// the response body path, attaching header mutations to the first chunk.
func addStreamedResponseBodyResponse(bodyBytes []byte, headerMutation *eppb.HeaderMutation) []*eppb.ProcessingResponse {
	commonResponses := common.BuildChunkedBodyResponses(bodyBytes, true)
	responses := make([]*eppb.ProcessingResponse, 0, len(commonResponses))
	for i, commonResp := range commonResponses {
		if i == 0 {
			commonResp.HeaderMutation = headerMutation
		}
		responses = append(responses, &eppb.ProcessingResponse{
			Response: &eppb.ProcessingResponse_ResponseBody{
				ResponseBody: &eppb.BodyResponse{
					Response: commonResp,
				},
			},
		})
	}
	return responses
}

// HandleResponseTrailers handles response trailers.
func (s *Server) HandleResponseTrailers(trailers *eppb.HttpTrailers) ([]*eppb.ProcessingResponse, error) {
	return []*eppb.ProcessingResponse{
		{
			Response: &eppb.ProcessingResponse_ResponseTrailers{
				ResponseTrailers: &eppb.TrailersResponse{},
			},
		},
	}, nil
}
