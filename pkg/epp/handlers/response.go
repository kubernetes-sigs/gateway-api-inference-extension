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

	reqenvoy "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/request"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

// HandleResponseBody always returns the requestContext even in the error case, as the request context is used in error handling.
func (s *Server) HandleResponseBody(ctx context.Context, reqCtx *RequestContext, responseBytes []byte) (*RequestContext, error) {
	logger := log.FromContext(ctx)

	parsedResponse, parseErr := s.parser.ParseResponse(ctx, responseBytes, reqCtx.Response.Headers, true)
	if parseErr != nil {
		logger.Error(parseErr, "response parsing")
	}
	if parsedResponse != nil && parsedResponse.Usage != nil {
		reqCtx.Usage = *parsedResponse.Usage
		logger.V(logutil.VERBOSE).Info("Response generated", "usage", reqCtx.Usage)
	}
	return s.director.HandleResponseBodyComplete(ctx, reqCtx)
}

// The function is to handle streaming response if the modelServer is streaming.
func (s *Server) HandleResponseBodyModelStreaming(ctx context.Context, reqCtx *RequestContext, responseBytes []byte, endOfStream bool) {
	logger := log.FromContext(ctx)
	_, err := s.director.HandleResponseBodyStreaming(ctx, reqCtx)
	if err != nil {
		logger.Error(err, "error in HandleResponseBodyStreaming")
	}
	parsedResp, err := s.parser.ParseResponse(ctx, responseBytes, reqCtx.Response.Headers, endOfStream)
	if err != nil {
		logger.Error(err, "streaming response parsing")
	} else if parsedResp != nil && parsedResp.Usage != nil {
		reqCtx.Usage = *parsedResp.Usage
		metrics.RecordInputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.Usage.PromptTokens)
		metrics.RecordOutputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.Usage.CompletionTokens)
		if reqCtx.Usage.PromptTokenDetails != nil {
			metrics.RecordPromptCachedTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.Usage.PromptTokenDetails.CachedTokens)
		}
	}
}

func (s *Server) HandleResponseHeaders(ctx context.Context, reqCtx *RequestContext, resp *extProcPb.ProcessingRequest_ResponseHeaders) (*RequestContext, error) {
	for _, header := range resp.ResponseHeaders.Headers.Headers {
		reqCtx.Response.Headers[header.Key] = reqenvoy.GetHeaderValue(header)
	}

	reqCtx, err := s.director.HandleResponseReceived(ctx, reqCtx)

	return reqCtx, err
}

func (s *Server) generateResponseHeaderResponse(reqCtx *RequestContext) *extProcPb.ProcessingResponse {
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseHeaders{
			ResponseHeaders: &extProcPb.HeadersResponse{
				Response: &extProcPb.CommonResponse{
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: s.generateResponseHeaders(reqCtx),
					},
				},
			},
		},
	}
}

func (s *Server) generateResponseHeaders(reqCtx *RequestContext) []*configPb.HeaderValueOption {
	// can likely refactor these two bespoke headers to be updated in PostDispatch, to centralize logic.
	headers := []*configPb.HeaderValueOption{
		{
			Header: &configPb.HeaderValue{
				// This is for debugging purpose only.
				Key:      "x-went-into-resp-headers",
				RawValue: []byte("true"),
			},
		},
	}

	// Include any non-system-owned headers.
	for key, value := range reqCtx.Response.Headers {
		if request.IsSystemOwnedHeader(key) {
			continue
		}
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      key,
				RawValue: []byte(value),
			},
		})
	}
	return headers
}
