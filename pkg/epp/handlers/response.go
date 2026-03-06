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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/common"
	reqenvoy "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/request"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

// HandleResponseStream processes response data for both streaming and non-streaming models.
//
// Streaming case:
//
//	Invoked multiple times as data chunks arrive. The final call is identified by
//	endOfStream=true, triggering final metric collection and plugin cleanup.
//
// Non-streaming case:
//
//	Invoked exactly once with endOfStream=true. It processes the entire response
//	body as a single "stream" event.
func (s *StreamingServer) HandleResponseBodyStreaming(ctx context.Context, reqCtx *RequestContext, responseBytes []byte, endOfStream bool) *RequestContext {
	logger := log.FromContext(ctx)
	parsedResp, err := s.parser.ParseResponse(ctx, responseBytes, reqCtx.Response.Headers, endOfStream)
	if err != nil {
		logger.Error(err, "failed to parse response")
	} else if parsedResp != nil && parsedResp.Usage != nil {
		reqCtx.Usage = *parsedResp.Usage
		metrics.RecordInputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.Usage.PromptTokens)
		metrics.RecordOutputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.Usage.CompletionTokens)
		if reqCtx.Usage.PromptTokenDetails != nil {
			metrics.RecordPromptCachedTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.Usage.PromptTokenDetails.CachedTokens)
		}
	}
	if endOfStream {
		metrics.RecordNormalizedTimePerOutputToken(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.RequestReceivedTimestamp, reqCtx.ResponseCompleteTimestamp, reqCtx.Usage.CompletionTokens)
		metrics.RecordRequestLatencies(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.RequestReceivedTimestamp, reqCtx.ResponseCompleteTimestamp)
		metrics.RecordResponseSizes(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.ResponseSize)
	}
	return s.director.HandleResponseBodyStreaming(ctx, reqCtx)
}

func (s *StreamingServer) HandleResponseHeaders(ctx context.Context, reqCtx *RequestContext, resp *extProcPb.ProcessingRequest_ResponseHeaders) *RequestContext {
	for _, header := range resp.ResponseHeaders.Headers.Headers {
		reqCtx.Response.Headers[header.Key] = reqenvoy.GetHeaderValue(header)
	}
	return s.director.HandleResponseReceived(ctx, reqCtx)
}

func (s *StreamingServer) generateResponseHeaderResponse(reqCtx *RequestContext) *extProcPb.ProcessingResponse {
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

func generateResponseBodyResponses(responseBodyBytes []byte, setEoS bool) []*extProcPb.ProcessingResponse {
	commonResponses := common.BuildChunkedBodyResponses(responseBodyBytes, setEoS)
	responses := make([]*extProcPb.ProcessingResponse, 0, len(commonResponses))
	for _, commonResp := range commonResponses {
		resp := &extProcPb.ProcessingResponse{
			Response: &extProcPb.ProcessingResponse_ResponseBody{
				ResponseBody: &extProcPb.BodyResponse{
					Response: commonResp,
				},
			},
		}
		responses = append(responses, resp)
	}
	return responses
}

func (s *StreamingServer) generateResponseHeaders(reqCtx *RequestContext) []*configPb.HeaderValueOption {
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
