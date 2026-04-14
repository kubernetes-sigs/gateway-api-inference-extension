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
	"google.golang.org/protobuf/types/known/structpb"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"

	envoy "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

// HandleResponseBody processes response data for both streaming and non-streaming models.
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
func (s *StreamingServer) HandleResponseBody(ctx context.Context, reqCtx *RequestContext, responseBytes []byte, endOfStream bool) *RequestContext {
	logger := log.FromContext(ctx)
	logger.V(logutil.DEBUG).Info("HandleResponseBody is triggered", "len(responseBytes)", len(responseBytes), "endOfStream", endOfStream)

	reqCtx.ResponseSize += len(responseBytes)

	parsedResp, err := s.parser.ParseResponse(ctx, responseBytes, reqCtx.Response.Headers, endOfStream)
	if err != nil {
		logger.Error(err, "parsing response")
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
	return s.director.HandleResponseBody(ctx, reqCtx, endOfStream)
}

func (s *StreamingServer) HandleResponseHeaders(ctx context.Context, reqCtx *RequestContext, resp *extProcPb.ProcessingRequest_ResponseHeaders) *RequestContext {
	for _, header := range resp.ResponseHeaders.Headers.Headers {
		reqCtx.Response.Headers[header.Key] = envoy.GetHeaderValue(header)
	}
	return s.director.HandleResponseHeader(ctx, reqCtx)
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

func generateResponseBodyResponses(responseBodyBytes []byte, setEoS bool, dynamicMetadata *structpb.Struct) []*extProcPb.ProcessingResponse {
	commonResponses := envoy.BuildChunkedBodyResponses(responseBodyBytes, setEoS)
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

	// Attach dynamic metadata to the last response if available.
	if len(responses) > 0 && dynamicMetadata != nil {
		responses[len(responses)-1].DynamicMetadata = dynamicMetadata
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
