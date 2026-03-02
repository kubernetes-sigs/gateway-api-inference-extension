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
	"strings"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/common"
	reqenvoy "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/request"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkrq "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
	resputil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/response"
)

const (
	streamingRespPrefix = "data: "
	streamingEndMsg     = "data: [DONE]"
)

// HandleResponseBody always returns the requestContext even in the error case, as the request context is used in error handling.
func (s *StreamingServer) HandleResponseBody(ctx context.Context, reqCtx *RequestContext, responseBytes []byte) (*RequestContext, error) {
	logger := log.FromContext(ctx)

	var usage fwkrq.Usage
	if s.parser != nil {
		parsedResponse, parseErr := s.parser.ParseResponse(responseBytes)
		if parseErr != nil || parsedResponse == nil || parsedResponse.Usage == nil {
			return reqCtx, parseErr
		}
		usage = *parsedResponse.Usage
	} else {
		extractedUsage, err := resputil.ExtractUsage(responseBytes)
		if err != nil || extractedUsage == nil {
			if logger.V(logutil.DEBUG).Enabled() {
				logger.V(logutil.DEBUG).Error(err, "Error unmarshalling response body", "body", string(responseBytes))
			} else {
				logger.V(logutil.DEFAULT).Error(err, "Error unmarshalling response body", "body", string(responseBytes))
			}
			return reqCtx, err
		}
		usage = *extractedUsage
	}
	reqCtx.Usage = usage
	logger.V(logutil.VERBOSE).Info("Response generated", "usage", reqCtx.Usage)
	return s.director.HandleResponseBodyComplete(ctx, reqCtx)
}

// The function is to handle streaming response if the modelServer is streaming.
func (s *StreamingServer) HandleResponseBodyModelStreaming(ctx context.Context, reqCtx *RequestContext, responseBytes []byte) {
	logger := log.FromContext(ctx)
	_, err := s.director.HandleResponseBodyStreaming(ctx, reqCtx)
	if err != nil {
		logger.Error(err, "error in HandleResponseBodyStreaming")
	}
	responseText := string(responseBytes)
	if s.parser != nil {
		parsedResp, err := s.parser.ParseStreamResponse(responseBytes)
		if err != nil || parsedResp.Usage == nil {
			logger.Error(err, "error in HandleResponseBodyStreaming using parser")
		} else {
			reqCtx.Usage = *parsedResp.Usage
		}
	} else {
		// Parse usage on EVERY chunk to catch split streams (where usage and [DONE] are in different chunks).
		if resp := resputil.ExtractUsageStreaming(responseText); resp.Usage != nil && resp.Usage.TotalTokens > 0 {
			reqCtx.Usage = *resp.Usage
		}
	}
	if strings.Contains(responseText, streamingEndMsg) {
		metrics.RecordInputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.Usage.PromptTokens)
		metrics.RecordOutputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.Usage.CompletionTokens)
		cachedToken := 0
		if reqCtx.Usage.PromptTokenDetails != nil {
			cachedToken = reqCtx.Usage.PromptTokenDetails.CachedTokens
		}
		metrics.RecordPromptCachedTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, cachedToken)
	}
}

func (s *StreamingServer) HandleResponseHeaders(ctx context.Context, reqCtx *RequestContext, resp *extProcPb.ProcessingRequest_ResponseHeaders) (*RequestContext, error) {
	for _, header := range resp.ResponseHeaders.Headers.Headers {
		reqCtx.Response.Headers[header.Key] = reqenvoy.GetHeaderValue(header)
	}

	reqCtx, err := s.director.HandleResponseReceived(ctx, reqCtx)

	return reqCtx, err
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

type ResponseBody struct {
	Usage fwkrq.Usage `json:"usage"`
}

type PromptTokenDetails struct {
	CachedTokens int `json:"cached_tokens"`
}
