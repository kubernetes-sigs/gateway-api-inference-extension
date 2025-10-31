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
	"bytes"
	"context"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	streamingRespPrefix = "data: "
	streamingEndMsg     = "data: [DONE]"
)

// HandleResponseBody always returns the requestContext even in the error case, as the request context is used in error handling.
func (s *StreamingServer) HandleResponseBody(ctx context.Context, reqCtx *RequestContext, body []byte) (*RequestContext, error) {
	logger := log.FromContext(ctx)
	llmResponse, err := types.NewLLMResponseFromBytes(body)
	if err != nil {
		logger.Error(err, "failed to create LLMResponse from bytes")
	} else {
		reqCtx.SchedulingResponse = llmResponse
		if usage := reqCtx.SchedulingResponse.Usage(); usage != nil {
			reqCtx.Usage = usage
			logger.V(logutil.VERBOSE).Info("Response generated", "usage", usage)
		}
	}
	reqCtx.ResponseSize = len(body)
	// ResponseComplete is to indicate the response is complete. In non-streaming
	// case, it will be set to be true once the response is processed; in
	// streaming case, it will be set to be true once the last chunk is processed.
	// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/178)
	// will add the processing for streaming case.
	reqCtx.ResponseComplete = true

	reqCtx.respBodyResp = generateResponseBodyResponses(body, true)

	return s.director.HandleResponseBodyComplete(ctx, reqCtx)
}

// The function is to handle streaming response if the modelServer is streaming.
func (s *StreamingServer) HandleResponseBodyModelStreaming(ctx context.Context, reqCtx *RequestContext, streamBody []byte) {
	logger := log.FromContext(ctx)
	_, err := s.director.HandleResponseBodyStreaming(ctx, reqCtx)
	if err != nil {
		logger.Error(err, "error in HandleResponseBodyStreaming")
	}
}

func (s *StreamingServer) HandleResponseBodyModelStreamingComplete(ctx context.Context, reqCtx *RequestContext, streamBody []byte) {
	logger := log.FromContext(ctx)
	if bytes.Contains(streamBody, []byte(streamingEndMsg)) {
		reqCtx.ResponseComplete = true
		resp, err := types.NewLLMResponseFromStream(streamBody)
		if err != nil {
			logger.Error(err, "error in converting stream response to LLMResponse.")
		} else {
			reqCtx.SchedulingResponse = resp
			if usage := resp.Usage(); usage != nil {
				reqCtx.Usage = usage
				metrics.RecordInputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, usage.PromptTokens)
				metrics.RecordOutputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, usage.CompletionTokens)
			}
		}
		_, err = s.director.HandleResponseBodyComplete(ctx, reqCtx)
		if err != nil {
			logger.Error(err, "error in HandleResponseBodyComplete")
		}
	}
}

func (s *StreamingServer) HandleResponseHeaders(ctx context.Context, reqCtx *RequestContext, resp *extProcPb.ProcessingRequest_ResponseHeaders) (*RequestContext, error) {
	for _, header := range resp.ResponseHeaders.Headers.Headers {
		if header.RawValue != nil {
			reqCtx.Response.Headers[header.Key] = string(header.RawValue)
		} else {
			reqCtx.Response.Headers[header.Key] = header.Value
		}
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
	commonResponses := buildCommonResponses(responseBodyBytes, bodyByteLimit, setEoS)
	responses := []*extProcPb.ProcessingResponse{}
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

	// include all headers
	for key, value := range reqCtx.Response.Headers {
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
	Usage Usage `json:"usage"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
