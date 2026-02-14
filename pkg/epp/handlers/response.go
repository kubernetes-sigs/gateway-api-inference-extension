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
	"strings"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/common"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkrq "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

const (
	streamingRespPrefix = "data: "
	streamingEndMsg     = "data: [DONE]"

	// OpenAI API object types
	objectTypeResponse            = "response"
	objectTypeConversation        = "conversation"
	objectTypeChatCompletion      = "chat.completion"
	objectTypeChatCompletionChunk = "chat.completion.chunk"
	objectTypeTextCompletion      = "text_completion"
)

// extractUsageByAPIType extracts usage statistics using the appropriate field names
// based on the OpenAI API type identified by the "object" field.
func extractUsageByAPIType(usg map[string]any, objectType string) fwkrq.Usage {
	usage := fwkrq.Usage{}

	switch {
	case strings.HasPrefix(objectType, objectTypeResponse) || strings.HasPrefix(objectType, objectTypeConversation):
		// Responses/Conversations APIs use input_tokens/output_tokens
		if usg["input_tokens"] != nil {
			usage.PromptTokens = int(usg["input_tokens"].(float64))
		}
		if usg["output_tokens"] != nil {
			usage.CompletionTokens = int(usg["output_tokens"].(float64))
		}
	case objectType == objectTypeChatCompletion || objectType == objectTypeChatCompletionChunk || objectType == objectTypeTextCompletion:
		// Traditional APIs use prompt_tokens/completion_tokens
		if usg["prompt_tokens"] != nil {
			usage.PromptTokens = int(usg["prompt_tokens"].(float64))
		}
		if usg["completion_tokens"] != nil {
			usage.CompletionTokens = int(usg["completion_tokens"].(float64))
		}
	default:
		// Fallback: try both field naming conventions
		if usg["input_tokens"] != nil {
			usage.PromptTokens = int(usg["input_tokens"].(float64))
		} else if usg["prompt_tokens"] != nil {
			usage.PromptTokens = int(usg["prompt_tokens"].(float64))
		}

		if usg["output_tokens"] != nil {
			usage.CompletionTokens = int(usg["output_tokens"].(float64))
		} else if usg["completion_tokens"] != nil {
			usage.CompletionTokens = int(usg["completion_tokens"].(float64))
		}
	}

	// total_tokens field name is consistent across all API types
	if usg["total_tokens"] != nil {
		usage.TotalTokens = int(usg["total_tokens"].(float64))
	}

	return usage
}

// HandleResponseBody always returns the requestContext even in the error case, as the request context is used in error handling.
func (s *StreamingServer) HandleResponseBody(ctx context.Context, reqCtx *RequestContext, response map[string]any) (*RequestContext, error) {
	logger := log.FromContext(ctx)
	responseBytes, err := json.Marshal(response)
	if err != nil {
		return reqCtx, fmt.Errorf("error marshalling responseBody - %w", err)
	}
	if response["usage"] != nil {
		usg := response["usage"].(map[string]any)
		objectType, _ := response["object"].(string)
		usage := extractUsageByAPIType(usg, objectType)
		if usg["prompt_token_details"] != nil {
			detailsMap := usg["prompt_token_details"].(map[string]any)
			if cachedTokens, ok := detailsMap["cached_tokens"]; ok {
				usage.PromptTokenDetails = &fwkrq.PromptTokenDetails{
					CachedTokens: int(cachedTokens.(float64)),
				}
			}
		}
		reqCtx.Usage = usage
		logger.V(logutil.VERBOSE).Info("Response generated", "usage", reqCtx.Usage)
	}
	reqCtx.ResponseSize = len(responseBytes)
	// ResponseComplete is to indicate the response is complete. In non-streaming
	// case, it will be set to be true once the response is processed; in
	// streaming case, it will be set to be true once the last chunk is processed.
	// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/178)
	// will add the processing for streaming case.
	reqCtx.ResponseComplete = true

	reqCtx.respBodyResp = generateResponseBodyResponses(responseBytes, true)

	return s.director.HandleResponseBodyComplete(ctx, reqCtx)
}

// The function is to handle streaming response if the modelServer is streaming.
func (s *StreamingServer) HandleResponseBodyModelStreaming(ctx context.Context, reqCtx *RequestContext, responseText string) {
	logger := log.FromContext(ctx)
	_, err := s.director.HandleResponseBodyStreaming(ctx, reqCtx)
	if err != nil {
		logger.Error(err, "error in HandleResponseBodyStreaming")
	}

	// Record TTFT on the first token chunk.
	// We check for "data: " prefix to ensure it's a data chunk, and exclude "[DONE]" message.
	if reqCtx.GeneratedTokenCount == 0 && strings.Contains(responseText, streamingRespPrefix) && !strings.Contains(responseText, streamingEndMsg) {
		ttft := time.Since(reqCtx.RequestReceivedTimestamp).Seconds()
		reqCtx.TTFT = ttft
		metrics.RecordRequestTTFT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, ttft)
		reqCtx.GeneratedTokenCount = 1
		reqCtx.LastTokenTimestamp = time.Now()
	} else if reqCtx.GeneratedTokenCount > 0 && strings.Contains(responseText, streamingRespPrefix) && !strings.Contains(responseText, streamingEndMsg) {
		// Record ITL for subsequent tokens
		itl := time.Since(reqCtx.LastTokenTimestamp).Seconds()
		metrics.RecordRequestITL(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, itl)
		reqCtx.LastTokenTimestamp = time.Now()
		reqCtx.GeneratedTokenCount++
	}

	// Parse usage on EVERY chunk to catch split streams (where usage and [DONE] are in different chunks).
	if resp := parseRespForUsage(ctx, responseText); resp.Usage.TotalTokens > 0 {
		reqCtx.Usage = resp.Usage
	}

	if strings.Contains(responseText, streamingEndMsg) {
		reqCtx.ResponseComplete = true
		metrics.RecordInputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.Usage.PromptTokens)
		metrics.RecordOutputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.Usage.CompletionTokens)
		cachedToken := 0
		if reqCtx.Usage.PromptTokenDetails != nil {
			cachedToken = reqCtx.Usage.PromptTokenDetails.CachedTokens
		}
		metrics.RecordPromptCachedTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, cachedToken)

		// Record Time Per Output Token
		// TPOT = (Total Duration - TTFT) / (OutputTokens - 1)
		if reqCtx.Usage.CompletionTokens > 1 && reqCtx.TTFT > 0 {
			totalDuration := time.Since(reqCtx.RequestReceivedTimestamp).Seconds()
			generationDuration := totalDuration - reqCtx.TTFT
			metrics.RecordRequestDecodeDuration(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, generationDuration)
			metrics.RecordRequestE2ELatency(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, totalDuration)

			// Avoid division by zero just in case
			if count := float64(reqCtx.Usage.CompletionTokens - 1); count > 0 {
				avgTPOT := generationDuration / count
				reqCtx.TPOT = avgTPOT
				metrics.RecordRequestTPOT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, avgTPOT)
			}
		}
	}
}

func (s *StreamingServer) HandleResponseHeaders(ctx context.Context, reqCtx *RequestContext, resp *extProcPb.ProcessingRequest_ResponseHeaders) (*RequestContext, error) {
	for _, header := range resp.ResponseHeaders.Headers.Headers {
		reqCtx.Response.Headers[header.Key] = request.GetHeaderValue(header)
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

// Example message if "stream_options": {"include_usage": "true"} is included in the request:
// data: {"id":"...","object":"text_completion","created":1739400043,"model":"food-review-0","choices":[],
// "usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}
//
// data: [DONE]
//
// Noticed that vLLM returns two entries in one response.
// We need to strip the `data:` prefix and next Data: [DONE] from the message to fetch response data.
//
// If include_usage is not included in the request, `data: [DONE]` is returned separately, which
// indicates end of streaming.
func parseRespForUsage(ctx context.Context, responseText string) ResponseBody {
	response := ResponseBody{}
	logger := log.FromContext(ctx)

	lines := strings.SplitSeq(responseText, "\n")
	for line := range lines {
		if !strings.HasPrefix(line, streamingRespPrefix) {
			continue
		}
		content := strings.TrimPrefix(line, streamingRespPrefix)
		if content == "[DONE]" {
			continue
		}

		byteSlice := []byte(content)
		if err := json.Unmarshal(byteSlice, &response); err != nil {
			logger.Error(err, "unmarshaling response body")
			continue
		}
	}

	return response
}

type ResponseBody struct {
	Usage fwkrq.Usage `json:"usage"`
}

type PromptTokenDetails struct {
	CachedTokens int `json:"cached_tokens"`
}
