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

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	filterPb "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	streamingRespPrefix = "data: "
	streamingEndMsg     = "data: [DONE]"
)

// HandleResponseBody always returns the requestContext even in the error case, as the request context is used in error handling.
func (s *StreamingServer) HandleResponseBody(ctx context.Context, reqCtx *RequestContext, response map[string]any) (*RequestContext, error) {
	logger := log.FromContext(ctx)
	responseBytes, err := json.Marshal(response)
	if err != nil {
		return reqCtx, fmt.Errorf("error marshalling responseBody - %w", err)
	}
	if response["usage"] != nil {
		usg := response["usage"].(map[string]any)
		usage := Usage{
			PromptTokens:     int(usg["prompt_tokens"].(float64)),
			CompletionTokens: int(usg["completion_tokens"].(float64)),
			TotalTokens:      int(usg["total_tokens"].(float64)),
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

	reqCtx.respBodyResp = generateResponseBodyResponses(responseBytes, true, reqCtx, logger)

	return s.director.HandleResponseBodyComplete(ctx, reqCtx)
}

// The function is to handle streaming response if the modelServer is streaming.
func (s *StreamingServer) HandleResponseBodyModelStreaming(ctx context.Context, reqCtx *RequestContext, responseText string) {
	logger := log.FromContext(ctx)
	_, err := s.director.HandleResponseBodyStreaming(ctx, reqCtx)
	if err != nil {
		logger.Error(err, "error in HandleResponseBodyStreaming")
	}
	if strings.Contains(responseText, streamingEndMsg) {
		reqCtx.ResponseComplete = true
		resp := parseRespForUsage(ctx, responseText)
		reqCtx.Usage = resp.Usage
		metrics.RecordInputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, resp.Usage.PromptTokens)
		metrics.RecordOutputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, resp.Usage.CompletionTokens)
		if s.director != nil {
			s.director.HandleResponseBodyComplete(ctx, reqCtx)
		}
	}
	if s.director != nil {
		s.director.HandleResponseBodyStreaming(ctx, reqCtx)
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
		ModeOverride: &filterPb.ProcessingMode{
			ResponseTrailerMode: filterPb.ProcessingMode_SEND,
		},
	}
}

func generateResponseBodyResponses(
	responseBodyBytes []byte,
	setEoS bool,
	reqCtx *RequestContext,
	logger logr.Logger,
) []*extProcPb.ProcessingResponse {
	if reqCtx != nil && reqCtx.modelServerStreaming {
		// For streaming responses, process SSE format
		raw := string(responseBodyBytes)

		// Handle the case where we receive partial SSE data
		if !strings.HasSuffix(raw, "\n\n") && !setEoS {
			// This is a partial chunk, pass it through as-is
			commonResponses := buildCommonResponses(responseBodyBytes, bodyByteLimit, setEoS)
			out := make([]*extProcPb.ProcessingResponse, 0, len(commonResponses))
			for _, cr := range commonResponses {
				out = append(out, &extProcPb.ProcessingResponse{
					Response: &extProcPb.ProcessingResponse_ResponseBody{
						ResponseBody: &extProcPb.BodyResponse{
							Response: cr,
						},
					},
				})
			}
			return out
		}

		// Process complete SSE events
		events := strings.Split(raw, "\n\n")
		var rebuilt strings.Builder

		for _, ev := range events {
			if ev == "" {
				continue
			}

			if !strings.HasPrefix(ev, "data: ") {
				// Pass through non-data events as-is
				rebuilt.WriteString(ev)
				rebuilt.WriteString("\n\n")
				continue
			}

			payload := strings.TrimPrefix(ev, "data: ")
			if payload == "[DONE]" {
				rebuilt.WriteString("data: [DONE]\n\n")
				continue
			}

			// Try to parse and modify JSON payload
			var obj map[string]interface{}
			if err := json.Unmarshal([]byte(payload), &obj); err != nil {
				logger.V(logutil.DEBUG).Info("SSE payload is not JSON, passing through", "payload", payload)
				rebuilt.WriteString("data: ")
				rebuilt.WriteString(payload)
				rebuilt.WriteString("\n\n")
				continue
			}

			// Add metrics to usage if present
			if usage, ok := obj["usage"].(map[string]interface{}); ok && usage != nil {
				usage["ttft_ms"] = reqCtx.TTFT
				usage["predicted_ttft_ms"] = reqCtx.PredictedTTFT
				usage["tpot_observations_ms"] = reqCtx.TPOTObservations
				usage["predicted_tpot_observations_ms"] = reqCtx.PredictedTPOTObservations
				usage["avg_tpot_ms"] = reqCtx.AvgTPOT
				usage["avg_predicted_tpot_ms"] = reqCtx.AvgPredictedTPOT
			}

			// Re-marshal and reconstruct SSE format
			if modifiedBytes, err := json.Marshal(obj); err != nil {
				logger.Error(err, "failed to re-marshal modified JSON", "obj", obj)
				rebuilt.WriteString("data: ")
				rebuilt.WriteString(payload)
				rebuilt.WriteString("\n\n")
			} else {
				rebuilt.WriteString("data: ")
				rebuilt.WriteString(string(modifiedBytes))
				rebuilt.WriteString("\n\n")
			}
		}

		// Convert back to bytes and chunk appropriately
		modifiedBytes := []byte(rebuilt.String())
		commonResponses := buildCommonResponses(modifiedBytes, bodyByteLimit, setEoS)

		// Convert to ProcessingResponses
		out := make([]*extProcPb.ProcessingResponse, 0, len(commonResponses))
		for _, cr := range commonResponses {
			out = append(out, &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_ResponseBody{
					ResponseBody: &extProcPb.BodyResponse{
						Response: cr,
					},
				},
			})
		}
		return out
	} else {
		// Non-streaming response
		commonResponses := buildCommonResponses(responseBodyBytes, bodyByteLimit, setEoS)
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

	lines := strings.Split(responseText, "\n")
	for _, line := range lines {
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
	Usage Usage `json:"usage"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
