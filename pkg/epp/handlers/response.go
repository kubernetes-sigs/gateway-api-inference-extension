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
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	streamingRespPrefix = "data: "
	streamingEndMsg     = "data: [DONE]"
)

// HandleResponseHeaders processes response headers from the backend model server.
func (s *Server) HandleResponseHeaders(
	ctx context.Context,
	reqCtx *RequestContext,
	req *extProcPb.ProcessingRequest,
) (*extProcPb.ProcessingResponse, error) {
	loggerVerbose := log.FromContext(ctx).V(logutil.VERBOSE)
	loggerVerbose.Info("Processing ResponseHeaders")
	h := req.Request.(*extProcPb.ProcessingRequest_ResponseHeaders)
	loggerVerbose.Info("Headers before", "headers", h)

	// Example header
	// {
	// 	"ResponseHeaders": {
	// 	  "headers": [
	// 		{
	// 		  "key": ":status",
	// 		  "raw_value": "200"
	// 		},
	// 		{
	// 		  "key": "date",
	// 		  "raw_value": "Thu, 30 Jan 2025 18:50:48 GMT"
	// 		},
	// 		{
	// 		  "key": "server",
	// 		  "raw_value": "uvicorn"
	// 		},
	// 		{
	// 		  "key": "content-type",
	// 		  "raw_value": "text/event-stream; charset=utf-8"
	// 		},
	// 		{
	// 		  "key": "transfer-encoding",
	// 		  "raw_value": "chunked"
	// 		}
	// 	  ]
	// 	}
	// }
	for _, header := range h.ResponseHeaders.Headers.GetHeaders() {
		var statusFound, typeFound bool
		if header.Key == "status" {
			code := header.RawValue[0]
			if string(code) != "200" {
				reqCtx.ResponseStatusCode = errutil.ModelServerError
				statusFound = true
			}
		}
		if header.Key == "content-type" {
			contentType := header.RawValue
			if strings.Contains(string(contentType), "text/event-stream") {
				reqCtx.Streaming = true
			} else {
				reqCtx.Streaming = false
			}
			typeFound = true
		}

		if statusFound && typeFound {
			break
		}
	}

	resp := &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseHeaders{
			ResponseHeaders: &extProcPb.HeadersResponse{
				Response: &extProcPb.CommonResponse{
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: []*configPb.HeaderValueOption{
							{
								Header: &configPb.HeaderValue{
									// This is for debugging purpose only.
									Key:      "x-went-into-resp-headers",
									RawValue: []byte("true"),
								},
							},
						},
					},
				},
			},
		},
	}
	return resp, nil
}

// HandleResponseBody parses response body to update information such as number of completion tokens.
// NOTE: The current implementation only supports Buffered mode, which is not enabled by default. To
// use it, you need to configure EnvoyExtensionPolicy to have response body in Buffered mode.
// https://www.envoyproxy.io/docs/envoy/latest/api-v3/extensions/filters/http/ext_proc/v3/processing_mode.proto#envoy-v3-api-msg-extensions-filters-http-ext-proc-v3-processingmode
// Example response
/*
{
    "id": "cmpl-573498d260f2423f9e42817bbba3743a",
    "object": "text_completion",
    "created": 1732563765,
    "model": "meta-llama/Llama-2-7b-hf",
    "choices": [
        {
            "index": 0,
            "text": " Chronicle\nThe San Francisco Chronicle has a new book review section, and it's a good one. The reviews are short, but they're well-written and well-informed. The Chronicle's book review section is a good place to start if you're looking for a good book review.\nThe Chronicle's book review section is a good place to start if you're looking for a good book review. The Chronicle's book review section",
            "logprobs": null,
            "finish_reason": "length",
            "stop_reason": null,
            "prompt_logprobs": null
        }
    ],
    "usage": {
        "prompt_tokens": 11,
        "total_tokens": 111,
        "completion_tokens": 100
    }
}*/
func (s *Server) HandleResponseBody(
	ctx context.Context,
	reqCtx *RequestContext,
	req *extProcPb.ProcessingRequest,
) (*extProcPb.ProcessingResponse, error) {
	logger := log.FromContext(ctx)
	loggerVerbose := logger.V(logutil.VERBOSE)
	body := req.Request.(*extProcPb.ProcessingRequest_ResponseBody)

	if reqCtx.Streaming {
		logger.V(logutil.DEBUG).Info("Processing HandleResponseBody")
		if err := s.HandleStreaming(ctx, reqCtx, body, loggerVerbose); err != nil {
			return nil, err
		}
	} else {
		loggerVerbose.Info("Processing HandleResponseBody")
		if err := s.HandleNonStreaming(ctx, reqCtx, body, loggerVerbose); err != nil {
			return nil, err
		}
	}

	resp := &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseBody{
			ResponseBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{},
			},
		},
	}
	return resp, nil
}

func (s *Server) HandleNonStreaming(
	ctx context.Context,
	reqCtx *RequestContext,
	body *extProcPb.ProcessingRequest_ResponseBody,
	loggerVerbose logr.Logger,
) error {
	loggerVerbose.Info("Processing HandleResponseBody")

	res := Response{}
	if err := json.Unmarshal(body.ResponseBody.Body, &res); err != nil {
		return errutil.Error{Code: errutil.Internal, Msg: fmt.Sprintf("unmarshaling response body: %v", err)}
	}
	reqCtx.Response = res
	reqCtx.ResponseSize = len(body.ResponseBody.Body)
	reqCtx.ResponseComplete = true
	loggerVerbose.Info("Response generated", "response", res)
	return nil
}

func (s *Server) HandleStreaming(
	ctx context.Context,
	reqCtx *RequestContext,
	body *extProcPb.ProcessingRequest_ResponseBody,
	loggerVerbose logr.Logger,
) error {
	responseText := string(body.ResponseBody.Body)
	if strings.Contains(responseText, streamingEndMsg) {
		parsedResp := ParseRespForUsage(ctx, responseText, loggerVerbose)
		reqCtx.Response = parsedResp
	}

	if body.ResponseBody.EndOfStream {
		loggerVerbose.Info("Streaming is completed")
		reqCtx.ResponseComplete = true
	} else {
		reqCtx.ResponseSize += len(body.ResponseBody.Body)
	}

	return nil
}

// Example message if "stream_options": {"include_usage": "true"} is included in the request:
// data: {"id":"...","object":"text_completion","created":1739400043,"model":"tweet-summary-0","choices":[],
// "usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}
//
// data: [DONE]
//
// Noticed that vLLM returns two entries in one response.
// We need to strip the `data:` prefix and next Data: [DONE] from the message to fetch response data.
//
// If include_usage is not included in the request, `data: [DONE]` is returned separately, which
// indicates end of streaming.
func ParseRespForUsage(
	ctx context.Context,
	responseText string,
	loggerVerbose logr.Logger,
) Response {
	response := Response{}

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
			loggerVerbose.Error(err, "unmarshaling response body")
			continue
		}
	}

	return response
}

type Response struct {
	Usage Usage `json:"usage"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
