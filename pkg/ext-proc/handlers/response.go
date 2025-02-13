package handlers

import (
	"encoding/json"
	"fmt"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	klog "k8s.io/klog/v2"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

// HandleResponseHeaders processes response headers from the backend model server.
func (s *Server) HandleResponseHeaders(reqCtx *RequestContext, req *extProcPb.ProcessingRequest) (*extProcPb.ProcessingResponse, error) {
	klog.V(logutil.VERBOSE).InfoS("Processing ResponseHeaders")
	h := req.Request.(*extProcPb.ProcessingRequest_ResponseHeaders)
	klog.V(logutil.VERBOSE).InfoS("Headers before", "headers", h)

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
func (s *Server) HandleResponseBody(reqCtx *RequestContext, req *extProcPb.ProcessingRequest) (*extProcPb.ProcessingResponse, error) {
	klog.V(logutil.VERBOSE).InfoS("Processing HandleResponseBody")
	body := req.Request.(*extProcPb.ProcessingRequest_ResponseBody)

	res := Response{}
	if err := json.Unmarshal(body.ResponseBody.Body, &res); err != nil {
		return nil, fmt.Errorf("unmarshaling response body: %v", err)
	}
	reqCtx.Response = res
	reqCtx.ResponseSize = len(body.ResponseBody.Body)
	// ResponseComplete is to indicate the response is complete. In non-streaming
	// case, it will be set to be true once the response is processed; in
	// streaming case, it will be set to be true once the last chunk is processed.
	// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/178)
	// will add the processing for streaming case.
	reqCtx.ResponseComplete = true
	klog.V(logutil.VERBOSE).InfoS("Response generated", "response", res)

	resp := &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseBody{
			ResponseBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{},
			},
		},
	}
	return resp, nil
}

type Response struct {
	Usage Usage `json:"usage"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
