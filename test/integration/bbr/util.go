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

package bbr

import (
	"encoding/json"
	"strconv"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// --- Response Expectations (Streaming) ---

// ExpectBBRHeader asserts that BBR set the specific model header and cleared the route cache.
// baseModelName is the expected base model name (e.g., "llama" for both "llama" and "sql-lora-sheddable")
func ExpectBBRHeader(modelName, baseModelName string, contentLength string) *extProcPb.ProcessingResponse {
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestHeaders{
			RequestHeaders: &extProcPb.HeadersResponse{
				Response: &extProcPb.CommonResponse{
					ClearRouteCache: true,
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: []*envoyCorev3.HeaderValueOption{
							{
								Header: &envoyCorev3.HeaderValue{
									Key:      "Content-Length",
									RawValue: []byte(contentLength),
								},
							},
							{
								Header: &envoyCorev3.HeaderValue{
									Key:      "X-Gateway-Base-Model-Name",
									RawValue: []byte(baseModelName),
								},
							},
							{
								Header: &envoyCorev3.HeaderValue{
									Key:      "X-Gateway-Model-Name",
									RawValue: []byte(modelName),
								},
							},
						},
					},
				},
			},
		},
	}
}

// ExpectBBRBodyPassThrough asserts that BBR reconstructs and passes the body through.
// BBR buffers the body to inspect it, then sends it downstream as a single chunk (usually).
func ExpectBBRBodyPassThrough(prompt, model string) *extProcPb.ProcessingResponse {
	j := map[string]any{
		"max_tokens": 100, "prompt": prompt, "temperature": 0,
	}
	if model != "" {
		j["model"] = model
	}
	b, _ := json.Marshal(j)

	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestBody{
			RequestBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					BodyMutation: &extProcPb.BodyMutation{
						Mutation: &extProcPb.BodyMutation_StreamedResponse{
							StreamedResponse: &extProcPb.StreamedBodyResponse{
								Body:        b,
								EndOfStream: true,
							},
						},
					},
				},
			},
		},
	}
}

// ExpectBBRNoOpHeader asserts that BBR did nothing to the headers (e.g., when no model is found).
func ExpectBBRNoOpHeader() *extProcPb.ProcessingResponse {
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestHeaders{
			RequestHeaders: &extProcPb.HeadersResponse{},
		},
	}
}

// --- Response Phase Expectations ---

// ExpectResponseHeadersPassThrough asserts that BBR passed response headers through with no mutations.
func ExpectResponseHeadersPassThrough() *extProcPb.ProcessingResponse {
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseHeaders{
			ResponseHeaders: &extProcPb.HeadersResponse{},
		},
	}
}

// ExpectResponseBodyPassThrough asserts that BBR passed the response body through with no mutations
// (i.e., no response plugins configured).
func ExpectResponseBodyPassThrough() *extProcPb.ProcessingResponse {
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseBody{
			ResponseBody: &extProcPb.BodyResponse{},
		},
	}
}

// ExpectResponseBodyMutation asserts that a response plugin mutated the response body (unary mode).
// Includes the Content-Length header mutation.
func ExpectResponseBodyMutation(body map[string]any) *extProcPb.ProcessingResponse {
	b, _ := json.Marshal(body)
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseBody{
			ResponseBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					ClearRouteCache: true,
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: []*envoyCorev3.HeaderValueOption{
							{
								Header: &envoyCorev3.HeaderValue{
									Key:      "Content-Length",
									RawValue: []byte(strconv.Itoa(len(b))),
								},
							},
						},
					},
					BodyMutation: &extProcPb.BodyMutation{
						Mutation: &extProcPb.BodyMutation_Body{
							Body: b,
						},
					},
				},
			},
		},
	}
}

// --- Request Phase Expectations (Unary) ---

// ExpectBBRUnaryResponse creates expected response for unary tests where the body is mutated directly.
// baseModelName is the expected base model name (e.g., "llama" for both "llama" and "sql-lora-sheddable")
func ExpectBBRUnaryResponse(modelName, baseModelName string, prompt string) *extProcPb.ProcessingResponse {
	resp := &extProcPb.ProcessingResponse{}

	if modelName != "" {
		resp.Response = &extProcPb.ProcessingResponse_RequestBody{
			RequestBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					ClearRouteCache: true,
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: []*envoyCorev3.HeaderValueOption{
							{
								Header: &envoyCorev3.HeaderValue{
									Key:      "X-Gateway-Model-Name",
									RawValue: []byte(modelName),
								},
							},
							{
								Header: &envoyCorev3.HeaderValue{
									Key:      "X-Gateway-Base-Model-Name",
									RawValue: []byte(baseModelName),
								},
							},
						},
					},
				},
			},
		}
	} else {
		resp.Response = &extProcPb.ProcessingResponse_RequestBody{
			RequestBody: &extProcPb.BodyResponse{},
		}
	}
	return resp
}
