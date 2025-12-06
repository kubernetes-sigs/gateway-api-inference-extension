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

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// --- Response Expectations (Streaming) ---

// ExpectBBRHeader asserts that BBR set the specific model header and cleared the route cache.
func ExpectBBRHeader(modelName string) *extProcPb.ProcessingResponse {
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestHeaders{
			RequestHeaders: &extProcPb.HeadersResponse{
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

// --- Response Expectations (Unary) ---

// ExpectBBRUnaryResponse creates expected response for unary tests where the body is mutated directly.
func ExpectBBRUnaryResponse(modelName string) *extProcPb.ProcessingResponse {
	resp := &extProcPb.ProcessingResponse{}

	// If modelName is present, we expect header mutations.
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
						},
					},
				},
			},
		}
	} else {
		// Otherwise, expect a No-Op on the body.
		resp.Response = &extProcPb.ProcessingResponse_RequestBody{
			RequestBody: &extProcPb.BodyResponse{},
		}
	}
	return resp
}
