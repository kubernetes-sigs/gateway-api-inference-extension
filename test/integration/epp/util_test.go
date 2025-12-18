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

package epp

import (
	"encoding/json"
	"fmt"
	"strings"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
	"sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

// --- Domain Request Builders ---

// ReqSubset creates a request sequence with Envoy Endpoint Metadata.
// This simulates the "Subset Load Balancing" flow where EPP picks a specific pod IP.
func ReqSubset(prompt, model, target string, subsets ...string) []*extProcPb.ProcessingRequest {
	// Uses the shared low-level generator which handles the metadata construction
	return integration.GenerateStreamedRequestSet(logger, prompt, model, target, subsets)
}

// ReqResponseOnly creates a sequence simulating only the response phase from Envoy.
// It skips the RequestHeaders phase entirely.
func ReqResponseOnly(
	respHeaders map[string]string,
	bodyChunks ...string,
) []*extProcPb.ProcessingRequest {
	reqs := []*extProcPb.ProcessingRequest{}

	// 1. Response Headers
	hListResp := []*envoyCorev3.HeaderValue{}
	for k, v := range respHeaders {
		hListResp = append(hListResp, &envoyCorev3.HeaderValue{Key: k, RawValue: []byte(v)})
	}
	reqs = append(reqs, &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &extProcPb.HttpHeaders{Headers: &envoyCorev3.HeaderMap{Headers: hListResp}},
		},
	})

	// 2. Response Body Chunks
	for i, chunk := range bodyChunks {
		reqs = append(reqs, &extProcPb.ProcessingRequest{
			Request: &extProcPb.ProcessingRequest_ResponseBody{
				ResponseBody: &extProcPb.HttpBody{
					Body:        []byte(chunk),
					EndOfStream: i == len(bodyChunks)-1,
				},
			},
		})
	}
	return reqs
}

// --- Response Expectations ---

// ExpectRouteTo asserts that the request was successfully routed to the specified endpoint and that the body was
// rewritten to match the target model.
func ExpectRouteTo(endpoint, targetModel, prompt string) []*extProcPb.ProcessingResponse {
	// Reconstruct the expected rewritten body.
	j, _ := json.Marshal(map[string]any{
		"max_tokens": 100, "model": targetModel, "prompt": prompt, "temperature": 0,
	})
	return integration.NewRequestBufferedResponse(
		endpoint, string(j),
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{Key: "hi", RawValue: []byte("mom")}},
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      requtil.RequestIdHeaderKey,
			RawValue: []byte("test-request-id"),
		}},
	)
}

// ExpectReject asserts that the EPP immediately rejected the request with the given code and message.
func ExpectReject(code envoyTypePb.StatusCode, msg string) []*extProcPb.ProcessingResponse {
	return integration.NewImmediateErrorResponse(code, msg)
}

// ExpectBufferResp asserts that the EPP buffers the response and rewrites the body.
// This uses the shared primitive but adds EPP-specific headers we expect.
func ExpectBufferResp(body string, contentType string) []*extProcPb.ProcessingResponse {
	return integration.NewResponseBufferedResponse(
		body,
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      "x-went-into-resp-headers",
			RawValue: []byte("true"),
		}},
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      "content-type",
			RawValue: []byte(contentType),
		}},
	)
}

// ExpectStreamResp asserts that the EPP streams the response chunks (pass-through).
// It constructs a sequence of:
// 1. ResponseHeaders (with "x-went-into-resp-headers" and "text/event-stream")
// 2. ResponseBody chunks (with EndOfStream=true on the final chunk)
func ExpectStreamResp(chunks ...string) []*extProcPb.ProcessingResponse {
	// 1. The Header Response Frame
	res := []*extProcPb.ProcessingResponse{
		integration.NewResponseHeaders(
			&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
				Key:      "x-went-into-resp-headers",
				RawValue: []byte("true"),
			}},
			&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
				Key:      "content-type",
				RawValue: []byte("text/event-stream"),
			}},
			&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{Key: "status", RawValue: []byte("200")}},
		),
	}

	// 2. The Body Chunk Frames
	for i, chunk := range chunks {
		res = append(res, integration.NewResponseStreamChunk(chunk, i == len(chunks)-1))
	}
	return res
}

// --- Data Structures & Metrics Helpers ---

type podState struct {
	index        int
	queueSize    int
	kvCacheUsage float64
	activeModels []string
}

// P constructs a Pod State: Index, Queue, KV%, Models...
// Usage: P(0, 5, 0.2, "model-a")
func P(idx int, q int, kv float64, models ...string) podState {
	return podState{index: idx, queueSize: q, kvCacheUsage: kv, activeModels: models}
}

type label struct{ name, value string }

func labelsToString(labels []label) string {
	var sb strings.Builder
	for i, l := range labels {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(fmt.Sprintf("%s=%q", l.name, l.value))
	}
	return sb.String()
}

func metricReqTotal(model, target string) string {
	return fmt.Sprintf(`
		# HELP inference_objective_request_total [ALPHA] Counter of inference objective requests broken out for each model and target model.
		# TYPE inference_objective_request_total counter
		inference_objective_request_total{%s} 1
		`, labelsToString([]label{{"model_name", model}, {"target_model_name", target}}))
}

func metricReadyPods(count int) string {
	return fmt.Sprintf(`
		# HELP inference_pool_ready_pods [ALPHA] The number of ready pods in the inference server pool.
		# TYPE inference_pool_ready_pods gauge
		inference_pool_ready_pods{%s} %d
		`, labelsToString([]label{{"name", testPoolName}}), count)
}
