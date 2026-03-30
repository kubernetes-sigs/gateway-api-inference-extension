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
	"strconv"
	"strings"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	reqcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/request"
	pb "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/gen"
	"sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

// buildEnvoyHeaders is a helper to convert a map of strings into Envoy HeaderValues.
func buildEnvoyHeaders(headers map[string]string) []*envoyCorev3.HeaderValue {
	hList := make([]*envoyCorev3.HeaderValue, 0, len(headers))
	for k, v := range headers {
		hList = append(hList, &envoyCorev3.HeaderValue{Key: k, RawValue: []byte(v)})
	}
	return hList
}

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
	reqs := make([]*extProcPb.ProcessingRequest, 0, 1+len(bodyChunks))

	// 1. Response Headers
	reqs = append(reqs, &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{Headers: buildEnvoyHeaders(respHeaders)},
			},
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

// ReqResponseGRPCWithTailer creates a sequence simulating only the response phase from Envoy while passing gRPC payload.
// It also generate a response trailer.
// It skips the RequestHeaders phase entirely.
func ReqResponseGRPCWithTailer(
	respHeaders map[string]string,
	bodyChunks ...[]byte,
) []*extProcPb.ProcessingRequest {
	reqs := make([]*extProcPb.ProcessingRequest, 0, 1+len(bodyChunks))

	// 1. Response Headers
	reqs = append(reqs, &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{Headers: buildEnvoyHeaders(respHeaders)},
			},
		},
	})

	// 2. Response Body Chunks
	for _, chunk := range bodyChunks {
		reqs = append(reqs, &extProcPb.ProcessingRequest{
			Request: &extProcPb.ProcessingRequest_ResponseBody{
				ResponseBody: &extProcPb.HttpBody{
					Body:        chunk,
					EndOfStream: false,
				},
			},
		})
	}

	// 3. Response Trailer
	reqs = append(reqs, &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_ResponseTrailers{
			ResponseTrailers: &extProcPb.HttpTrailers{},
		},
	})
	return reqs
}

// --- Response Expectations ---

func buildRouteResponse(endpoint, targetModel, prompt string, stream bool) []*extProcPb.ProcessingResponse {
	bodyMap := map[string]any{
		"max_tokens": 100, "model": targetModel, "prompt": prompt, "temperature": 0,
	}
	if stream {
		bodyMap["stream"] = true
	}
	j, _ := json.Marshal(bodyMap)

	return integration.NewRequestBufferedResponse(
		endpoint, j,
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{Key: "hi", RawValue: []byte("mom")}},
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      reqcommon.RequestIdHeaderKey,
			RawValue: []byte("test-request-id"),
		}},
	)
}

func buildGRPCRouteResponse(endpoint, prompt, method string, stream bool) []*extProcPb.ProcessingResponse {
	req := &pb.GenerateRequest{
		Input: &pb.GenerateRequest_Text{
			Text: prompt,
		},
		Stream: stream,
	}
	j, _ := integration.CreateGrpcPayload(req)

	return integration.NewRequestBufferedResponse(
		endpoint, j,
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{Key: "hi", RawValue: []byte("mom")}},
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      reqcommon.RequestIdHeaderKey,
			RawValue: []byte("test-request-id"),
		}},
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{Key: ":path", RawValue: []byte(method)}},
	)
}

// ExpectRouteTo asserts that the request was successfully routed to the specified endpoint and that the body was
// rewritten to match the target model.
func ExpectRouteTo(endpoint, targetModel, prompt string) []*extProcPb.ProcessingResponse {
	return buildRouteResponse(endpoint, targetModel, prompt, false)
}

// ExpectRouteToWithStream asserts that the request was successfully routed with streaming enabled.
func ExpectRouteToWithStream(endpoint, targetModel, prompt string) []*extProcPb.ProcessingResponse {
	return buildRouteResponse(endpoint, targetModel, prompt, true)
}

// ExpectGRPCRouteTo asserts that the request was successfully routed to the specified endpoint.
func ExpectGRPCRouteTo(endpoint, prompt, method string) []*extProcPb.ProcessingResponse {
	return buildGRPCRouteResponse(endpoint, prompt, method, false)
}

// ExpectGRPCRouteToWithStream asserts that the request was successfully routed with streaming enabled.
func ExpectGRPCRouteToWithStream(endpoint, prompt, method string) []*extProcPb.ProcessingResponse {
	return buildGRPCRouteResponse(endpoint, prompt, method, true)
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
		contentType != "application/grpc", // For gRPC, the EoS will not be set to true but it will return a trailer.
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
	res := make([]*extProcPb.ProcessingResponse, 0, 1+len(chunks))
	res = append(res, integration.NewResponseHeaders(
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      "x-went-into-resp-headers",
			RawValue: []byte("true"),
		}},
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      "content-type",
			RawValue: []byte("text/event-stream"),
		}},
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{Key: "status", RawValue: []byte("200")}},
	))

	// 2. The Body Chunk Frames
	for i, chunk := range chunks {
		res = append(res, integration.NewResponseStreamChunk(chunk, i == len(chunks)-1))
	}
	return res
}

// ExpectGRPCStreamResp asserts that the EPP streams the gRPC response chunks.
func ExpectGRPCStreamResp(chunks ...string) []*extProcPb.ProcessingResponse {
	res := make([]*extProcPb.ProcessingResponse, 0, 1+len(chunks))
	res = append(res, integration.NewResponseHeaders(
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      "x-went-into-resp-headers",
			RawValue: []byte("true"),
		}},
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      "content-type",
			RawValue: []byte("application/grpc"),
		}},
	))

	for _, chunk := range chunks {
		res = append(res, integration.NewResponseStreamChunk(chunk, false))
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
	parts := make([]string, len(labels))
	for i, l := range labels {
		parts[i] = fmt.Sprintf("%s=%q", l.name, l.value)
	}
	return strings.Join(parts, ",")
}

func metricReqTotal(model, target string, priority int) string {
	return fmt.Sprintf(`
    # HELP inference_objective_request_total [ALPHA] Counter of inference objective requests broken out for each model and target model.
    # TYPE inference_objective_request_total counter
    inference_objective_request_total{%s} 1
    `, labelsToString([]label{{"model_name", model}, {"priority", strconv.Itoa(priority)}, {"target_model_name", target}}))
}

func metricReadyPods(count int) string {
	return fmt.Sprintf(`
    # HELP inference_pool_ready_pods [ALPHA] The number of ready pods in the inference server pool.
    # TYPE inference_pool_ready_pods gauge
    inference_pool_ready_pods{%s} %d
    `, labelsToString([]label{{"name", testPoolName}}), count)
}

// cleanMetric removes indentation from multiline metric strings and ensures a trailing newline exists, which is
// required by the Prometheus text parser.
func cleanMetric(s string) string {
	lines := strings.Split(s, "\n")
	var cleaned []string
	for _, l := range lines {
		trimmed := strings.TrimSpace(l)
		if trimmed != "" {
			cleaned = append(cleaned, trimmed)
		}
	}
	return strings.Join(cleaned, "\n") + "\n"
}
