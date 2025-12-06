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

// Package integration provides shared utilities, request builders, and assertions for the hermetic integration test
// suites of the Gateway API Inference Extension.
//
// It encapsulates the complexity of constructing Envoy ext_proc Protobuf messages and managing gRPC streams, allowing
// individual test suites (e.g., test/integration/epp, test/integration/bbr) to focus on behavioral assertions rather
// than protocol boilerplate.
package integration

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"strconv"
	"testing"
	"time"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"google.golang.org/protobuf/types/known/structpb"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

// --- Request Builders (High-Level DSL) ---

// ReqLLM creates a sequence of gRPC messages representing a standard, streamed LLM inference request.
// It generates:
//  1. A RequestHeaders message containing standard inference headers (Objective, Model Rewrite, Request ID).
//  2. A RequestBody message containing the JSON payload with EndOfStream=true.
//
// Use this for the majority of "Happy Path" EPP and BBR streaming tests.
func ReqLLM(logger logr.Logger, prompt, model, targetModel string) []*extProcPb.ProcessingRequest {
	return GenerateStreamedRequestSet(logger, prompt, model, targetModel, nil)
}

// ReqLLMUnary creates a single `ProcessingRequest` containing a complete JSON body.
// This simulates a scenario where Envoy has buffered the request body before sending it to the external processor
// (unary mode).
//
// Use this for tests where `streaming: false` or when testing legacy buffered behavior.
func ReqLLMUnary(logger logr.Logger, prompt, model string) *extProcPb.ProcessingRequest {
	return GenerateRequest(logger, prompt, model, nil)
}

// ReqRaw creates a custom sequence of gRPC messages with specific headers and arbitrary body chunks.
// This is a lower-level helper useful for testing edge cases, such as:
//   - Invalid JSON bodies (to test error handling).
//   - Fragmentation (split bodies) to ensure the processor handles accumulation correctly.
//   - Protocol attacks (e.g., missing headers).
func ReqRaw(headers map[string]string, bodyChunks ...string) []*extProcPb.ProcessingRequest {
	reqs := []*extProcPb.ProcessingRequest{}

	// 1. Headers Phase
	hList := []*envoyCorev3.HeaderValue{}
	for k, v := range headers {
		hList = append(hList, &envoyCorev3.HeaderValue{Key: k, Value: v})
	}
	reqs = append(reqs, &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestHeaders{
			RequestHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{Headers: hList},
			},
		},
	})

	// 2. Body Phase (Chunks)
	for i, chunk := range bodyChunks {
		reqs = append(reqs, &extProcPb.ProcessingRequest{
			Request: &extProcPb.ProcessingRequest_RequestBody{
				RequestBody: &extProcPb.HttpBody{
					Body:        []byte(chunk),
					EndOfStream: i == len(bodyChunks)-1,
				},
			},
		})
	}
	return reqs
}

// ReqHeaderOnly creates a request sequence consisting solely of headers, with no body.
// It sets `EndOfStream: true` on the headers frame.
//
// Use this for testing non-inference traffic, such as GET requests, health checks, or requests that should bypass the
// inference processor logic.
func ReqHeaderOnly(headers map[string]string) []*extProcPb.ProcessingRequest {
	hList := []*envoyCorev3.HeaderValue{}
	for k, v := range headers {
		hList = append(hList, &envoyCorev3.HeaderValue{Key: k, Value: v})
	}
	return []*extProcPb.ProcessingRequest{{
		Request: &extProcPb.ProcessingRequest_RequestHeaders{
			RequestHeaders: &extProcPb.HttpHeaders{
				Headers:     &envoyCorev3.HeaderMap{Headers: hList},
				EndOfStream: true,
			},
		},
	}}
}

// --- Request Builders (Low-Level Generators) ---

// GenerateRequest constructs a `ProcessingRequest` containing a JSON-formatted LLM payload.
// It accepts a filterMetadata slice to inject Envoy Dynamic Metadata (used for subset load balancing).
func GenerateRequest(logger logr.Logger, prompt, model string, filterMetadata []string) *extProcPb.ProcessingRequest {
	j := map[string]any{
		"prompt":      prompt,
		"max_tokens":  100,
		"temperature": 0,
	}
	if model != "" {
		j["model"] = model
	}

	// Panic on marshal failure is acceptable in test helpers as it implies a bug in the test code itself.
	llmReq, err := json.Marshal(j)
	if err != nil {
		panic(fmt.Errorf("failed to marshal LLM request: %w", err))
	}

	return &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{Body: llmReq, EndOfStream: true},
		},
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: GenerateRequestMetadata(filterMetadata),
		},
	}
}

// GenerateStreamedRequestSet creates a slice of requests simulating an Envoy stream:
// 1. A Headers frame with standard Inference Extension headers.
// 2. A Body frame with the JSON payload.
func GenerateStreamedRequestSet(
	logger logr.Logger,
	prompt, model, targetModel string,
	filterMetadata []string,
) []*extProcPb.ProcessingRequest {
	requests := []*extProcPb.ProcessingRequest{}

	// Headers
	headers := []*envoyCorev3.HeaderValue{
		{Key: "hi", Value: "mom"},
		{Key: requtil.RequestIdHeaderKey, Value: "test-request-id"},
	}
	if model != "" {
		headers = append(headers, &envoyCorev3.HeaderValue{Key: metadata.ObjectiveKey, Value: model})
	}
	if targetModel != "" {
		headers = append(headers, &envoyCorev3.HeaderValue{Key: metadata.ModelNameRewriteKey, Value: targetModel})
	}

	headerReq := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestHeaders{
			RequestHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{Headers: headers},
			},
		},
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: GenerateRequestMetadata(filterMetadata),
		},
	}
	requests = append(requests, headerReq)

	// Body
	requests = append(requests, GenerateRequest(logger, prompt, model, filterMetadata))
	return requests
}

// GenerateRequestMetadata constructs the Envoy Dynamic Metadata structure.
// This is primarily used to inject "envoy.lb" subset keys for testing logic that depends on specific backend subsets.
func GenerateRequestMetadata(filterMetadata []string) map[string]*structpb.Struct {
	requestMetadata := make(map[string]*structpb.Struct)
	interfaceList := make([]any, len(filterMetadata))
	for i, val := range filterMetadata {
		interfaceList[i] = val
	}
	if filterMetadata != nil {
		structVal, _ := structpb.NewStruct(map[string]any{
			metadata.SubsetFilterKey: interfaceList,
		})
		requestMetadata[metadata.SubsetFilterNamespace] = structVal
	}
	return requestMetadata
}

// --- Response Builders ---

// NewRequestBufferedResponse creates a complete set of responses for the Request phase.
// It simulates the EPP deciding to:
//  1. Modify headers (e.g., set destination endpoint).
//  2. Replace the entire request body (e.g., rewriting the model name).
//
// It returns two messages: one for the Header response and one for the Body response.
func NewRequestBufferedResponse(
	destinationEndpoint string,
	rewrittenBody string,
	otherHeaders ...*envoyCorev3.HeaderValueOption,
) []*extProcPb.ProcessingResponse {
	setHeaders := []*envoyCorev3.HeaderValueOption{
		{
			Header: &envoyCorev3.HeaderValue{
				Key:      metadata.DestinationEndpointKey,
				RawValue: []byte(destinationEndpoint),
			},
		},
		{
			Header: &envoyCorev3.HeaderValue{
				Key:      headerKeyContentLength,
				RawValue: []byte(strconv.Itoa(len(rewrittenBody))),
			},
		},
	}
	setHeaders = append(setHeaders, otherHeaders...)

	headerResponse := &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestHeaders{
			RequestHeaders: &extProcPb.HeadersResponse{
				Response: &extProcPb.CommonResponse{
					ClearRouteCache: true,
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: setHeaders,
					},
				},
			},
		},
		DynamicMetadata: makeDestinationMetadata(destinationEndpoint),
	}

	bodyResponse := &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestBody{
			RequestBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					BodyMutation: &extProcPb.BodyMutation{
						Mutation: &extProcPb.BodyMutation_StreamedResponse{
							StreamedResponse: &extProcPb.StreamedBodyResponse{
								Body:        []byte(rewrittenBody),
								EndOfStream: true,
							},
						},
					},
				},
			},
		},
	}

	return []*extProcPb.ProcessingResponse{headerResponse, bodyResponse}
}

// NewResponseBufferedResponse creates a complete set of responses for the Response phase.
// It simulates the EPP modifying the upstream response before sending it to the client.
// It returns a Header mutation message followed by a Body replacement message.
func NewResponseBufferedResponse(
	rewrittenBody string,
	headersToSet ...*envoyCorev3.HeaderValueOption,
) []*extProcPb.ProcessingResponse {
	return []*extProcPb.ProcessingResponse{
		NewResponseHeaders(headersToSet...),
		NewResponseStreamChunk(rewrittenBody, true),
	}
}

// NewResponseHeaders creates a single response message to modify response headers.
// Use this when testing header mutations without body changes, or as the first step in a streamed response test.
func NewResponseHeaders(headersToSet ...*envoyCorev3.HeaderValueOption) *extProcPb.ProcessingResponse {
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseHeaders{
			ResponseHeaders: &extProcPb.HeadersResponse{
				Response: &extProcPb.CommonResponse{
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: headersToSet,
					},
				},
			},
		},
	}
}

// NewResponseStreamChunk creates a single gRPC message representing one chunk of a streaming response.
// Use this to verify that EPP correctly passes through chunks (e.g., SSE events) or injects specific chunks.
func NewResponseStreamChunk(body string, endOfStream bool) *extProcPb.ProcessingResponse {
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseBody{
			ResponseBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					BodyMutation: &extProcPb.BodyMutation{
						Mutation: &extProcPb.BodyMutation_StreamedResponse{
							StreamedResponse: &extProcPb.StreamedBodyResponse{
								Body:        []byte(body),
								EndOfStream: endOfStream,
							},
						},
					},
				},
			},
		},
	}
}

// NewImmediateErrorResponse creates a response that immediately terminates the request with a specific HTTP status code
// and body.
// Use this for testing Load Shedding (503), Rate Limiting (429), or Bad Request (400) logic.
func NewImmediateErrorResponse(code envoyTypePb.StatusCode, body string) []*extProcPb.ProcessingResponse {
	return []*extProcPb.ProcessingResponse{{
		Response: &extProcPb.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &extProcPb.ImmediateResponse{
				Status: &envoyTypePb.HttpStatus{Code: code},
				Body:   []byte(body),
			},
		},
	}}
}

// --- Execution Helpers ---

// SendRequest is a helper for Unary (One-Shot) test scenarios.
// It sends a single request message and waits for exactly one response.
func SendRequest(
	t *testing.T,
	client extProcPb.ExternalProcessor_ProcessClient,
	req *extProcPb.ProcessingRequest,
) (*extProcPb.ProcessingResponse, error) {
	t.Helper()
	t.Logf("Sending request: %v", req)

	if err := client.Send(req); err != nil {
		t.Logf("Failed to send request: %v", err)
		return nil, err
	}

	res, err := client.Recv()
	if err != nil {
		t.Logf("Failed to receive response: %v", err)
		return nil, err
	}
	t.Logf("Received response: %+v", res)
	return res, err
}

// StreamedRequest is a helper for Full-Duplex Streaming test scenarios.
// It performs the following actions:
//  1. Sends all requests in the provided slice to the server.
//  2. Listens for responses on the stream until 'expectedResponses' count is reached.
//  3. Enforces a 10-second timeout to prevent deadlocks if the server hangs.
//  4. Handles io.EOF gracefully (server closed stream).
func StreamedRequest(
	t *testing.T,
	client extProcPb.ExternalProcessor_ProcessClient,
	requests []*extProcPb.ProcessingRequest,
	expectedResponses int,
) ([]*extProcPb.ProcessingResponse, error) {
	t.Helper()

	// 1. Send Phase
	for _, req := range requests {
		t.Logf("Sending request: %v", req)
		if err := client.Send(req); err != nil {
			t.Logf("Failed to send request: %v", err)
			return nil, err
		}
	}

	// 2. Receive Phase
	// We use a channel and a separate goroutine for receiving to allow for a strict timeout via select{}.
	type recvResult struct {
		res *extProcPb.ProcessingResponse
		err error
	}

	// Buffered channel avoids blocking the goroutine on the last read.
	recvChan := make(chan recvResult, expectedResponses+1)

	// Start reading in background.
	go func() {
		for range expectedResponses {
			res, err := client.Recv()
			recvChan <- recvResult{res, err}
			if err != nil {
				return // Stop reading on error or EOF.
			}
		}
	}()

	var responses []*extProcPb.ProcessingResponse

	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()

	// Collect results with timeout.
	for i := range expectedResponses {
		select {
		case <-ctx.Done():
			t.Logf("Timeout waiting for response %d of %d: %v", i+1, expectedResponses, ctx.Err())
			return responses, fmt.Errorf("timeout waiting for responses: %w", ctx.Err())

		case result := <-recvChan:
			if result.err != nil {
				// io.EOF is a valid termination from the server side (e.g. rejection).
				if result.err == io.EOF {
					return responses, nil
				}
				t.Logf("Failed to receive: %v", result.err)
				return nil, result.err
			}
			t.Logf("Received response: %+v", result.res)
			responses = append(responses, result.res)
		}
	}

	return responses, nil
}

// --- System Utilities ---

// GetFreePort finds an available IPv4 TCP port on localhost.
// It works by asking the OS to allocate a port by listening on port 0, capturing the assigned address, and then
// immediately closing the listener.
//
// Note: There is a theoretical race condition where another process grabs the port between the Close() call and the
// subsequent usage, but this is generally acceptable in hermetic test environments.
func GetFreePort() (*net.TCPAddr, error) {
	// Force IPv4 to prevent flakes on dual-stack CI environments
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("failed to listen on a free port: %w", err)
	}

	// Critical: Close the listener immediately so the caller can bind to it.
	defer listener.Close()

	addr, ok := listener.Addr().(*net.TCPAddr)
	if !ok {
		return nil, errors.New("failed to cast listener address to TCPAddr")
	}
	return addr, nil
}

// --- Internal Helpers ---

// makeDestinationMetadata helper to construct the Envoy dynamic metadata for routing.
func makeDestinationMetadata(endpoint string) *structpb.Struct {
	return &structpb.Struct{
		Fields: map[string]*structpb.Value{
			metadata.DestinationEndpointNamespace: {
				Kind: &structpb.Value_StructValue{
					StructValue: &structpb.Struct{
						Fields: map[string]*structpb.Value{
							metadata.DestinationEndpointKey: {
								Kind: &structpb.Value_StringValue{
									StringValue: endpoint,
								},
							},
						},
					},
				},
			},
		},
	}
}

const (
	headerKeyContentLength = "Content-Length"
)
