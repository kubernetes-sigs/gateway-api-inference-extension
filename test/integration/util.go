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

package integration

import (
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
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

const (
	headerKeyContentLength = "Content-Length"
)

// GetFreePort finds and returns an available TCP port on the host.
// It works by asking the OS to allocate a port by listening on port 0, capturing the assigned address, and then
// immediately closing the listener.
func GetFreePort() (*net.TCPAddr, error) {
	// A port number of 0 instructs the OS to select a random, available port.
	listener, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		return nil, fmt.Errorf("failed to listen on a free port: %w", err)
	}
	defer listener.Close()

	addr, ok := listener.Addr().(*net.TCPAddr)
	if !ok {
		return nil, errors.New("failed to cast listener address to TCPAddr")
	}
	return addr, nil
}

func SendRequest(t *testing.T, client extProcPb.ExternalProcessor_ProcessClient, req *extProcPb.ProcessingRequest) (*extProcPb.ProcessingResponse, error) {
	t.Logf("Sending request: %v", req)
	if err := client.Send(req); err != nil {
		t.Logf("Failed to send request %+v: %v", req, err)
		return nil, err
	}

	res, err := client.Recv()
	if err != nil {
		t.Logf("Failed to receive: %v", err)
		return nil, err
	}
	t.Logf("Received response %+v", res)
	return res, err
}

// StreamedRequest sends a series of requests and collects the specified number of responses.
func StreamedRequest(
	t *testing.T,
	client extProcPb.ExternalProcessor_ProcessClient,
	requests []*extProcPb.ProcessingRequest,
	expectedResponses int,
) ([]*extProcPb.ProcessingResponse, error) {
	for _, req := range requests {
		t.Logf("Sending request: %v", req)
		if err := client.Send(req); err != nil {
			t.Logf("Failed to send request %+v: %v", req, err)
			return nil, err
		}
	}

	var responses []*extProcPb.ProcessingResponse
	for i := range expectedResponses {
		type recvResult struct {
			res *extProcPb.ProcessingResponse
			err error
		}
		recvChan := make(chan recvResult, 1)

		go func() {
			res, err := client.Recv()
			recvChan <- recvResult{res, err}
		}()

		select {
		case <-time.After(10 * time.Second):
			t.Logf("Timeout waiting for response %d of %d", i+1, expectedResponses)
			return responses, nil
		case result := <-recvChan:
			if result.err != nil {
				if result.err == io.EOF {
					return responses, nil
				}
				t.Logf("Failed to receive: %v", result.err)
				return nil, result.err
			}
			t.Logf("Received response %+v", result.res)
			responses = append(responses, result.res)
		}
	}
	return responses, nil
}

func GenerateRequest(logger logr.Logger, prompt, model string, filterMetadata []string) *extProcPb.ProcessingRequest {
	j := map[string]any{
		"prompt":      prompt,
		"max_tokens":  100,
		"temperature": 0,
	}
	if model != "" {
		j["model"] = model
	}

	llmReq, err := json.Marshal(j)
	if err != nil {
		logutil.Fatal(logger, err, "Failed to unmarshal LLM request")
	}
	req := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{Body: llmReq, EndOfStream: true},
		},
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: GenerateRequestMetadata(filterMetadata),
		},
	}
	return req
}

func GenerateStreamedRequestSet(logger logr.Logger, prompt, model, targetModel string, filterMetadata []string) []*extProcPb.ProcessingRequest {
	return GenerateStreamedRequestSetWithHeaders(logger, prompt, model, targetModel, filterMetadata, nil)
}

// GenerateStreamedRequestSetWithHeaders creates a complete set of gRPC messages to simulate a realistic, multi-chunk
// HTTP request. It includes a headers message, followed by two body messages, which is representative of how Envoy
// streams request bodies. It allows adding extra headers for specialized test cases.
func GenerateStreamedRequestSetWithHeaders(logger logr.Logger, prompt, model, targetModel string, filterMetadata []string, extraHeaders map[string]string) []*extProcPb.ProcessingRequest {
	requests := []*extProcPb.ProcessingRequest{}
	headers := []*envoyCorev3.HeaderValue{
		{
			Key:   "hi",
			Value: "mom",
		},
		{
			Key:   metadata.ObjectiveKey,
			Value: model,
		},
		{
			Key:   metadata.ModelNameRewriteKey,
			Value: targetModel,
		},
		{
			Key:   requtil.RequestIdHeaderKey,
			Value: "test-request-id",
		},
	}

	for k, v := range extraHeaders {
		headers = append(headers, &envoyCorev3.HeaderValue{
			Key:   k,
			Value: v,
		})
	}

	headerReq := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestHeaders{
			RequestHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{
					Headers: headers,
				},
			},
		},
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: GenerateRequestMetadata(filterMetadata),
		},
	}
	requests = append(requests, headerReq)

	// Create and split the request body.
	j := map[string]any{
		"prompt":      prompt,
		"max_tokens":  100,
		"temperature": 0,
	}
	if model != "" {
		j["model"] = model
	}
	llmReq, err := json.Marshal(j)
	if err != nil {
		logutil.Fatal(logger, err, "Failed to marshal LLM request")
	}

	// Simulate a multi-chunk body by splitting the marshaled JSON.
	// This is a more realistic representation of how a streaming body might arrive.
	splitPoint := len(llmReq) / 2
	chunk1 := llmReq[:splitPoint]
	chunk2 := llmReq[splitPoint:]

	requests = append(requests, &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{Body: chunk1, EndOfStream: false},
		},
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: GenerateRequestMetadata(filterMetadata),
		},
	})
	requests = append(requests, &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{Body: chunk2, EndOfStream: true},
		},
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: GenerateRequestMetadata(filterMetadata),
		},
	})

	return requests
}

// GenerateRequestMetadata constructs the nested metadata structure required by Envoy for subset load balancing.
// It takes a list of endpoint addresses and embeds them into the `envoy.lb` filter metadata field.
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

// NewRequestBufferedResponse simulates a complete buffered mutation of the request phase. It returns a slice of
// two messages: one to replace the request headers (for routing) and one to replace the request body.
func NewRequestBufferedResponse(destinationEndpoint string, rewrittenBody string, otherHeaders ...*envoyCorev3.HeaderValueOption) []*extProcPb.ProcessingResponse {
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
		DynamicMetadata: MakeMetadata(destinationEndpoint),
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

// NewResponseBufferedResponse simulates a complete buffered mutation of the response phase. It returns a slice of
// messages to first modify the response headers and then replace the entire response body.
func NewResponseBufferedResponse(rewrittenBody string, headersToSet ...*envoyCorev3.HeaderValueOption) []*extProcPb.ProcessingResponse {
	return []*extProcPb.ProcessingResponse{
		NewResponseHeaders(headersToSet...),
		NewResponseStreamChunk(rewrittenBody, true),
	}
}

// NewResponseHeaders creates a single response message to modify the response headers.
// This is the first step in either a buffered or streaming response modification.
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

// NewResponseStreamChunk creates a single gRPC message to send one chunk of a streaming response body.
// This is used to test streaming behaviors, such as passing through a text/event-stream.
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

// NewImmediateErrorResponse creates an immediate response to terminate processing.
// This is used for errors like load shedding or bad requests.
func NewImmediateErrorResponse(code envoyTypePb.StatusCode, body string) []*extProcPb.ProcessingResponse {
	response := &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &extProcPb.ImmediateResponse{
				Status: &envoyTypePb.HttpStatus{
					Code: code,
				},
				Body: []byte(body),
			},
		},
	}
	return []*extProcPb.ProcessingResponse{response}
}

// MakeMetadata creates the dynamic metadata struct that Envoy uses for routing hints.
func MakeMetadata(endpoint string) *structpb.Struct {
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
