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
	"io"
	"strconv"
	"testing"
	"time"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"google.golang.org/protobuf/types/known/structpb"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	headerKeyDestination   = "x-gateway-destination-endpoint"
	headerKeyContentLength = "Content-Length"
)

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

func StreamedRequest(t *testing.T, client extProcPb.ExternalProcessor_ProcessClient, requests []*extProcPb.ProcessingRequest, expectedResponses int) ([]*extProcPb.ProcessingResponse, error) {
	for _, req := range requests {
		t.Logf("Sending request: %v", req)
		if err := client.Send(req); err != nil {
			t.Logf("Failed to send request %+v: %v", req, err)
			return nil, err
		}
	}
	responses := []*extProcPb.ProcessingResponse{}

	// Make an incredible simple timeout func in the case where
	// there is less than the expected amount of responses; bail and fail.
	var simpleTimeout bool
	go func() {
		time.Sleep(10 * time.Second)
		simpleTimeout = true
	}()

	for range expectedResponses {
		if simpleTimeout {
			break
		}
		res, err := client.Recv()
		if err != nil && err != io.EOF {
			t.Logf("Failed to receive: %v", err)
			return nil, err
		}
		t.Logf("Received response %+v", res)
		responses = append(responses, res)
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

func GenerateStreamedRequestSet(logger logr.Logger, prompt, model string, filterMetadata []string) []*extProcPb.ProcessingRequest {
	requests := []*extProcPb.ProcessingRequest{}
	headerReq := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestHeaders{
			RequestHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{
					Headers: []*envoyCorev3.HeaderValue{
						{
							Key:   "hi",
							Value: "mom",
						},
						{
							Key:   handlers.ObjectiveKey,
							Value: model,
						},
					},
				},
			},
		},
	}

	headerReq.MetadataContext = &envoyCorev3.Metadata{
		FilterMetadata: GenerateRequestMetadata(filterMetadata),
	}

	requests = append(requests, headerReq)
	requests = append(requests, GenerateRequest(logger, prompt, model, filterMetadata))
	return requests
}

func GenerateRequestMetadata(filterMetadata []string) map[string]*structpb.Struct {
	metadata := make(map[string]*structpb.Struct)
	interfaceList := make([]any, len(filterMetadata))
	for i, val := range filterMetadata {
		interfaceList[i] = val
	}
	if filterMetadata != nil {
		structVal, _ := structpb.NewStruct(map[string]any{
			"x-gateway-destination-endpoint-subset": interfaceList,
		})
		metadata["envoy.lb.subset_hint"] = structVal
	}
	return metadata
}

// NewRequestBufferedResponse creates a complete set of responses for the request phase.
// It modifies request headers (e.g., for routing) and replaces the entire request body.
// It returns a slice of two messages, representing the complete buffered action.
func NewRequestBufferedResponse(
	destinationEndpoint string,
	rewrittenBody string,
	otherHeaders ...*envoyCorev3.HeaderValueOption,
) []*extProcPb.ProcessingResponse {
	setHeaders := []*envoyCorev3.HeaderValueOption{
		{
			Header: &envoyCorev3.HeaderValue{
				Key:      headerKeyDestination,
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
		DynamicMetadata: makeMetadata(destinationEndpoint),
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

// NewResponseBufferedResponse creates a complete set of responses for the response phase.
// It modifies response headers and replaces the entire response body.
// It is used when the processor buffers the upstream response before sending its own.
func NewResponseBufferedResponse(
	rewrittenBody string,
	headersToSet ...*envoyCorev3.HeaderValueOption,
) []*extProcPb.ProcessingResponse {
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

// NewResponseStreamChunk creates a single response for one body chunk in a stream.
// This is used to test streaming behaviors like text/event-stream pass-through.
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

// makeMetadata creates the dynamic metadata struct that Envoy uses for routing hints.
func makeMetadata(endpoint string) *structpb.Struct {
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
