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

package server

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	pb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
	testutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
	"sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

const (
	bufSize    = 1024 * 1024
	podName    = "pod1"
	podAddress = "1.2.3.4"
	poolPort   = int32(5678)
	namespace  = "ns1"
)

func TestServer(t *testing.T) {
	tests := []struct {
		name              string
		streamingResponse bool
		hasTrailer        bool
	}{
		{
			name:              "Streaming response with trailers",
			streamingResponse: true,
			hasTrailer:        true,
		},
		{
			name:              "Non-streaming response with trailers",
			streamingResponse: false,
			hasTrailer:        true,
		},
		{
			name:              "Streaming response without trailers",
			streamingResponse: true,
			hasTrailer:        false,
		},
		{
			name:              "Non-streaming response without trailers",
			streamingResponse: false,
			hasTrailer:        false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			runStreamingTest(t, test.streamingResponse, test.hasTrailer)
		})
	}
}

func runStreamingTest(t *testing.T, streamingResponse bool, hasTrailers bool) {
	t.Helper()

	expectedRequestHeaders := map[string]string{metadata.DestinationEndpointKey: fmt.Sprintf("%s:%d", podAddress, poolPort),
		"Content-Length": "42", ":method": "POST", "x-test": "body", "x-request-id": "test-request-id"}
	expectedResponseHeaders := map[string]string{"x-went-into-resp-headers": "true", ":method": "POST", "x-test": "body"}
	expectedSchedulerHeaders := map[string]string{":method": "POST", "x-test": "body", "x-request-id": "test-request-id"}

	model := testutil.MakeInferenceObjective("v1").
		CreationTimestamp(metav1.Unix(1000, 0)).ObjRef()

	director := &testDirector{}
	ctx, cancel, ds, _ := utils.PrepareForTestExtProcServer([]*v1alpha2.InferenceObjective{model},
		[]*v1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: podName}}}, "test-pool1", namespace, poolPort)
	extProcServer := handlers.NewServer(ds, director, openai.NewOpenAIParser())

	testListener, errChan := utils.SetupTestExtProcServer(t, ctx, ds, extProcServer)
	process, conn := utils.GetExtProcServerClient(ctx, t)
	defer conn.Close()

	// Send request headers - no response expected
	headers := utils.BuildEnvoyGRPCHeaders(map[string]string{
		"x-test":                   "body",
		":method":                  "POST",
		metadata.FlowFairnessIDKey: "a-very-interesting-fairness-id",
		"x-request-id":             "test-request-id",
	}, true)
	request := &pb.ProcessingRequest{
		Request: &pb.ProcessingRequest_RequestHeaders{
			RequestHeaders: headers,
		},
	}
	err := process.Send(request)
	if err != nil {
		t.Error("Error sending request headers", err)
	}

	// Send request body
	requestBody := "{\"model\":\"food-review\",\"prompt\":\"Is banana tasty?\"}"
	expectedBody := "{\"model\":\"v1\",\"prompt\":\"Is banana tasty?\"}"
	request = &pb.ProcessingRequest{
		Request: &pb.ProcessingRequest_RequestBody{
			RequestBody: &pb.HttpBody{
				Body:        []byte(requestBody),
				EndOfStream: true,
			},
		},
	}
	err = process.Send(request)
	if err != nil {
		t.Error("Error sending request body", err)
	}

	// Receive request headers and check
	responseReqHeaders, err := process.Recv()
	if err != nil {
		t.Error("Error receiving response", err)
	} else {
		if responseReqHeaders == nil || responseReqHeaders.GetRequestHeaders() == nil ||
			responseReqHeaders.GetRequestHeaders().Response == nil ||
			responseReqHeaders.GetRequestHeaders().Response.HeaderMutation == nil ||
			responseReqHeaders.GetRequestHeaders().Response.HeaderMutation.SetHeaders == nil {
			t.Error("Invalid request headers response")
		} else if !utils.CheckEnvoyGRPCHeaders(t, responseReqHeaders.GetRequestHeaders().Response, expectedRequestHeaders) {
			t.Error("Incorrect request headers")
		}
	}

	// Receive request body and check
	responseReqBody, err := process.Recv()
	if err != nil {
		t.Error("Error receiving response", err)
	} else {
		if responseReqBody == nil || responseReqBody.GetRequestBody() == nil ||
			responseReqBody.GetRequestBody().Response == nil ||
			responseReqBody.GetRequestBody().Response.BodyMutation == nil ||
			responseReqBody.GetRequestBody().Response.BodyMutation.GetStreamedResponse() == nil {
			t.Error("Invalid request body response")
		} else {
			body := responseReqBody.GetRequestBody().Response.BodyMutation.GetStreamedResponse().Body
			if string(body) != expectedBody {
				t.Errorf("Incorrect body %s expected %s", string(body), expectedBody)
			}
		}
	}

	// Check headers passed to the scheduler
	for expectedKey, expectedValue := range expectedSchedulerHeaders {
		got, ok := director.requestHeaders[expectedKey]
		if !ok {
			t.Errorf("Missing header %s", expectedKey)
		} else if got != expectedValue {
			t.Errorf("Incorrect value for header %s, want %s got %s", expectedKey, expectedValue, got)
		}
	}

	// Send response headers
	if streamingResponse {
		// If response is streaming, the header should have text/event-stream.
		headers = utils.BuildEnvoyGRPCHeaders(map[string]string{"x-test": "body", ":method": "POST", "content-type": "text/event-stream"}, true)
		expectedResponseHeaders = map[string]string{"x-went-into-resp-headers": "true", ":method": "POST", "x-test": "body", "content-type": "text/event-stream"}
	} else {
		headers = utils.BuildEnvoyGRPCHeaders(map[string]string{"x-test": "body", ":method": "POST"}, false)
	}

	request = &pb.ProcessingRequest{
		Request: &pb.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: headers,
		},
	}
	err = process.Send(request)
	if err != nil {
		t.Error("Error sending response", err)
	}

	// Receive response headers and check
	response, err := process.Recv()
	if err != nil {
		t.Error("Error receiving response", err)
	} else {
		if response == nil || response.GetResponseHeaders() == nil || response.GetResponseHeaders().Response == nil ||
			response.GetResponseHeaders().Response.HeaderMutation == nil ||
			response.GetResponseHeaders().Response.HeaderMutation.SetHeaders == nil {
			t.Error("Invalid response")
		} else if !utils.CheckEnvoyGRPCHeaders(t, response.GetResponseHeaders().Response, expectedResponseHeaders) {
			t.Error("Incorrect response headers")
		}
	}

	// EndOfStream (eos) is true ONLY if we are NOT sending trailers later.
	if err := sendResponseBody(process, []byte("test response body"), !hasTrailers); err != nil {
		t.Fatalf("failed to send response body: %v", err)
	}

	switch {
	case !hasTrailers:
		if err := recvResponseBody(process); err != nil {
			t.Fatalf("failed to receive response body (no trailers case): %v", err)
		}
	case streamingResponse:
		// For streaming case, ext_proc will first receive the response before getting trailers.
		if err := recvResponseBody(process); err != nil {
			t.Fatalf("failed to receive response body (streaming case): %v", err)
		}
		if err := sendResponseTrailers(process); err != nil {
			t.Fatalf("failed to send response trailers (streaming case): %v", err)
		}
		if err := recvResponseTrailers(process); err != nil {
			t.Fatalf("failed to receive response trailers (streaming case): %v", err)
		}
	default:
		// For non-streaming case, ext_proc will receive response until the trailer is sent by the client.
		if err := sendResponseTrailers(process); err != nil {
			t.Fatalf("failed to send response trailers (non-streaming case): %v", err)
		}
		if err := recvResponseBody(process); err != nil {
			t.Fatalf("failed to receive response body (non-streaming case): %v", err)
		}
		if err := recvResponseTrailers(process); err != nil {
			t.Fatalf("failed to receive response trailers (non-streaming case): %v", err)
		}
	}

	cancel()
	<-errChan
	testListener.Close()
}

func sendResponseBody(stream pb.ExternalProcessor_ProcessClient, data []byte, eos bool) error {
	req := &pb.ProcessingRequest{
		Request: &pb.ProcessingRequest_ResponseBody{
			ResponseBody: &pb.HttpBody{
				Body:        data,
				EndOfStream: eos,
			},
		},
	}
	if err := stream.Send(req); err != nil {
		return fmt.Errorf("sendBody failed: %w", err)
	}
	return nil
}

func sendResponseTrailers(stream pb.ExternalProcessor_ProcessClient) error {
	req := &pb.ProcessingRequest{
		Request: &pb.ProcessingRequest_ResponseTrailers{
			ResponseTrailers: &pb.HttpTrailers{},
		},
	}
	if err := stream.Send(req); err != nil {
		return fmt.Errorf("sendTrailers failed: %w", err)
	}
	return nil
}

func recvResponseBody(stream pb.ExternalProcessor_ProcessClient) error {
	resp, err := stream.Recv()
	if err != nil {
		return fmt.Errorf("recvBody failed: %w", err)
	}
	if resp == nil || resp.GetResponseBody() == nil {
		return fmt.Errorf("recvBody: expected ResponseBody, got %v", resp)
	}
	return nil
}

func recvResponseTrailers(stream pb.ExternalProcessor_ProcessClient) error {
	resp, err := stream.Recv()
	if err != nil {
		return fmt.Errorf("recvTrailers failed: %w", err)
	}
	if resp == nil || resp.GetResponseTrailers() == nil {
		return fmt.Errorf("recvTrailers: expected ResponseTrailers, got %v", resp)
	}
	return nil
}

type testDirector struct {
	requestHeaders map[string]string
}

func (ts *testDirector) HandleRequest(ctx context.Context, reqCtx *handlers.RequestContext) (*handlers.RequestContext, error) {
	ts.requestHeaders = reqCtx.Request.Headers

	bodyMap := make(map[string]any)
	if err := json.Unmarshal(reqCtx.Request.RawBody, &bodyMap); err != nil {
		return reqCtx, err
	}
	bodyMap["model"] = "v1"

	var marshalErr error
	reqCtx.Request.RawBody, marshalErr = json.Marshal(bodyMap)
	if marshalErr != nil {
		return reqCtx, marshalErr
	}
	reqCtx.RequestSize = len(reqCtx.Request.RawBody)
	reqCtx.TargetEndpoint = fmt.Sprintf("%s:%d", podAddress, poolPort)
	return reqCtx, nil
}

func (ts *testDirector) HandleResponseReceived(ctx context.Context, reqCtx *handlers.RequestContext) (*handlers.RequestContext, error) {
	return reqCtx, nil
}

func (ts *testDirector) HandleResponseBodyStreaming(ctx context.Context, reqCtx *handlers.RequestContext) (*handlers.RequestContext, error) {
	return reqCtx, nil
}

func (ts *testDirector) HandleResponseBodyComplete(ctx context.Context, reqCtx *handlers.RequestContext) (*handlers.RequestContext, error) {
	return reqCtx, nil
}

func (ts *testDirector) GetRandomEndpoint() *fwkdl.EndpointMetadata {
	return nil
}
