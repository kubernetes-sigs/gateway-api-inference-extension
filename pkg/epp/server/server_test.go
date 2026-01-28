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
	"encoding/binary"
	"fmt"
	"strconv"
	"testing"

	pb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/protobuf/proto"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	vllm "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/api/gen"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/codec"

	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
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
	expectedRequestHeaders := map[string]string{metadata.DestinationEndpointKey: fmt.Sprintf("%s:%d", podAddress, poolPort),
		"Content-Length": "42", ":method": "POST", "x-test": "body", "x-request-id": "test-request-id"}
	expectedResponseHeaders := map[string]string{"x-went-into-resp-headers": "true", ":method": "POST", "x-test": "body"}
	expectedSchedulerHeaders := map[string]string{":method": "POST", "x-test": "body", "x-request-id": "test-request-id"}

	t.Run("server", func(t *testing.T) {
		model := testutil.MakeInferenceObjective("v1").
			CreationTimestamp(metav1.Unix(1000, 0)).ObjRef()

		director := &testDirector{}
		ctx, cancel, ds, _ := utils.PrepareForTestStreamingServer([]*v1alpha2.InferenceObjective{model},
			[]*v1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: podName}}}, "test-pool1", namespace, poolPort)
		streamingServer := handlers.NewStreamingServer(ds, director)

		testListener, errChan := utils.SetupTestStreamingServer(t, ctx, ds, streamingServer)
		process, conn := utils.GetStreamingServerClient(ctx, t)
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
		headers = utils.BuildEnvoyGRPCHeaders(map[string]string{"x-test": "body", ":method": "POST"}, false)
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

		cancel()
		<-errChan
		testListener.Close()
	})
}

// TestServerGRPC is test cases for gRPC-in-gRPC-out case.
func TestServerGRPC(t *testing.T) {
	vllmReq := &vllm.GenerateRequest{
		Input: &vllm.GenerateRequest_Tokenized{
			Tokenized: &vllm.TokenizedInput{
				OriginalText: "Hello gRPC",
				InputIds:     []uint32{0, 1, 2, 3, 4}, // Fake tokens.
			},
		},
	}
	grpcPayload, err := toGrpcFrame(vllmReq)
	if err != nil {
		t.Fatalf("Failed to encode gRPC payload: %v", err)
	}

	vllmResp := &vllm.GenerateResponse{
		Response: &vllm.GenerateResponse_Complete{
			Complete: &vllm.GenerateComplete{
				OutputIds: []uint32{0, 1, 2, 3, 4},
			},
		},
	}
	grpcResponsePayload, err := toGrpcFrame(vllmResp)
	if err != nil {
		t.Fatalf("Failed to envode gRPC response paylaod: %v", err)
	}

	expectedRequestHeaders := map[string]string{
		metadata.DestinationEndpointKey: fmt.Sprintf("%s:%d", podAddress, poolPort),
		":method":                       "POST",
		"x-request-id":                  "test-request-id",
		"content-type":                  "application/grpc",
		":path":                         handlers.VllmGeneratePath,
		"Content-Length":                strconv.Itoa(len(grpcPayload)),
	}
	expectedResponseHeaders := map[string]string{"x-went-into-resp-headers": "true", ":method": "POST", "x-test": "body"}

	t.Run("server-grpc", func(t *testing.T) {
		model := testutil.MakeInferenceObjective("v1").
			CreationTimestamp(metav1.Unix(1000, 0)).ObjRef()

		director := &testDirector{}
		ctx, cancel, ds, _ := utils.PrepareForTestStreamingServer([]*v1alpha2.InferenceObjective{model},
			[]*v1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: podName}}}, "test-pool1", namespace, poolPort)
		streamingServer := handlers.NewStreamingServer(ds, director)

		testListener, errChan := utils.SetupTestStreamingServer(t, ctx, ds, streamingServer)
		process, conn := utils.GetStreamingServerClient(ctx, t)
		defer conn.Close()

		t.Log("Sending request headers")

		// Send request headers - no response expected
		headers := utils.BuildEnvoyGRPCHeaders(map[string]string{
			":method":                  "POST",
			"content-type":             "application/grpc",
			":path":                    handlers.VllmGeneratePath,
			metadata.FlowFairnessIDKey: "a-very-interesting-fairness-id",
			"x-request-id":             "test-request-id",
		}, true)
		request := &pb.ProcessingRequest{
			Request: &pb.ProcessingRequest_RequestHeaders{
				RequestHeaders: headers,
			},
		}
		if err := process.Send(request); err != nil {
			t.Error("Error sending request headers", err)
		}

		request = &pb.ProcessingRequest{
			Request: &pb.ProcessingRequest_RequestBody{
				RequestBody: &pb.HttpBody{
					Body:        grpcPayload,
					EndOfStream: true,
				},
			},
		}
		if err := process.Send(request); err != nil {
			t.Error("Error sending request body", err)
		}

		// Receive response headers and check
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
				// Verify the body is the same (since we didn't rewrite it in this test setup)
				// Or check if it's a valid gRPC payload that decodes to the same content
				decodedReq, _, err := codec.ConvertToLLMRequestBody(body)
				if err != nil {
					t.Errorf("Failed to decode response body: %v", err)
				}
				if decodedReq.Completions.Prompt != "Hello gRPC" {
					t.Errorf("Expected prompt 'Hello gRPC', got '%s'", decodedReq.Completions.Prompt)
				}
			}
		}

		// Send response headers
		headers = utils.BuildEnvoyGRPCHeaders(map[string]string{"x-test": "body", ":method": "POST", "content-type": "application/grpc"}, true)
		request = &pb.ProcessingRequest{
			Request: &pb.ProcessingRequest_ResponseHeaders{
				ResponseHeaders: headers,
			},
		}
		if err := process.Send(request); err != nil {
			t.Error("Error sending response headers", err)
		}

		// Receive response headers and check
		response, err := process.Recv()
		if err != nil {
			t.Error("Error receiving response headers", err)
		} else if response == nil || response.GetResponseHeaders() == nil {
			if response == nil || response.GetResponseHeaders() == nil || response.GetResponseHeaders().Response == nil ||
				response.GetResponseHeaders().Response.HeaderMutation == nil ||
				response.GetResponseHeaders().Response.HeaderMutation.SetHeaders == nil {
				t.Error("Invalid response")
			} else if !utils.CheckEnvoyGRPCHeaders(t, response.GetResponseHeaders().Response, expectedResponseHeaders) {
				t.Error("Incorrect response headers")
			}
		}

		request = &pb.ProcessingRequest{
			Request: &pb.ProcessingRequest_ResponseBody{
				ResponseBody: &pb.HttpBody{
					Body:        grpcResponsePayload,
					EndOfStream: false,
				},
			},
		}
		err = process.Send(request)
		if err != nil {
			t.Error("Error sending ressponse body", err)
		}

		// Sending response trailers
		trailers := utils.BuildEnvoyGRPCTrailers(map[string]string{
			"grpc-status": "0",
		}, false)
		request = &pb.ProcessingRequest{
			Request: &pb.ProcessingRequest_ResponseTrailers{
				ResponseTrailers: trailers,
			},
		}
		if err := process.Send(request); err != nil {
			t.Error("Error sending response trailers", err)
		}

		// Receiving response trailers
		response, err = process.Recv()
		if err != nil {
			t.Error("Error receiving response trailers/body", err)
		} else {
			// We expect a body response (to flush any buffered body if any, or just pass through)
			if response.GetResponseBody() != nil {
				t.Log("Received response body, now waiting for trailers")
				// OK, this is likely the body completion.
				// Receive next for trailers.
				response, err = process.Recv()
				if err != nil {
					t.Error("Error receiving trailers after body", err)
				}
			}

			if response.GetResponseTrailers() == nil {
				t.Errorf("Expected ResponseTrailers, got %v", response.Response)
			}
		}

		cancel()
		<-errChan
		testListener.Close()
	})
}

// TestServerGRPC is test cases for gRPC-in-gRPC-out case.
func TestServerGRPC_GenerateStreaming(t *testing.T) {
	vllmReq := &vllm.GenerateRequest{
		Input: &vllm.GenerateRequest_Tokenized{
			Tokenized: &vllm.TokenizedInput{
				OriginalText: "Hello gRPC",
				InputIds:     []uint32{0, 1, 2, 3, 4}, // Fake tokens.
			},
		},
		Stream: true,
	}
	grpcPayload, err := toGrpcFrame(vllmReq)
	if err != nil {
		t.Fatalf("Failed to encode gRPC payload: %v", err)
	}

	vllmResp := &vllm.GenerateResponse{
		Response: &vllm.GenerateResponse_Complete{
			Complete: &vllm.GenerateComplete{
				OutputIds: []uint32{0, 1, 2, 3, 4},
			},
		},
	}
	grpcResponsePayload, err := toGrpcFrame(vllmResp)
	if err != nil {
		t.Fatalf("Failed to envode gRPC response paylaod: %v", err)
	}

	expectedRequestHeaders := map[string]string{
		metadata.DestinationEndpointKey: fmt.Sprintf("%s:%d", podAddress, poolPort),
		":method":                       "POST",
		"x-request-id":                  "test-request-id",
		"content-type":                  "application/grpc",
		":path":                         handlers.VllmGeneratePath,
		"Content-Length":                strconv.Itoa(len(grpcPayload)),
	}
	expectedResponseHeaders := map[string]string{"x-went-into-resp-headers": "true", ":method": "POST", "x-test": "body"}

	t.Run("server-grpc", func(t *testing.T) {
		model := testutil.MakeInferenceObjective("v1").
			CreationTimestamp(metav1.Unix(1000, 0)).ObjRef()

		director := &testDirector{}
		ctx, cancel, ds, _ := utils.PrepareForTestStreamingServer([]*v1alpha2.InferenceObjective{model},
			[]*v1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: podName}}}, "test-pool1", namespace, poolPort)
		streamingServer := handlers.NewStreamingServer(ds, director)

		testListener, errChan := utils.SetupTestStreamingServer(t, ctx, ds, streamingServer)
		process, conn := utils.GetStreamingServerClient(ctx, t)
		defer conn.Close()

		t.Log("Sending request headers")

		// Send request headers - no response expected
		headers := utils.BuildEnvoyGRPCHeaders(map[string]string{
			":method":                  "POST",
			"content-type":             "application/grpc",
			":path":                    handlers.VllmGeneratePath,
			metadata.FlowFairnessIDKey: "a-very-interesting-fairness-id",
			"x-request-id":             "test-request-id",
		}, true)
		request := &pb.ProcessingRequest{
			Request: &pb.ProcessingRequest_RequestHeaders{
				RequestHeaders: headers,
			},
		}
		if err := process.Send(request); err != nil {
			t.Error("Error sending request headers", err)
		}

		request = &pb.ProcessingRequest{
			Request: &pb.ProcessingRequest_RequestBody{
				RequestBody: &pb.HttpBody{
					Body:        grpcPayload,
					EndOfStream: true,
				},
			},
		}
		if err := process.Send(request); err != nil {
			t.Error("Error sending request body", err)
		}

		// Receive response headers and check
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
				// Verify the body is the same (since we didn't rewrite it in this test setup)
				// Or check if it's a valid gRPC payload that decodes to the same content
				decodedReq, _, err := codec.ConvertToLLMRequestBody(body)
				if err != nil {
					t.Errorf("Failed to decode response body: %v", err)
				}
				if decodedReq.Completions.Prompt != "Hello gRPC" {
					t.Errorf("Expected prompt 'Hello gRPC', got '%s'", decodedReq.Completions.Prompt)
				}
			}
		}

		// Send response headers
		headers = utils.BuildEnvoyGRPCHeaders(map[string]string{"x-test": "body", ":method": "POST", "content-type": "application/grpc"}, true)
		request = &pb.ProcessingRequest{
			Request: &pb.ProcessingRequest_ResponseHeaders{
				ResponseHeaders: headers,
			},
		}
		if err := process.Send(request); err != nil {
			t.Error("Error sending response headers", err)
		}

		// Receive response headers and check
		response, err := process.Recv()
		if err != nil {
			t.Error("Error receiving response headers", err)
		} else if response == nil || response.GetResponseHeaders() == nil {
			if response == nil || response.GetResponseHeaders() == nil || response.GetResponseHeaders().Response == nil ||
				response.GetResponseHeaders().Response.HeaderMutation == nil ||
				response.GetResponseHeaders().Response.HeaderMutation.SetHeaders == nil {
				t.Error("Invalid response")
			} else if !utils.CheckEnvoyGRPCHeaders(t, response.GetResponseHeaders().Response, expectedResponseHeaders) {
				t.Error("Incorrect response headers")
			}
		}

		request = &pb.ProcessingRequest{
			Request: &pb.ProcessingRequest_ResponseBody{
				ResponseBody: &pb.HttpBody{
					Body:        grpcResponsePayload,
					EndOfStream: false,
				},
			},
		}
		err = process.Send(request)
		if err != nil {
			t.Error("Error sending ressponse body", err)
		}

		// Receiving response
		response, err = process.Recv()
		if err != nil {
			t.Error("Error receiving response body", err)
		} else if response.GetResponseBody() == nil {
			t.Error("Error receiving response body", err)
		}

		// Sending response trailers
		trailers := utils.BuildEnvoyGRPCTrailers(map[string]string{
			"grpc-status": "0",
		}, true)
		request = &pb.ProcessingRequest{
			Request: &pb.ProcessingRequest_ResponseTrailers{
				ResponseTrailers: trailers,
			},
		}
		if err := process.Send(request); err != nil {
			t.Error("Error sending response trailers", err)
		}

		// Receiving response trailers
		response, err = process.Recv()
		if err != nil {
			t.Error("Error receiving response trailers/body", err)
		} else if response.GetResponseTrailers() == nil {
			t.Errorf("Expected ResponseTrailers, got %v", response.Response)
		}

		cancel()
		<-errChan
		testListener.Close()
	})
}

func toGrpcFrame(req proto.Message) ([]byte, error) {
	payload, err := proto.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal protobuf: %w", err)
	}
	// Header: [Compression Flag (1 byte)] [Message Length (4 bytes)]
	msgLen := len(payload)
	frame := make([]byte, 5+msgLen)

	// 0 means uncompressed.
	frame[0] = 0
	binary.BigEndian.PutUint32(frame[1:5], uint32(msgLen))
	copy(frame[5:], payload)

	return frame, nil
}

type testDirector struct {
	requestHeaders map[string]string
}

func (ts *testDirector) HandleRequest(ctx context.Context, reqCtx *handlers.RequestContext) (*handlers.RequestContext, error) {
	ts.requestHeaders = reqCtx.Request.Headers

	reqCtx.Request.Body["model"] = "v1"
	reqCtx.TargetEndpoint = fmt.Sprintf("%s:%d", podAddress, poolPort)

	// Simulate Director populating SchedulingRequest
	if reqCtx.SchedulingRequest == nil {
		var body *schedulingtypes.LLMRequestBody
		if reqCtx.SchedulingRequestBody != nil {
			body = reqCtx.SchedulingRequestBody
		} else {
			// For JSON test, we don't really use SchedulingRequest in the old flow, but we might need it now?
			// The JSON flow in server.go uses reqCtx.Request.Body for marshalling, so it doesn't access SchedulingRequest.Body.
			// But to be safe and consistent:
			body = &schedulingtypes.LLMRequestBody{}
		}
		reqCtx.SchedulingRequest = &schedulingtypes.LLMRequest{
			Body: body,
		}
	}

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
