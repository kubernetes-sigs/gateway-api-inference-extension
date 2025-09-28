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
	"fmt"
	"testing"

	pb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
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

type testDirector struct {
	requestHeaders map[string]string
}

func (ts *testDirector) HandleRequest(ctx context.Context, reqCtx *handlers.RequestContext) (*handlers.RequestContext, error) {
	ts.requestHeaders = reqCtx.Request.Headers

	reqCtx.Request.Body["model"] = "v1"
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

func (ts *testDirector) GetRandomPod() *backend.Pod {
	return nil
}
