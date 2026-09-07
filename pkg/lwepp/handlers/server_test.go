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

package handlers

import (
	"context"
	"io"
	"testing"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/metadata"
)

type mockProcessServer struct {
	grpc.ServerStream
	ctx          context.Context
	sentMessages []*extProcPb.ProcessingResponse
	recvMessages []*extProcPb.ProcessingRequest
	recvIndex    int
}

func (m *mockProcessServer) Context() context.Context {
	return m.ctx
}

func (m *mockProcessServer) Send(resp *extProcPb.ProcessingResponse) error {
	m.sentMessages = append(m.sentMessages, resp)
	return nil
}

func (m *mockProcessServer) Recv() (*extProcPb.ProcessingRequest, error) {
	if m.recvIndex >= len(m.recvMessages) {
		return nil, io.EOF
	}
	msg := m.recvMessages[m.recvIndex]
	m.recvIndex++
	return msg, nil
}

func TestProcess_DeferredHeaderMutationOnStreamingBody(t *testing.T) {
	pods := []*datastore.Endpoint{
		{Address: "10.0.0.1", Port: "8080"},
	}
	ds := &mockDatastore{pods: pods}
	server := NewStreamingServer(ds)

	// Construct a standard Envoy request stream:
	// 1. Request headers (with EndOfStream = false)
	// 2. Request body (with EndOfStream = true)
	reqHeaders := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestHeaders{
			RequestHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{
					Headers: []*envoyCorev3.HeaderValue{
						{Key: "test-epp-endpoint-selection", Value: "10.0.0.1"},
					},
				},
				EndOfStream: false,
			},
		},
	}

	reqBody := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{
				Body:        []byte(`{"prompt": "hello"}`),
				EndOfStream: true,
			},
		},
	}
	respBody := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_ResponseBody{
			ResponseBody: &extProcPb.HttpBody{
				Body:        []byte(`{"response": "world"}`),
				EndOfStream: true,
			},
		},
	}

	stream := &mockProcessServer{
		ctx:          context.Background(),
		recvMessages: []*extProcPb.ProcessingRequest{reqHeaders, reqBody, respBody},
	}

	err := server.Process(stream)
	assert.NoError(t, err)

	// Assertions on the sequence of sent responses:
	// - We expect exactly 3 responses sent back.
	// - Response 1: The DEFERRED RequestHeaders response containing the target routing mutation headers.
	// - Response 2: The RequestBody response preserving the original request body.
	// - Response 3: The ResponseBody response preserving the original response body.
	require := assert.New(t)
	if require.Len(stream.sentMessages, 3) {
		// Response 1: RequestHeaders Response
		firstResp := stream.sentMessages[0].GetRequestHeaders()
		require.NotNil(firstResp, "First response must be a RequestHeaders frame")

		setHeaders := firstResp.GetResponse().GetHeaderMutation().GetSetHeaders()
		require.Len(setHeaders, 2)
		assert.Equal(t, metadata.DestinationEndpointKey, setHeaders[0].GetHeader().GetKey())
		assert.Equal(t, "10.0.0.1:8080", string(setHeaders[0].GetHeader().GetRawValue()))
		assert.Equal(t, "X-Echo-Set-Header", setHeaders[1].GetHeader().GetKey())
		assert.Equal(t, metadata.ConformanceTestResultHeader+":10.0.0.1:8080", string(setHeaders[1].GetHeader().GetRawValue()))

		// Response 2: RequestBody Response
		secondResp := stream.sentMessages[1].GetRequestBody()
		require.NotNil(secondResp, "Second response must be a RequestBody frame")
		require.Nil(secondResp.GetResponse().GetHeaderMutation(), "Deferred body response must not contain redundant mutations")
		requestBody := secondResp.GetResponse().GetBodyMutation().GetStreamedResponse()
		require.NotNil(requestBody)
		assert.Equal(t, reqBody.GetRequestBody().GetBody(), requestBody.GetBody())
		assert.Equal(t, reqBody.GetRequestBody().GetEndOfStream(), requestBody.GetEndOfStream())

		// Response 3: ResponseBody Response
		thirdResp := stream.sentMessages[2].GetResponseBody()
		require.NotNil(thirdResp, "Third response must be a ResponseBody frame")
		responseBody := thirdResp.GetResponse().GetBodyMutation().GetStreamedResponse()
		require.NotNil(responseBody)
		assert.Equal(t, respBody.GetResponseBody().GetBody(), responseBody.GetBody())
		assert.Equal(t, respBody.GetResponseBody().GetEndOfStream(), responseBody.GetEndOfStream())
	}
}

// TestProcess_DeferredHeaderMutationOnChunkedStreamingBody covers a body that arrives in
// more than one chunk. The endpoint is only known once the whole body has been received, so
// the request headers response is deferred to end of stream. Envoy accepts responses in the
// order it asked for them, and a request body response sent while the headers response is
// still outstanding is treated as spurious: Envoy abandons the stream and the request fails.
//
// The body chunks must therefore be held until the deferred headers response has been sent.
func TestProcess_DeferredHeaderMutationOnChunkedStreamingBody(t *testing.T) {
	pods := []*datastore.Endpoint{
		{Address: "10.0.0.1", Port: "8080"},
	}
	ds := &mockDatastore{pods: pods}
	server := NewStreamingServer(ds)

	reqHeaders := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestHeaders{
			RequestHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{
					Headers: []*envoyCorev3.HeaderValue{
						{Key: "test-epp-endpoint-selection", Value: "10.0.0.1"},
					},
				},
				EndOfStream: false,
			},
		},
	}

	// The body arrives in two chunks. Only the second ends the stream, so at the time the
	// first is received the endpoint is still unknown and the headers response is deferred.
	firstChunk := []byte(`{"prompt": "hel`)
	lastChunk := []byte(`lo"}`)

	reqBodyFirst := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{
				Body:        firstChunk,
				EndOfStream: false,
			},
		},
	}
	reqBodyLast := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{
				Body:        lastChunk,
				EndOfStream: true,
			},
		},
	}

	stream := &mockProcessServer{
		ctx:          t.Context(),
		recvMessages: []*extProcPb.ProcessingRequest{reqHeaders, reqBodyFirst, reqBodyLast},
	}

	err := server.Process(stream)
	assert.NoError(t, err)

	require := assert.New(t)

	// Nothing may be sent before the deferred headers response. A body response emitted for
	// the first chunk would appear here and would be the failure this test guards against.
	if require.NotEmpty(stream.sentMessages, "expected the deferred headers response") {
		require.NotNil(stream.sentMessages[0].GetRequestHeaders(),
			"the first response must be the deferred RequestHeaders frame, not a body frame")
	}

	// One headers response and one body response carrying the reassembled body.
	if require.Len(stream.sentMessages, 2) {
		setHeaders := stream.sentMessages[0].GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders()
		require.Len(setHeaders, 2)
		assert.Equal(t, metadata.DestinationEndpointKey, setHeaders[0].GetHeader().GetKey())
		assert.Equal(t, "10.0.0.1:8080", string(setHeaders[0].GetHeader().GetRawValue()))

		bodyResp := stream.sentMessages[1].GetRequestBody()
		require.NotNil(bodyResp, "the second response must be a RequestBody frame")
		streamed := bodyResp.GetResponse().GetBodyMutation().GetStreamedResponse()
		require.NotNil(streamed)
		assert.Equal(t, append(append([]byte{}, firstChunk...), lastChunk...), streamed.GetBody(),
			"both chunks must be forwarded, in arrival order")
		assert.True(t, streamed.GetEndOfStream(), "the flushed body response ends the stream")
	}
}
