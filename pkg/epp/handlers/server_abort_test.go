/*
Copyright 2026 The Kubernetes Authors.

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
	"testing"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/metadata"
)

// mockProcessServer implements ExternalProcessor_ProcessServer for testing.
type mockProcessServer struct {
	sentResponses []*extProcPb.ProcessingResponse
}

func (m *mockProcessServer) Send(resp *extProcPb.ProcessingResponse) error {
	m.sentResponses = append(m.sentResponses, resp)
	return nil
}

// Unused methods to satisfy the interface.
func (m *mockProcessServer) Recv() (*extProcPb.ProcessingRequest, error) { return nil, nil }
func (m *mockProcessServer) SetHeader(metadata.MD) error                 { return nil }
func (m *mockProcessServer) SendHeader(metadata.MD) error                { return nil }
func (m *mockProcessServer) SetTrailer(metadata.MD)                      {}
func (m *mockProcessServer) Context() context.Context                    { return context.Background() }
func (m *mockProcessServer) SendMsg(any) error                           { return nil }
func (m *mockProcessServer) RecvMsg(any) error                           { return nil }

func TestUpdateStateAndSendIfNeeded_Evicted(t *testing.T) {
	t.Parallel()
	srv := &mockProcessServer{}
	logger := logr.Discard()

	reqCtx := &RequestContext{
		RequestState: RequestEvicted,
	}

	err := reqCtx.updateStateAndSendIfNeeded(srv, logger)
	require.NoError(t, err)

	require.Len(t, srv.sentResponses, 1, "Should send exactly one response")
	ir := srv.sentResponses[0].GetImmediateResponse()
	require.NotNil(t, ir, "Response should be an ImmediateResponse")
	assert.Equal(t, envoyTypePb.StatusCode_TooManyRequests, ir.Status.Code)
	assert.Equal(t, []byte("request evicted by flow control"), ir.Body)
}

func TestUpdateStateAndSendIfNeeded_NotEvicted(t *testing.T) {
	t.Parallel()
	srv := &mockProcessServer{}
	logger := logr.Discard()

	// Normal state — no responses queued, nothing should be sent.
	reqCtx := &RequestContext{
		RequestState: RequestReceived,
	}

	err := reqCtx.updateStateAndSendIfNeeded(srv, logger)
	require.NoError(t, err)
	assert.Empty(t, srv.sentResponses, "Should not send any response for normal state without queued responses")
}
