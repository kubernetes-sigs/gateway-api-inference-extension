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
	"io"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/metadata"
)

// mockProcessServer implements ExternalProcessor_ProcessServer for testing recvOrAbort.
type mockProcessServer struct {
	recvCh chan *extProcPb.ProcessingRequest
	sentCh chan *extProcPb.ProcessingResponse
	ctx    context.Context
}

func newMockProcessServer(ctx context.Context) *mockProcessServer {
	return &mockProcessServer{
		recvCh: make(chan *extProcPb.ProcessingRequest, 1),
		sentCh: make(chan *extProcPb.ProcessingResponse, 1),
		ctx:    ctx,
	}
}

func (m *mockProcessServer) Send(resp *extProcPb.ProcessingResponse) error {
	m.sentCh <- resp
	return nil
}

func (m *mockProcessServer) Recv() (*extProcPb.ProcessingRequest, error) {
	select {
	case req := <-m.recvCh:
		if req == nil {
			return nil, io.EOF
		}
		return req, nil
	case <-m.ctx.Done():
		return nil, m.ctx.Err()
	}
}

func (m *mockProcessServer) SetHeader(metadata.MD) error  { return nil }
func (m *mockProcessServer) SendHeader(metadata.MD) error { return nil }
func (m *mockProcessServer) SetTrailer(metadata.MD)       {}
func (m *mockProcessServer) Context() context.Context     { return m.ctx }
func (m *mockProcessServer) SendMsg(any) error            { return nil }
func (m *mockProcessServer) RecvMsg(any) error            { return nil }

func TestRecvOrAbort_NormalRecv(t *testing.T) {
	t.Parallel()
	s := &StreamingServer{}
	ctx := context.Background()
	srv := newMockProcessServer(ctx)
	abortCh := make(chan struct{})

	// Send a request before calling recvOrAbort.
	expectedReq := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_ResponseBody{
			ResponseBody: &extProcPb.HttpBody{
				Body: []byte("token"),
			},
		},
	}
	srv.recvCh <- expectedReq

	req, recvErr, abortErr := s.recvOrAbort(srv, abortCh)

	assert.NoError(t, abortErr, "No abort should have occurred")
	assert.NoError(t, recvErr, "Recv should succeed")
	assert.Equal(t, expectedReq, req, "Should receive the expected request")
}

func TestRecvOrAbort_AbortBeforeRecv(t *testing.T) {
	t.Parallel()
	s := &StreamingServer{}
	ctx := context.Background()
	srv := newMockProcessServer(ctx)
	abortCh := make(chan struct{})

	// Close the abort channel before sending any request.
	close(abortCh)

	req, _, abortErr := s.recvOrAbort(srv, abortCh)

	assert.ErrorIs(t, abortErr, errEvicted, "Should return eviction error")
	assert.Nil(t, req, "Request should be nil on abort")

	// Verify that ImmediateResponse was sent.
	select {
	case sent := <-srv.sentCh:
		ir := sent.GetImmediateResponse()
		require.NotNil(t, ir, "Should have sent ImmediateResponse")
		assert.Equal(t, envoyTypePb.StatusCode_ServiceUnavailable, ir.Status.Code)
		assert.Equal(t, []byte("request evicted by flow control"), ir.Body)
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for ImmediateResponse")
	}
}

func TestRecvOrAbort_AbortDuringRecvWait(t *testing.T) {
	t.Parallel()
	s := &StreamingServer{}
	ctx := context.Background()
	srv := newMockProcessServer(ctx)
	abortCh := make(chan struct{})

	// Don't send any request — Recv() will block.
	// Close abort channel after a short delay.
	go func() {
		time.Sleep(50 * time.Millisecond)
		close(abortCh)
	}()

	req, _, abortErr := s.recvOrAbort(srv, abortCh)

	assert.ErrorIs(t, abortErr, errEvicted, "Should return eviction error")
	assert.Nil(t, req, "Request should be nil on abort")

	// Verify ImmediateResponse was sent.
	select {
	case sent := <-srv.sentCh:
		ir := sent.GetImmediateResponse()
		require.NotNil(t, ir)
		assert.Equal(t, envoyTypePb.StatusCode_ServiceUnavailable, ir.Status.Code)
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for ImmediateResponse")
	}
}

func TestRecvOrAbort_RecvEOF(t *testing.T) {
	t.Parallel()
	s := &StreamingServer{}
	ctx := context.Background()
	srv := newMockProcessServer(ctx)
	abortCh := make(chan struct{})

	// Send nil to simulate EOF.
	srv.recvCh <- nil

	req, recvErr, abortErr := s.recvOrAbort(srv, abortCh)

	assert.NoError(t, abortErr, "No abort should have occurred")
	assert.ErrorIs(t, recvErr, io.EOF)
	assert.Nil(t, req)
}
