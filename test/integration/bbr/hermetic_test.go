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

// Package bbr contains integration tests for the body-based routing extension.
package bbr

import (
	"context"
	"fmt"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/testing/protocmp"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/server"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/utils"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	"sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

// TestBodyBasedRouting validates the "Unary" (Non-Streaming) behavior of BBR.
// This simulates scenarios where Envoy buffers the body before sending it to ext_proc.
func TestBodyBasedRouting(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		req          *extProcPb.ProcessingRequest
		wantResponse *extProcPb.ProcessingResponse
		wantErr      bool
	}{
		{
			name:         "success: extracts model and sets header",
			req:          integration.ReqLLMUnary(logger, "test", "llama"),
			wantResponse: ExpectBBRUnaryResponse("llama"),
			wantErr:      false,
		},
		{
			name:         "noop: no model parameter in body",
			req:          integration.ReqLLMUnary(logger, "test1", ""),
			wantResponse: ExpectBBRUnaryResponse(""), // Expect no headers.
			wantErr:      false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx := context.Background()
			h := NewBBRHarness(t, ctx, false)

			res, err := integration.SendRequest(t, h.Client, tc.req)

			if tc.wantErr {
				require.Error(t, err, "expected error during request processing")
			} else {
				require.NoError(t, err, "unexpected error during request processing")
			}

			if diff := cmp.Diff(tc.wantResponse, res, protocmp.Transform()); diff != "" {
				t.Errorf("Response mismatch (-want +got): %v", diff)
			}
		})
	}
}

// TestFullDuplexStreamed_BodyBasedRouting validates the "Streaming" behavior of BBR.
// This validates that BBR correctly buffers streamed chunks, inspects the body, and injects the header.
func TestFullDuplexStreamed_BodyBasedRouting(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		reqs          []*extProcPb.ProcessingRequest
		wantResponses []*extProcPb.ProcessingResponse
		wantErr       bool
	}{
		{
			name: "success: adds model header from simple body",
			reqs: integration.ReqLLM(logger, "test", "foo", "bar"),
			wantResponses: []*extProcPb.ProcessingResponse{
				ExpectBBRHeader("foo"),
				ExpectBBRBodyPassThrough("test", "foo"),
			},
		},
		{
			name: "success: buffers split chunks and extracts model",
			reqs: integration.ReqRaw(
				map[string]string{"hi": "mom"},
				`{"max_tokens":100,"model":"sql-lo`,
				`ra-sheddable","prompt":"test","temperature":0}`,
			),
			wantResponses: []*extProcPb.ProcessingResponse{
				ExpectBBRHeader("sql-lora-sheddable"),
				ExpectBBRBodyPassThrough("test", "sql-lora-sheddable"),
			},
		},
		{
			name: "noop: handles missing model field gracefully",
			reqs: integration.ReqLLM(logger, "test", "", ""),
			wantResponses: []*extProcPb.ProcessingResponse{
				ExpectBBRNoOpHeader(),
				ExpectBBRBodyPassThrough("test", ""),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx := context.Background()
			h := NewBBRHarness(t, ctx, true)

			responses, err := integration.StreamedRequest(t, h.Client, tc.reqs, len(tc.wantResponses))

			if tc.wantErr {
				require.Error(t, err, "expected stream error")
			} else {
				require.NoError(t, err, "unexpected stream error")
			}

			if diff := cmp.Diff(tc.wantResponses, responses, protocmp.Transform()); diff != "" {
				t.Errorf("Response mismatch (-want +got): %v", diff)
			}
		})
	}
}

func setUpHermeticServer(streaming bool) (client extProcPb.ExternalProcessor_ProcessClient, cleanup func()) {
	port := 9004

	serverCtx, stopServer := context.WithCancel(context.Background())

	//Initialize PluginRegistry and request/response PluginsChain instances based on the minimal configuration setting vi env vars
	registry, requestChain, responseChain, err := utils.InitPlugins()
	if err != nil {
		logutil.Fatal(logger, err, "failed to initialize BBR pluggable framework: %v", err)
	}

	serverRunner := runserver.NewDefaultExtProcServerRunner(port, false, registry, requestChain, responseChain)
	serverRunner.SecureServing = false
	serverRunner.Streaming = streaming

	go func() {
		if err := serverRunner.AsRunnable(logger.WithName("ext-proc")).Start(serverCtx); err != nil {
			logutil.Fatal(logger, err, "Failed to start ext-proc server")
		}
	}()

	address := fmt.Sprintf("localhost:%v", port)
	// Create a grpc connection
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		logutil.Fatal(logger, err, "Failed to connect", "address", address)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	client, err = extProcPb.NewExternalProcessorClient(conn).Process(ctx)
	if err != nil {
		logutil.Fatal(logger, err, "Failed to create client")
	}
	return client, func() {
		cancel()
		conn.Close()
		stopServer()

		// wait a little until the goroutines actually exit
		time.Sleep(5 * time.Second)
	}
}
