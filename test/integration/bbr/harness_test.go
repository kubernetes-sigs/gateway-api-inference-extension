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

package bbr

import (
	"context"
	"net"
	"strconv"
	"strings"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/server"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	"sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

var logger = logutil.NewTestLogger().V(logutil.VERBOSE)

// BBRHarness encapsulates the environment for a single isolated BBR test run.
type BBRHarness struct {
	t      *testing.T
	ctx    context.Context
	Client extProcPb.ExternalProcessor_ProcessClient

	// Internal handles for cleanup
	server   *runserver.ExtProcServerRunner
	grpcConn *grpc.ClientConn
}

// NewBBRHarness boots up an isolated BBR server on a random port.
// streaming: determines if the BBR server runs in streaming mode or unary/buffered mode.
func NewBBRHarness(t *testing.T, ctx context.Context, streaming bool) *BBRHarness {
	t.Helper()

	// 1. Allocate Free Port
	tcpAddr, err := integration.GetFreePort()
	require.NoError(t, err, "failed to acquire free port for BBR server")
	port := tcpAddr.Port
	serverAddr := net.JoinHostPort(tcpAddr.IP.String(), strconv.Itoa(tcpAddr.Port))

	// 2. Configure BBR Server
	// BBR is simpler than EPP; it doesn't need a K8s Manager.
	runner := runserver.NewDefaultExtProcServerRunner(port, false)
	runner.SecureServing = false
	runner.Streaming = streaming

	// 3. Start Server in Background
	serverCtx, serverCancel := context.WithCancel(ctx)

	// Channel to signal if the server dies immediately (e.g., port binding error)
	serverErrChan := make(chan error, 1)

	go func() {
		logger.Info("Starting BBR server", "address", serverAddr, "streaming", streaming)
		if err := runner.AsRunnable(logger.WithName("bbr-server")).Start(serverCtx); err != nil {
			if !strings.Contains(err.Error(), "context canceled") {
				logger.Error(err, "BBR server stopped unexpectedly")
				select {
				case serverErrChan <- err:
				default:
				}
			}
		}
	}()

	// 4. Wait for Server Readiness
	// We must poll the port until the server successfully binds and listens.
	require.Eventually(t, func() bool {
		// Check for premature crash.
		select {
		case err := <-serverErrChan:
			t.Fatalf("Server failed to start: %v", err)
		default:
		}

		// Check for TCP readiness.
		conn, err := net.DialTimeout("tcp", serverAddr, 100*time.Millisecond)
		if err != nil {
			return false
		}
		conn.Close()
		return true
	}, 5*time.Second, 50*time.Millisecond, "BBR Server failed to bind port %s", serverAddr)

	// 5. Connect Client
	// Blocking dial ensures the server is reachable before the test logic begins.
	conn, err := grpc.NewClient(serverAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err, "failed to create grpc connection to BBR server")

	extProcClient, err := extProcPb.NewExternalProcessorClient(conn).Process(ctx)
	require.NoError(t, err, "failed to initialize ext_proc stream client")

	h := &BBRHarness{
		t:        t,
		ctx:      ctx,
		Client:   extProcClient,
		server:   runner,
		grpcConn: conn,
	}

	// 6. Register Cleanup
	t.Cleanup(func() {
		logger.Info("Tearing down BBR server", "port", port)
		serverCancel()
		if err := h.grpcConn.Close(); err != nil {
			t.Logf("Warning: failed to close grpc connection: %v", err)
		}
	})

	return h
}
