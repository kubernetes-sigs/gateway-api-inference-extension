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
	"testing"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

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
	port, err := integration.GetFreePort()
	require.NoError(t, err, "failed to acquire free port for BBR server")

	// 2. Configure BBR Server
	// BBR is simpler than EPP; it doesn't need a K8s Manager.
	runner := runserver.NewDefaultExtProcServerRunner(port, false)
	runner.SecureServing = false
	runner.Streaming = streaming

	// 3. Start Server in Background
	serverCtx, serverCancel := context.WithCancel(ctx)

	runnable := runner.AsRunnable(logger.WithName("bbr-server")).Start
	client, conn := integration.StartExtProcServer(
		t,
		serverCtx,
		runnable,
		port,
		logger,
	)

	h := &BBRHarness{
		t:        t,
		ctx:      ctx,
		Client:   client,
		server:   runner,
		grpcConn: conn,
	}

	// 4. Register Cleanup
	t.Cleanup(func() {
		logger.Info("Tearing down BBR server", "port", port)
		serverCancel()
		if err := h.grpcConn.Close(); err != nil {
			t.Logf("Warning: failed to close grpc connection: %v", err)
		}
	})

	return h
}
