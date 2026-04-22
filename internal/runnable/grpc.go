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

package runnable

import (
	"context"
	"fmt"
	"net"

	"google.golang.org/grpc"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

// GRPCServer converts the given gRPC server into a runnable that listens on the given port.
// The server name is just being used for logging.
func GRPCServer(name string, srv *grpc.Server, port int) manager.Runnable {
	return manager.RunnableFunc(func(ctx context.Context) error {
		lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
		if err != nil {
			return fmt.Errorf("gRPC server failed to listen - %w", err)
		}
		return serveGRPC(ctx, name, srv, lis)
	})
}

// GRPCServerWithListener converts the given gRPC server into a runnable using
// a pre-created listener. This avoids TOCTOU port races in tests where
// GetFreePort() closes a listener and the subsequent bind can be interleaved
// by other processes grabbing the same port.
func GRPCServerWithListener(name string, srv *grpc.Server, lis net.Listener) manager.Runnable {
	return manager.RunnableFunc(func(ctx context.Context) error {
		return serveGRPC(ctx, name, srv, lis)
	})
}

func serveGRPC(ctx context.Context, name string, srv *grpc.Server, lis net.Listener) error {
	// Use "name" key as that is what manager.Server does as well.
	log := ctrl.Log.WithValues("name", name)
	log.Info("gRPC server starting")
	log.Info("gRPC server listening", "addr", lis.Addr().String())

	// Shutdown on context closed.
	// Terminate the server on context closed.
	// Make sure the goroutine does not leak.
	doneCh := make(chan struct{})
	defer close(doneCh)
	go func() {
		select {
		case <-ctx.Done():
			log.Info("gRPC server shutting down")
			srv.GracefulStop()
		case <-doneCh:
		}
	}()

	// Keep serving until terminated.
	if err := srv.Serve(lis); err != nil && err != grpc.ErrServerStopped {
		return fmt.Errorf("gRPC server failed - %w", err)
	}
	log.Info("gRPC server terminated")
	return nil
}
