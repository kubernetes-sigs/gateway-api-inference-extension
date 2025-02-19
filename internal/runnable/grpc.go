package runnable

import (
	"context"
	"fmt"
	"net"

	"google.golang.org/grpc"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

// GRPCServer converts the given gRPC server into a runnable.
// The server name is just being used for logging.
func GRPCServer(name string, srv *grpc.Server, port int) manager.Runnable {
	return manager.RunnableFunc(func(ctx context.Context) error {
		// Use "name" key as that is what manager.Server does as well.
		log := ctrl.Log.WithValues("name", name)
		log.Info("gRPC server starting")

		// Start listening.
		lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
		if err != nil {
			log.Error(err, "gRPC server failed to listen")
			return err
		}

		log.Info("gRPC server listening", "port", port)

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
			log.Error(err, "gRPC server failed")
			return err
		}
		log.Info("gRPC server terminated")
		return nil
	})
}
