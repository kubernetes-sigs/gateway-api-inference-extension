// Package main is the entry point for the server.
package main

import (
	"fmt"
	"net"

	"google3/base/go/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/body-based-routing/service"
	eppb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
)

func main() {
	address := "0.0.0.0:8181"
	lis, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatalf("Failed to listen on insecure port: %v", err)
	}
	fmt.Printf("Starting server on address %s\n", address)

	grpcServer := grpc.NewServer()
	eppb.RegisterExternalProcessorServer(grpcServer, &service.GRPCCalloutService{})

	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve gRPC on insecure port: %v", err)
	}
}
