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

// Package main is the entry point for the server.
package main

import (
	"fmt"
	"net"
	"log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/body-based-routing/service"
	"github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
)

func main() {
	address := ":8181"
	lis, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatalf("Failed to listen on insecure port: %v", err)
	}
	fmt.Printf("Starting server on address %s\n", address)

	grpcServer := grpc.NewServer()
	v3.RegisterExternalProcessorServer(grpcServer, &service.GRPCCalloutService{})

	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve gRPC on insecure port: %v", err)
	}
}
