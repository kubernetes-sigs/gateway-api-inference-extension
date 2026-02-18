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

package server

import (
	"context"
	"crypto/tls"
	"fmt"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/manager"

	"sigs.k8s.io/gateway-api-inference-extension/internal/runnable"
	tlsutil "sigs.k8s.io/gateway-api-inference-extension/internal/tls"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/controller"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/handlers"
)

// ExtProcServerRunner provides methods to manage an external process server.
type ExtProcServerRunner struct {
	GrpcPort        int
	Datastore       datastore.Datastore
	SecureServing   bool
	Streaming       bool
	RequestPlugins  []framework.PayloadProcessor
	ResponsePlugins []framework.PayloadProcessor
}

func NewDefaultExtProcServerRunner(port int, streaming bool) *ExtProcServerRunner {
	return &ExtProcServerRunner{
		GrpcPort:      port,
		SecureServing: true,
		Streaming:     streaming,
	}
	// Dependencies can be assigned later.
}

// SetupWithManager sets up the runner with the given manager.
func (r *ExtProcServerRunner) SetupWithManager(mgr ctrl.Manager) error {
	// Create the configmap controller and register it with the manager
	if err := (&controller.ConfigMapReconciler{
		Datastore: r.Datastore,
		Reader:    mgr.GetClient(),
	}).SetupWithManager(mgr); err != nil {
		return fmt.Errorf("failed setting up ConfigMap Reconciler - %w", err)
	}

	return nil
}

// AsRunnable returns a Runnable that can be used to start the ext-proc gRPC server.
// The runnable implements LeaderElectionRunnable with leader election disabled.
func (r *ExtProcServerRunner) AsRunnable(logger logr.Logger) manager.Runnable {
	return runnable.NoLeaderElection(manager.RunnableFunc(func(ctx context.Context) error {
		var srv *grpc.Server
		if r.SecureServing {
			cert, err := tlsutil.CreateSelfSignedTLSCertificate(logger)
			if err != nil {
				return fmt.Errorf("failed to create self signed certificate - %w", err)
			}
			creds := credentials.NewTLS(&tls.Config{
				Certificates: []tls.Certificate{cert},
				NextProtos:   []string{"h2"},
			})
			srv = grpc.NewServer(grpc.Creds(creds))
		} else {
			srv = grpc.NewServer()
		}

		extProcPb.RegisterExternalProcessorServer(srv, handlers.NewServer(r.Streaming, r.Datastore, r.RequestPlugins, r.ResponsePlugins))

		// Forward to the gRPC runnable.
		return runnable.GRPCServer("ext-proc", srv, r.GrpcPort).Start(ctx)
	}))
}
