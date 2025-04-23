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
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/gateway-api-inference-extension/internal/runnable"
	tlsutil "sigs.k8s.io/gateway-api-inference-extension/internal/tls"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/controller"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
)

// ExtProcServerRunner provides methods to manage an external process server.
type ExtProcServerRunner struct {
	GrpcPort                                 int
	DestinationEndpointHintMetadataNamespace string
	DestinationEndpointHintKey               string
	PoolNamespacedName                       types.NamespacedName
	Datastore                                datastore.Datastore
	SecureServing                            bool
	CertPath                                 string
	UseStreaming                             bool
	RefreshPrometheusMetricsInterval         time.Duration

	// This should only be used in tests. We won't need this once we don't inject metrics in the tests.
	// TODO:(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/432) Cleanup
	TestPodMetricsClient *backendmetrics.FakePodMetricsClient
}

// Default values for CLI flags in main
const (
	DefaultGrpcPort                                 = 9002                             // default for --grpcPort
	DefaultDestinationEndpointHintMetadataNamespace = "envoy.lb"                       // default for --destinationEndpointHintMetadataNamespace
	DefaultDestinationEndpointHintKey               = "x-gateway-destination-endpoint" // default for --destinationEndpointHintKey
	DefaultPoolName                                 = ""                               // required but no default
	DefaultPoolNamespace                            = "default"                        // default for --poolNamespace
	DefaultRefreshMetricsInterval                   = 50 * time.Millisecond            // default for --refreshMetricsInterval
	DefaultRefreshPrometheusMetricsInterval         = 5 * time.Second                  // default for --refreshPrometheusMetricsInterval
	DefaultSecureServing                            = true                             // default for --secureServing
)

func NewDefaultExtProcServerRunner() *ExtProcServerRunner {
	return &ExtProcServerRunner{
		GrpcPort:                                 DefaultGrpcPort,
		DestinationEndpointHintKey:               DefaultDestinationEndpointHintKey,
		DestinationEndpointHintMetadataNamespace: DefaultDestinationEndpointHintMetadataNamespace,
		PoolNamespacedName:                       types.NamespacedName{Name: DefaultPoolName, Namespace: DefaultPoolNamespace},
		SecureServing:                            DefaultSecureServing,
		RefreshPrometheusMetricsInterval:         DefaultRefreshPrometheusMetricsInterval,
		// Datastore can be assigned later.
	}
}

// SetupWithManager sets up the runner with the given manager.
func (r *ExtProcServerRunner) SetupWithManager(ctx context.Context, mgr ctrl.Manager) error {
	// Create the controllers and register them with the manager
	if err := (&controller.InferencePoolReconciler{
		Datastore: r.Datastore,
		Client:    mgr.GetClient(),
		Record:    mgr.GetEventRecorderFor("InferencePool"),
	}).SetupWithManager(mgr); err != nil {
		return fmt.Errorf("failed setting up InferencePoolReconciler: %w", err)
	}

	if err := (&controller.InferenceModelReconciler{
		Datastore:          r.Datastore,
		Client:             mgr.GetClient(),
		PoolNamespacedName: r.PoolNamespacedName,
		Record:             mgr.GetEventRecorderFor("InferenceModel"),
	}).SetupWithManager(ctx, mgr); err != nil {
		return fmt.Errorf("failed setting up InferenceModelReconciler: %w", err)
	}

	if err := (&controller.PodReconciler{
		Datastore: r.Datastore,
		Client:    mgr.GetClient(),
		Record:    mgr.GetEventRecorderFor("pod"),
	}).SetupWithManager(mgr); err != nil {
		return fmt.Errorf("failed setting up EndpointSliceReconciler: %v", err)
	}
	return nil
}

// AsRunnable returns a Runnable that can be used to start the ext-proc gRPC server.
// The runnable implements LeaderElectionRunnable with leader election disabled.
func (r *ExtProcServerRunner) AsRunnable(logger logr.Logger) manager.Runnable {
	return runnable.NoLeaderElection(manager.RunnableFunc(func(ctx context.Context) error {
		backendmetrics.StartMetricsLogger(ctx, r.Datastore, r.RefreshPrometheusMetricsInterval)
		var srv *grpc.Server
		if r.SecureServing {
			var cert tls.Certificate
			var err error
			if r.CertPath != "" {
				cert, err = tls.LoadX509KeyPair(r.CertPath+"/tls.crt", r.CertPath+"/tls.key")
			} else {
				// Create tls based credential.
				cert, err = tlsutil.CreateSelfSignedTLSCertificate(logger)
			}
			if err != nil {
				logger.Error(err, "Failed to create self signed certificate")
				return err
			}

			creds := credentials.NewTLS(&tls.Config{
				Certificates: []tls.Certificate{cert},
			})
			// Init the server.
			srv = grpc.NewServer(grpc.Creds(creds))
		} else {
			srv = grpc.NewServer()
		}
		extProcServer := handlers.NewStreamingServer(scheduling.NewScheduler(r.Datastore), r.DestinationEndpointHintMetadataNamespace, r.DestinationEndpointHintKey, r.Datastore)
		extProcPb.RegisterExternalProcessorServer(
			srv,
			extProcServer,
		)

		// Forward to the gRPC runnable.
		return runnable.GRPCServer("ext-proc", srv, r.GrpcPort).Start(ctx)
	}))
}
