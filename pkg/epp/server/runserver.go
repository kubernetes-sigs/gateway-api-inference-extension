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
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/gateway-api-inference-extension/internal/runnable"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
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
	PoolName                                 string
	PoolNamespace                            string
	RefreshMetricsInterval                   time.Duration
	RefreshPrometheusMetricsInterval         time.Duration
	Datastore                                datastore.Datastore
	Provider                                 *backend.Provider
	SecureServing                            bool
	CertPath                                 string
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
		PoolName:                                 DefaultPoolName,
		PoolNamespace:                            DefaultPoolNamespace,
		RefreshMetricsInterval:                   DefaultRefreshMetricsInterval,
		RefreshPrometheusMetricsInterval:         DefaultRefreshPrometheusMetricsInterval,
		SecureServing:                            DefaultSecureServing,
		// Datastore can be assigned later.
	}
}

// SetupWithManager sets up the runner with the given manager.
func (r *ExtProcServerRunner) SetupWithManager(ctx context.Context, mgr ctrl.Manager) error {
	// Create the controllers and register them with the manager
	if err := (&controller.InferencePoolReconciler{
		Datastore: r.Datastore,
		Scheme:    mgr.GetScheme(),
		Client:    mgr.GetClient(),
		PoolNamespacedName: types.NamespacedName{
			Name:      r.PoolName,
			Namespace: r.PoolNamespace,
		},
		Record: mgr.GetEventRecorderFor("InferencePool"),
	}).SetupWithManager(mgr); err != nil {
		return fmt.Errorf("failed setting up InferencePoolReconciler: %w", err)
	}

	if err := (&controller.InferenceModelReconciler{
		Datastore: r.Datastore,
		Scheme:    mgr.GetScheme(),
		Client:    mgr.GetClient(),
		PoolNamespacedName: types.NamespacedName{
			Name:      r.PoolName,
			Namespace: r.PoolNamespace,
		},
		Record: mgr.GetEventRecorderFor("InferenceModel"),
	}).SetupWithManager(ctx, mgr); err != nil {
		return fmt.Errorf("failed setting up InferenceModelReconciler: %w", err)
	}

	if err := (&controller.PodReconciler{
		Datastore: r.Datastore,
		Client:    mgr.GetClient(),
		Namespace: r.PoolNamespace,
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
		// Initialize backend provider
		if err := r.Provider.Init(ctx, r.RefreshMetricsInterval, r.RefreshPrometheusMetricsInterval); err != nil {
			logger.Error(err, "Failed to initialize backend provider")
			return err
		}

		var srv *grpc.Server
		if r.SecureServing {
			var cert tls.Certificate
			var err error
			if r.CertPath != "" {
				cert, err = tls.LoadX509KeyPair(r.CertPath+"/tls.crt", r.CertPath+"/tls.key")
			} else {
				// Create tls based credential.
				cert, err = createSelfSignedTLSCertificate(logger)
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
		extProcPb.RegisterExternalProcessorServer(
			srv,
			handlers.NewServer(scheduling.NewScheduler(r.Datastore), r.DestinationEndpointHintMetadataNamespace, r.DestinationEndpointHintKey, r.Datastore),
		)

		// Forward to the gRPC runnable.
		return runnable.GRPCServer("ext-proc", srv, r.GrpcPort).Start(ctx)
	}))
}

func createSelfSignedTLSCertificate(logger logr.Logger) (tls.Certificate, error) {
	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		logger.Error(err, "Failed to create serial number for self-signed cert")
		return tls.Certificate{}, err
	}
	now := time.Now()
	notBefore := now.UTC()
	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			Organization: []string{"Inference Ext"},
		},
		NotBefore:             notBefore,
		NotAfter:              now.Add(time.Hour * 24 * 365 * 10).UTC(), // 10 years
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}

	priv, err := rsa.GenerateKey(rand.Reader, 4096)
	if err != nil {
		logger.Error(err, "Failed to generate key for self-signed cert")
		return tls.Certificate{}, err
	}

	derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		logger.Error(err, "Failed to create self-signed certificate")
		return tls.Certificate{}, err
	}

	certBytes := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: derBytes})

	privBytes, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		logger.Error(err, "Failed to marshal private key for self-signed certificate")
		return tls.Certificate{}, err
	}
	keyBytes := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: privBytes})

	return tls.X509KeyPair(certBytes, keyBytes)
}
