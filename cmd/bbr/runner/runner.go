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

package runner

import (
	"context"
	"flag"
	"fmt"

	uberzap "go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"google.golang.org/grpc"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	"sigs.k8s.io/gateway-api-inference-extension/internal/runnable"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/metrics"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/server"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

var (
	grpcPort       = flag.Int("grpc-port", 9004, "The gRPC port used for communicating with Envoy proxy")
	grpcHealthPort = flag.Int("grpc-health-port", 9005, "The port used for gRPC liveness and readiness probes")
	metricsPort    = flag.Int("metrics-port", 9090, "The metrics port")
	streaming      = flag.Bool("streaming", false, "Enables streaming support for Envoy full-duplex streaming mode")
	logVerbosity   = flag.Int("v", logging.DEFAULT, "number for the log level verbosity")

	setupLog = ctrl.Log.WithName("setup")
)

func NewRunner() *Runner {
	return &Runner{
		bbrExecutableName: "BBR",
	}
}

// Runner is used to run bbr with its plugins
type Runner struct {
	bbrExecutableName string
}

// WithExecutableName sets the name of the executable containing the runner.
// The name is used in the version log upon startup and is otherwise opaque.
func (r *Runner) WithExecutableName(exeName string) *Runner {
	r.bbrExecutableName = exeName
	return r
}

func (r *Runner) Run(ctx context.Context) error {
	opts := zap.Options{Development: true}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()
	initLogging(&opts)

	// Print all flag values
	flags := make(map[string]any)
	flag.VisitAll(func(f *flag.Flag) {
		flags[f.Name] = f.Value
	})
	setupLog.Info("Flags processed", "flags", flags)

	// Init runtime.
	cfg, err := ctrl.GetConfig()
	if err != nil {
		setupLog.Error(err, "Failed to get rest config")
		return err
	}

	metrics.Register()

	// Register metrics handler.
	// Metrics endpoint is enabled in 'config/default/kustomization.yaml'. The Metrics options configure the server.
	// More info:
	// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.19.1/pkg/metrics/server
	// - https://book.kubebuilder.io/reference/metrics.html
	metricsServerOptions := metricsserver.Options{
		BindAddress:    fmt.Sprintf(":%d", *metricsPort),
		FilterProvider: filters.WithAuthenticationAndAuthorization,
	}
	mgr, err := ctrl.NewManager(cfg, ctrl.Options{Metrics: metricsServerOptions})
	if err != nil {
		setupLog.Error(err, "Failed to create manager", "config", cfg)
		return err
	}

	// Setup runner.
	serverRunner := runserver.NewDefaultExtProcServerRunner(*grpcPort, *streaming)

	// Register health server.
	if err := registerHealthServer(mgr, *grpcHealthPort); err != nil {
		return err
	}

	// Register ext-proc server.
	if err := mgr.Add(serverRunner.AsRunnable(ctrl.Log.WithName("ext-proc"))); err != nil {
		setupLog.Error(err, "Failed to register ext-proc gRPC server")
		return err
	}

	// Start the manager. This blocks until a signal is received.
	setupLog.Info("Manager starting")
	if err := mgr.Start(ctx); err != nil {
		setupLog.Error(err, "Error starting manager")
		return err
	}
	setupLog.Info("Manager terminated")
	return nil
}

// registerHealthServer adds the Health gRPC server as a Runnable to the given manager.
func registerHealthServer(mgr manager.Manager, port int) error {
	srv := grpc.NewServer()
	healthPb.RegisterHealthServer(srv, &healthServer{})
	if err := mgr.Add(
		runnable.NoLeaderElection(runnable.GRPCServer("health", srv, port))); err != nil {
		setupLog.Error(err, "Failed to register health server")
		return err
	}
	return nil
}

func initLogging(opts *zap.Options) {
	useV := true
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "zap-log-level" {
			useV = false
		}
	})
	if useV {
		// See https://pkg.go.dev/sigs.k8s.io/controller-runtime/pkg/log/zap#Options.Level
		lvl := -1 * (*logVerbosity)
		opts.Level = uberzap.NewAtomicLevelAt(zapcore.Level(int8(lvl)))
	}

	logger := zap.New(zap.UseFlagOptions(opts), zap.RawZapOpts(uberzap.AddCaller()))
	ctrl.SetLogger(logger)
}
