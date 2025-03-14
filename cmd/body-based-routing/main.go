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

package main

import (
	"flag"
	"net"
	"net/http"
	"os"
	"strconv"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	uberzap "go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"google.golang.org/grpc"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"k8s.io/client-go/rest"
	"k8s.io/component-base/metrics/legacyregistry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	"sigs.k8s.io/gateway-api-inference-extension/internal/runnable"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/body-based-routing/server"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

var (
	grpcPort = flag.Int(
		"grpcPort",
		runserver.DefaultGrpcPort,
		"The gRPC port used for communicating with Envoy proxy")
	grpcHealthPort = flag.Int(
		"grpcHealthPort",
		9005,
		"The port used for gRPC liveness and readiness probes")
	metricsPort = flag.Int(
		"metricsPort", 9090, "The metrics port")
	logVerbosity = flag.Int("v", logging.DEFAULT, "number for the log level verbosity")

	setupLog = ctrl.Log.WithName("setup")
)

func main() {
	if err := run(); err != nil {
		os.Exit(1)
	}
}

func run() error {
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

	mgr, err := ctrl.NewManager(cfg, ctrl.Options{})
	if err != nil {
		setupLog.Error(err, "Failed to create manager", "config", cfg)
		return err
	}

	ctx := ctrl.SetupSignalHandler()

	// Setup runner.
	serverRunner := &runserver.ExtProcServerRunner{GrpcPort: *grpcPort}

	// Register health server.
	if err := registerHealthServer(mgr, ctrl.Log.WithName("health"), *grpcHealthPort); err != nil {
		return err
	}

	// Register ext-proc server.
	if err := mgr.Add(serverRunner.AsRunnable(ctrl.Log.WithName("ext-proc"))); err != nil {
		setupLog.Error(err, "Failed to register ext-proc gRPC server")
		return err
	}

	// Register metrics handler.
	if err := registerMetricsHandler(mgr, *metricsPort, cfg); err != nil {
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
func registerHealthServer(mgr manager.Manager, logger logr.Logger, port int) error {
	srv := grpc.NewServer()
	healthPb.RegisterHealthServer(srv, &healthServer{
		logger: logger,
	})
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

const metricsEndpoint = "/metrics"

// registerMetricsHandler adds the metrics HTTP handler as a Runnable to the given manager.
func registerMetricsHandler(mgr manager.Manager, port int, cfg *rest.Config) error {
	metrics.Register()

	// Init HTTP server.
	h, err := metricsHandlerWithAuthenticationAndAuthorization(cfg)
	if err != nil {
		return err
	}

	mux := http.NewServeMux()
	mux.Handle(metricsEndpoint, h)

	srv := &http.Server{
		Addr:    net.JoinHostPort("", strconv.Itoa(port)),
		Handler: mux,
	}

	if err := mgr.Add(&manager.Server{
		Name:   "metrics",
		Server: srv,
	}); err != nil {
		setupLog.Error(err, "Failed to register metrics HTTP handler")
		return err
	}
	return nil
}

func metricsHandlerWithAuthenticationAndAuthorization(cfg *rest.Config) (http.Handler, error) {
	h := promhttp.HandlerFor(
		legacyregistry.DefaultGatherer,
		promhttp.HandlerOpts{},
	)
	httpClient, err := rest.HTTPClientFor(cfg)
	if err != nil {
		setupLog.Error(err, "Failed to create http client for metrics auth")
		return nil, err
	}

	filter, err := filters.WithAuthenticationAndAuthorization(cfg, httpClient)
	if err != nil {
		setupLog.Error(err, "Failed to create metrics filter for auth")
		return nil, err
	}
	metricsLogger := ctrl.Log.WithName("metrics").WithValues("path", metricsEndpoint)
	metricsAuthHandler, err := filter(metricsLogger, h)
	if err != nil {
		setupLog.Error(err, "Failed to create metrics auth handler")
		return nil, err
	}
	return metricsAuthHandler, nil
}
