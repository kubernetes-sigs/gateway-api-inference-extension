package main

import (
	"flag"
	"fmt"
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
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/component-base/metrics/legacyregistry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/internal/runnable"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend/vllm"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/metrics"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/server"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

const (
	defaultMetricsEndpoint = "/metrics"
)

var (
	grpcPort = flag.Int(
		"grpcPort",
		runserver.DefaultGrpcPort,
		"The gRPC port used for communicating with Envoy proxy")
	grpcHealthPort = flag.Int(
		"grpcHealthPort",
		9003,
		"The port used for gRPC liveness and readiness probes")
	metricsPort = flag.Int(
		"metricsPort", 9090, "The metrics port")
	targetEndpointKey = flag.String(
		"targetEndpointKey",
		runserver.DefaultTargetEndpointKey,
		"Header key used by Envoy to route to the appropriate pod. This must match Envoy configuration.")
	poolName = flag.String(
		"poolName",
		runserver.DefaultPoolName,
		"Name of the InferencePool this Endpoint Picker is associated with.")
	poolNamespace = flag.String(
		"poolNamespace",
		runserver.DefaultPoolNamespace,
		"Namespace of the InferencePool this Endpoint Picker is associated with.")
	refreshMetricsInterval = flag.Duration(
		"refreshMetricsInterval",
		runserver.DefaultRefreshMetricsInterval,
		"interval to refresh metrics")
	refreshPrometheusMetricsInterval = flag.Duration(
		"refreshPrometheusMetricsInterval",
		runserver.DefaultRefreshPrometheusMetricsInterval,
		"interval to flush prometheus metrics")
	logVerbosity  = flag.Int("v", logging.DEFAULT, "number for the log level verbosity")
	secureServing = flag.Bool(
		"secureServing", runserver.DefaultSecureServing, "Enables secure serving. Defaults to true.")
	certPath = flag.String(
		"certPath", "", "The path to the certificate for secure serving. The certificate and private key files "+
			"are assumed to be named tls.crt and tls.key, respectively. If not set, and secureServing is enabled, "+
			"then a self-signed certificate is used.")

	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(v1alpha1.AddToScheme(scheme))
}

func main() {
	if err := run(); err != nil {
		os.Exit(1)
	}
}

func run() error {
	opts := zap.Options{
		Development: true,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()
	initLogging(&opts)

	// Validate flags
	if err := validateFlags(); err != nil {
		setupLog.Error(err, "Failed to validate flags")
		return err
	}

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

	mgr, err := ctrl.NewManager(cfg, ctrl.Options{Scheme: scheme})
	if err != nil {
		setupLog.Error(err, "Failed to create controller manager", "config", cfg)
		return err
	}

	// Setup runner.
	datastore := datastore.NewDatastore()
	provider := backend.NewProvider(&vllm.PodMetricsClientImpl{}, datastore)
	serverRunner := &runserver.ExtProcServerRunner{
		GrpcPort:                         *grpcPort,
		TargetEndpointKey:                *targetEndpointKey,
		PoolName:                         *poolName,
		PoolNamespace:                    *poolNamespace,
		RefreshMetricsInterval:           *refreshMetricsInterval,
		RefreshPrometheusMetricsInterval: *refreshPrometheusMetricsInterval,
		Datastore:                        datastore,
		SecureServing:                    *secureServing,
		CertPath:                         *certPath,
		Provider:                         provider,
	}
	if err := serverRunner.SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "Failed to setup ext-proc server")
		return err
	}

	// Register health server.
	if err := registerHealthServer(mgr, ctrl.Log.WithName("health"), datastore, *grpcHealthPort); err != nil {
		return err
	}

	// Register ext-proc server.
	if err := mgr.Add(serverRunner.AsRunnable(ctrl.Log.WithName("ext-proc"))); err != nil {
		setupLog.Error(err, "Failed to register ext-proc server")
		return err
	}

	// Register metrics handler.
	if err := registerMetricsHandler(mgr, *metricsPort, cfg); err != nil {
		return err
	}

	// Start the manager. This blocks until a signal is received.
	setupLog.Info("Controller manager starting")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "Error starting controller manager")
		return err
	}
	setupLog.Info("Controller manager terminated")
	return nil
}

func initLogging(opts *zap.Options) {
	// Unless -zap-log-level is explicitly set, use -v
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

// registerHealthServer adds the Health gRPC server as a Runnable to the given manager.
func registerHealthServer(mgr manager.Manager, logger logr.Logger, ds datastore.Datastore, port int) error {
	srv := grpc.NewServer()
	healthPb.RegisterHealthServer(srv, &healthServer{
		logger:    logger,
		datastore: ds,
	})
	if err := mgr.Add(
		runnable.NoLeaderElection(runnable.GRPCServer("health", srv, port))); err != nil {
		setupLog.Error(err, "Failed to register health server")
		return err
	}
	return nil
}

// registerMetricsHandler adds the metrics HTTP handler as a Runnable to the given manager.
func registerMetricsHandler(mgr manager.Manager, port int, cfg *rest.Config) error {
	metrics.Register()

	// Init HTTP server.
	h, err := metricsHandlerWithAuthenticationAndAuthorization(cfg)
	if err != nil {
		return err
	}

	mux := http.NewServeMux()
	mux.Handle(defaultMetricsEndpoint, h)

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
	metricsLogger := ctrl.Log.WithName("metrics").WithValues("path", defaultMetricsEndpoint)
	metricsAuthHandler, err := filter(metricsLogger, h)
	if err != nil {
		setupLog.Error(err, "Failed to create metrics auth handler")
		return nil, err
	}
	return metricsAuthHandler, nil
}

func validateFlags() error {
	if *poolName == "" {
		return fmt.Errorf("required %q flag not set", "poolName")
	}

	return nil
}
