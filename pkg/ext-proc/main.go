package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend/vllm"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/handlers"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/metrics"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/scheduling"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/component-base/metrics/legacyregistry"
	klog "k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
)

const (
	defaultMetricsEndpoint = "/metrics"
)

var (
	grpcPort = flag.Int(
		"grpcPort",
		9002,
		"The gRPC port used for communicating with Envoy proxy")
	grpcHealthPort = flag.Int(
		"grpcHealthPort",
		9003,
		"The port used for gRPC liveness and readiness probes")
	metricsPort = flag.Int(
		"metricsPort", 9090, "The metrics port")
	targetPodHeader = flag.String(
		"targetPodHeader",
		"target-pod",
		"Header key used by Envoy to route to the appropriate pod. This must match Envoy configuration.")
	poolName = flag.String(
		"poolName",
		"",
		"Name of the InferencePool this Endpoint Picker is associated with.")
	poolNamespace = flag.String(
		"poolNamespace",
		"default",
		"Namespace of the InferencePool this Endpoint Picker is associated with.")
	serviceName = flag.String(
		"serviceName",
		"",
		"Name of the Service that will be used to read EndpointSlices from")
	zone = flag.String(
		"zone",
		"",
		"The zone that this instance is created in. Will be passed to the corresponding endpointSlice. ")
	refreshPodsInterval = flag.Duration(
		"refreshPodsInterval",
		10*time.Second,
		"interval to refresh pods")
	refreshMetricsInterval = flag.Duration(
		"refreshMetricsInterval",
		50*time.Millisecond,
		"interval to refresh metrics")

	scheme = runtime.NewScheme()
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(v1alpha1.AddToScheme(scheme))
}

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	ctrl.SetLogger(klog.TODO())
	cfg, err := ctrl.GetConfig()
	if err != nil {
		klog.Fatalf("Failed to get rest config: %v", err)
	}
	// Validate flags
	if err := validateFlags(); err != nil {
		klog.Fatalf("Failed to validate flags: %v", err)
	}

	// Print all flag values
	flags := "Flags: "
	flag.VisitAll(func(f *flag.Flag) {
		flags += fmt.Sprintf("%s=%v; ", f.Name, f.Value)
	})
	klog.Info(flags)

	datastore := backend.NewK8sDataStore()

	runner := &runserver.ExtProcServerRunner{
		GrpcPort:               *grpcPort,
		TargetPodHeader:        *targetPodHeader,
		PoolName:               *poolName,
		PoolNamespace:          *poolNamespace,
		ServiceName:            *serviceName,
		Zone:                   *zone,
		RefreshPodsInterval:    *refreshPodsInterval,
		RefreshMetricsInterval: *refreshMetricsInterval,
		Scheme:                 scheme,
		Config:                 ctrl.GetConfigOrDie(),
		Datastore:              datastore,
	}
	runner.Setup()

	// Start health and ext-proc servers in goroutines
	healthSvr := startHealthServer(datastore, *grpcHealthPort)
	extProcSvr := runner.Start(
		datastore,
		*grpcPort,
		*refreshPodsInterval,
		*refreshMetricsInterval,
		*targetPodHeader,
	)
	// Start metrics handler
	metricsSvr := startMetricsHandler(*metricsPort, cfg)

	// Start the controller manager. Blocking and will return when shutdown is complete.
	klog.Infof("Starting controller manager")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		klog.Fatalf("Error starting controller manager: %v", err)
	}
	klog.Info("Controller manager shutting down")

	// Gracefully shutdown servers
	if healthSvr != nil {
		klog.Info("Health server shutting down")
		healthSvr.GracefulStop()
	}
	if extProcSvr != nil {
		klog.Info("Ext-proc server shutting down")
		extProcSvr.GracefulStop()
	}
	if metricsSvr != nil {
		klog.Info("Metrics server shutting down")
		if err := metricsSvr.Shutdown(context.Background()); err != nil {
			klog.Infof("Metrics server Shutdown: %v", err)
		}
	}

	klog.Info("All components shutdown")
}

// startHealthServer starts the gRPC health probe server in a goroutine.
func startHealthServer(ds *backend.K8sDatastore, port int) *grpc.Server {
	svr := grpc.NewServer()
	healthPb.RegisterHealthServer(svr, &healthServer{datastore: ds})

	go func() {
		lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
		if err != nil {
			klog.Fatalf("Health server failed to listen: %v", err)
		}
		klog.Infof("Health server listening on port: %d", port)

		// Blocking and will return when shutdown is complete.
		if err := svr.Serve(lis); err != nil && err != grpc.ErrServerStopped {
			klog.Fatalf("Health server failed: %v", err)
		}
		klog.Info("Health server shutting down")
	}()
	return svr
}

// startExternalProcessorServer starts the Envoy external processor server in a goroutine.
func startExternalProcessorServer(
	datastore *backend.K8sDatastore,
	port int,
	refreshPodsInterval, refreshMetricsInterval time.Duration,
	targetPodHeader string,
) *grpc.Server {
	svr := grpc.NewServer()

	go func() {
		lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
		if err != nil {
			klog.Fatalf("Ext-proc server failed to listen: %v", err)
		}
		klog.Infof("Ext-proc server listening on port: %d", port)

		// Initialize backend provider
		pp := backend.NewProvider(&vllm.PodMetricsClientImpl{}, datastore)
		if err := pp.Init(refreshPodsInterval, refreshMetricsInterval); err != nil {
			klog.Fatalf("Failed to initialize backend provider: %v", err)
		}

		// Register ext_proc handlers
		extProcPb.RegisterExternalProcessorServer(
			svr,
			handlers.NewServer(pp, scheduling.NewScheduler(pp), targetPodHeader, datastore),
		)

		// Blocking and will return when shutdown is complete.
		if err := svr.Serve(lis); err != nil && err != grpc.ErrServerStopped {
			klog.Fatalf("Ext-proc server failed: %v", err)
		}
		klog.Info("Ext-proc server shutting down")
	}()
	return svr
}

func startMetricsHandler(port int, cfg *rest.Config) *http.Server {
	metrics.Register()

	var svr *http.Server
	go func() {
		klog.Info("Starting metrics HTTP handler ...")

		mux := http.NewServeMux()
		mux.Handle(defaultMetricsEndpoint, metricsHandlerWithAuthenticationAndAuthorization(cfg))

		svr = &http.Server{
			Addr:    net.JoinHostPort("", strconv.Itoa(port)),
			Handler: mux,
		}
		if err := svr.ListenAndServe(); err != http.ErrServerClosed {
			klog.Fatalf("failed to start metrics HTTP handler: %v", err)
		}
	}()
	return svr
}

func metricsHandlerWithAuthenticationAndAuthorization(cfg *rest.Config) http.Handler {
	h := promhttp.HandlerFor(
		legacyregistry.DefaultGatherer,
		promhttp.HandlerOpts{},
	)
	httpClient, err := rest.HTTPClientFor(cfg)
	if err != nil {
		klog.Fatalf("failed to create http client for metrics auth: %v", err)
	}

	filter, err := filters.WithAuthenticationAndAuthorization(cfg, httpClient)
	if err != nil {
		klog.Fatalf("failed to create metrics filter for auth: %v", err)
	}
	metricsLogger := klog.LoggerWithValues(klog.NewKlogr(), "path", defaultMetricsEndpoint)
	metricsAuthHandler, err := filter(metricsLogger, h)
	if err != nil {
		klog.Fatalf("failed to create metrics auth handler: %v", err)
	}
	return metricsAuthHandler
}

func startMetricsHandler(port int, cfg *rest.Config) *http.Server {
	metrics.Register()

	var svr *http.Server
	go func() {
		klog.Info("Starting metrics HTTP handler ...")

		mux := http.NewServeMux()
		mux.Handle(defaultMetricsEndpoint, metricsHandlerWithAuthenticationAndAuthorization(cfg))

		svr = &http.Server{
			Addr:    net.JoinHostPort("", strconv.Itoa(port)),
			Handler: mux,
		}
		if err := svr.ListenAndServe(); err != http.ErrServerClosed {
			klog.Fatalf("failed to start metrics HTTP handler: %v", err)
		}
	}()
	return svr
}

func metricsHandlerWithAuthenticationAndAuthorization(cfg *rest.Config) http.Handler {
	h := promhttp.HandlerFor(
		legacyregistry.DefaultGatherer,
		promhttp.HandlerOpts{},
	)
	httpClient, err := rest.HTTPClientFor(cfg)
	if err != nil {
		klog.Fatalf("failed to create http client for metrics auth: %v", err)
	}

	filter, err := filters.WithAuthenticationAndAuthorization(cfg, httpClient)
	if err != nil {
		klog.Fatalf("failed to create metrics filter for auth: %v", err)
	}
	metricsLogger := klog.LoggerWithValues(klog.NewKlogr(), "path", defaultMetricsEndpoint)
	metricsAuthHandler, err := filter(metricsLogger, h)
	if err != nil {
		klog.Fatalf("failed to create metrics auth handler: %v", err)
	}
	return metricsAuthHandler
}

=======
>>>>>>> ad32d85 (Add updated hermetic test with k8s client API, these pull from example object yamls.)
func validateFlags() error {
	if *poolName == "" {
		return fmt.Errorf("required %q flag not set", "poolName")
	}

	if *serviceName == "" {
		return fmt.Errorf("required %q flag not set", "serviceName")
	}

	return nil
}
