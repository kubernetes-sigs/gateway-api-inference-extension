package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"strconv"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"golang.org/x/sync/errgroup"
	"google.golang.org/grpc"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend/vllm"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/metrics"
	runserver "inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/server"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/kubernetes"
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
	refreshPodsInterval = flag.Duration(
		"refreshPodsInterval",
		runserver.DefaultRefreshPodsInterval,
		"interval to refresh pods")
	refreshMetricsInterval = flag.Duration(
		"refreshMetricsInterval",
		runserver.DefaultRefreshMetricsInterval,
		"interval to refresh metrics")

	scheme = runtime.NewScheme()
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
	klog.InitFlags(nil)
	flag.Parse()

	ctrl.SetLogger(klog.TODO())
	cfg, err := ctrl.GetConfig()
	if err != nil {
		klog.ErrorS(err, "Failed to get rest config")
		return err
	}
	// Validate flags
	if err := validateFlags(); err != nil {
		klog.ErrorS(err, "Failed to validate flags")
		return err
	}

	// Print all flag values
	flags := "Flags: "
	flag.VisitAll(func(f *flag.Flag) {
		flags += fmt.Sprintf("%s=%v; ", f.Name, f.Value)
	})
	klog.Info(flags)

	datastore := backend.NewK8sDataStore()

	serverRunner := &runserver.ExtProcServerRunner{
		GrpcPort:               *grpcPort,
		TargetEndpointKey:      *targetEndpointKey,
		PoolName:               *poolName,
		PoolNamespace:          *poolNamespace,
		RefreshPodsInterval:    *refreshPodsInterval,
		RefreshMetricsInterval: *refreshMetricsInterval,
		Scheme:                 scheme,
		Config:                 ctrl.GetConfigOrDie(),
		Datastore:              datastore,
	}
	if err := serverRunner.Setup(); err != nil {
		klog.ErrorS(err, "Failed to setup ext-proc server")
		return err
	}

	k8sClient, err := kubernetes.NewForConfigAndClient(cfg, serverRunner.Manager.GetHTTPClient())
	if err != nil {
		klog.ErrorS(err, "Failed to create client")
		return err
	}
	datastore.SetClient(k8sClient)

	if err := serverRunner.Setup(); err != nil {
		klog.ErrorS(err, "Failed to setup server runner")
		return err
	}

	// Start processing signals and init the group to manage goroutines.
	g, ctx := errgroup.WithContext(ctrl.SetupSignalHandler())

	// Start health server.
	startHealthServer(ctx, g, datastore, *grpcHealthPort)

	// Start ext-proc server.
	g.Go(func() error {
		return serverRunner.Start(ctx, &vllm.PodMetricsClientImpl{})
	})

	// Start metrics handler.
	startMetricsHandler(ctx, g, *metricsPort, cfg)

	// Start manager.
	g.Go(func() error {
		return serverRunner.StartManager(ctx)
	})

	err = g.Wait()
	klog.InfoS("All components terminated")
	return err
}

// startHealthServer starts the gRPC health probe server using the given errgroup.
func startHealthServer(ctx context.Context, g *errgroup.Group, ds *backend.K8sDatastore, port int) {
	g.Go(func() error {
		klog.InfoS("Health server starting...")

		// Start listening.
		lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
		if err != nil {
			klog.ErrorS(err, "Health server failed to listen")
			return err
		}

		klog.InfoS("Health server listening", "port", port)

		svr := grpc.NewServer()
		healthPb.RegisterHealthServer(svr, &healthServer{datastore: ds})

		// Shutdown on context closed.
		g.Go(func() error {
			<-ctx.Done()
			klog.InfoS("Health server shutting down...")
			svr.GracefulStop()
			return nil
		})

		// Keep serving until terminated.
		if err := svr.Serve(lis); err != nil && err != grpc.ErrServerStopped {
			klog.ErrorS(err, "Health server failed")
			return err
		}
		klog.InfoS("Health server terminated")
		return nil
	})
}

// startMetricsHandler starts the metrics HTTP handler using the given errgroup.
func startMetricsHandler(ctx context.Context, g *errgroup.Group, port int, cfg *rest.Config) {
	g.Go(func() error {
		metrics.Register()
		klog.InfoS("Metrics HTTP handler starting...")

		// Start listening.
		lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
		if err != nil {
			klog.ErrorS(err, "Metrics HTTP handler failed to listen")
			return err
		}

		klog.InfoS("Metrics HTTP handler listening", "port", port)

		// Init HTTP server.
		h, err := metricsHandlerWithAuthenticationAndAuthorization(cfg)
		if err != nil {
			return err
		}

		mux := http.NewServeMux()
		mux.Handle(defaultMetricsEndpoint, h)

		svr := &http.Server{
			Addr:    net.JoinHostPort("", strconv.Itoa(port)),
			Handler: mux,
		}

		// Shutdown on interrupt.
		g.Go(func() error {
			<-ctx.Done()
			klog.InfoS("Metrics HTTP handler shutting down...")
			_ = svr.Shutdown(context.Background())
			return nil
		})

		// Keep serving until terminated.
		if err := svr.Serve(lis); err != http.ErrServerClosed {
			klog.ErrorS(err, "Metrics HTTP handler failed")
			return err
		}
		klog.InfoS("Metrics HTTP handler terminated")
		return nil
	})
}

func metricsHandlerWithAuthenticationAndAuthorization(cfg *rest.Config) (http.Handler, error) {
	h := promhttp.HandlerFor(
		legacyregistry.DefaultGatherer,
		promhttp.HandlerOpts{},
	)
	httpClient, err := rest.HTTPClientFor(cfg)
	if err != nil {
		klog.ErrorS(err, "Failed to create http client for metrics auth")
		return nil, err
	}

	filter, err := filters.WithAuthenticationAndAuthorization(cfg, httpClient)
	if err != nil {
		klog.ErrorS(err, "Failed to create metrics filter for auth")
		return nil, err
	}
	metricsLogger := klog.LoggerWithValues(klog.NewKlogr(), "path", defaultMetricsEndpoint)
	metricsAuthHandler, err := filter(metricsLogger, h)
	if err != nil {
		klog.ErrorS(err, "Failed to create metrics auth handler")
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
