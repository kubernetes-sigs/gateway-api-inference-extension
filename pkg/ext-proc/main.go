package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
	"inference.networking.x-k8s.io/llm-instance-gateway/api/v1alpha1"
	"inference.networking.x-k8s.io/llm-instance-gateway/pkg/ext-proc/backend"
	"inference.networking.x-k8s.io/llm-instance-gateway/pkg/ext-proc/backend/vllm"
	"inference.networking.x-k8s.io/llm-instance-gateway/pkg/ext-proc/handlers"
	"inference.networking.x-k8s.io/llm-instance-gateway/pkg/ext-proc/scheduling"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	klog "k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
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
	targetPodHeader = flag.String(
		"targetPodHeader",
		"target-pod",
		"Header key used by Envoy to route to the appropriate pod. This must match Envoy configuration.")
	serverPoolName = flag.String(
		"serverPoolName",
		"",
		"Name of the serverPool this Endpoint Picker is associated with.")
	serviceName = flag.String(
		"serviceName",
		"",
		"Name of the service that will be used to read the endpointslices from")
	namespace = flag.String(
		"namespace",
		"default",
		"The Namespace that the server pool should exist in.")
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

type healthServer struct {
	datastore *backend.K8sDatastore
}

func (s *healthServer) Check(ctx context.Context, in *healthPb.HealthCheckRequest) (*healthPb.HealthCheckResponse, error) {
	if !s.datastore.IsReady() {
		klog.Infof("gRPC health check not serving: %s", in.String())
		return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_NOT_SERVING}, nil
	}
	klog.Infof("gRPC health check serving: %s", in.String())
	return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_SERVING}, nil
}

func (s *healthServer) Watch(in *healthPb.HealthCheckRequest, srv healthPb.Health_WatchServer) error {
	return status.Error(codes.Unimplemented, "Watch is not implemented")
}

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(v1alpha1.AddToScheme(scheme))
}

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	ctrl.SetLogger(klog.TODO())

	// Print all flag values
	flags := "Flags: "
	flag.VisitAll(func(f *flag.Flag) {
		flags += fmt.Sprintf("%s=%v; ", f.Name, f.Value)
	})
	klog.Info(flags)

	// Create a WaitGroup to manage shutdown of all components
	var wg sync.WaitGroup

	// Error channel to handle server errors
	errChan := make(chan error, 5)

	// Channel to handle graceful shutdown
	shutdownChan := make(chan os.Signal, 1)
	signal.Notify(shutdownChan, syscall.SIGINT, syscall.SIGTERM)

	// Create a new manager for managing controllers
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme: scheme,
	})
	if err != nil {
		klog.Fatalf("unable to start manager: %v", err)
	}

	// Create the data store used to cache watched resources
	datastore := backend.NewK8sDataStore()

	// Create the controllers and register them with the manager
	if err := (&backend.InferencePoolReconciler{
		Datastore:      datastore,
		Scheme:         mgr.GetScheme(),
		Client:         mgr.GetClient(),
		ServerPoolName: *serverPoolName,
		Namespace:      *namespace,
		Record:         mgr.GetEventRecorderFor("InferencePool"),
	}).SetupWithManager(mgr); err != nil {
		klog.Fatalf("Error setting up InferencePoolReconciler: %v", err)
	}

	if err := (&backend.InferenceModelReconciler{
		Datastore:      datastore,
		Scheme:         mgr.GetScheme(),
		Client:         mgr.GetClient(),
		ServerPoolName: *serverPoolName,
		Namespace:      *namespace,
		Record:         mgr.GetEventRecorderFor("InferenceModel"),
	}).SetupWithManager(mgr); err != nil {
		klog.Fatalf("Error setting up InferenceModelReconciler: %v", err)
	}

	if err := (&backend.EndpointSliceReconciler{
		Datastore:      datastore,
		Scheme:         mgr.GetScheme(),
		Client:         mgr.GetClient(),
		Record:         mgr.GetEventRecorderFor("endpointslice"),
		ServiceName:    *serviceName,
		Zone:           *zone,
		ServerPoolName: *serverPoolName,
	}).SetupWithManager(mgr); err != nil {
		klog.Fatalf("Error setting up EndpointSliceReconciler: %v", err)
	}

	// Start the controller manager
	wg.Add(1)
	go func() {
		defer wg.Done()

		if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
			errChan <- fmt.Errorf("controller manager failed to start: %w", err)
		}
		klog.Info("controller manager started")
	}()

	// Start the health server
	var healthSvr *grpc.Server
	wg.Add(1)
	go func() {
		defer wg.Done()

		healthSvr = grpc.NewServer()
		healthPb.RegisterHealthServer(healthSvr, &healthServer{datastore: datastore})
		healthLis, err := net.Listen("tcp", fmt.Sprintf(":%d", *grpcHealthPort))
		if err != nil {
			errChan <- fmt.Errorf("health server failed to listen: %w", err)
			return
		}

		if err := healthSvr.Serve(healthLis); err != nil && err != grpc.ErrServerStopped {
			errChan <- fmt.Errorf("health server failed: %w", err)
		}
		klog.Infof("Health server serving on port: %d", *grpcHealthPort)
	}()

	// Start the ext-proc server
	var extSvr *grpc.Server
	wg.Add(1)
	go func() {
		defer wg.Done()

		extSvr = grpc.NewServer()
		lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *grpcPort))
		if err != nil {
			errChan <- fmt.Errorf("ext-proc server failed to listen: %w", err)
			return
		}

		pp := backend.NewProvider(&vllm.PodMetricsClientImpl{}, datastore)
		if err := pp.Init(*refreshPodsInterval, *refreshMetricsInterval); err != nil {
			errChan <- fmt.Errorf("failed to initialize backend provider: %w", err)
			return
		}
		extProcPb.RegisterExternalProcessorServer(
			extSvr,
			handlers.NewServer(
				pp,
				scheduling.NewScheduler(pp),
				*targetPodHeader,
				datastore,
			),
		)

		if err := extSvr.Serve(lis); err != nil && err != grpc.ErrServerStopped {
			errChan <- fmt.Errorf("ext-proc server failed: %w", err)
		}
		klog.Infof("Ext-proc server serving on port: %d", *grpcPort)
	}()

	// Monitor for shutdown and error signals
	select {
	case <-shutdownChan:
		klog.Info("Shutdown signal received, stopping servers...")
	case err := <-errChan:
		klog.Errorf("Fatal error: %v", err)
	}

	if healthSvr != nil {
		healthSvr.GracefulStop()
	}
	if extSvr != nil {
		extSvr.GracefulStop()
	}

	// Wait for all goroutines to finish
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		klog.Info("All components stopped gracefully")
	case <-shutdownCtx.Done():
		klog.Errorf("Shutdown timed out, forcing exit")
	}
}
