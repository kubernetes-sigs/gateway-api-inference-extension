package main

import (
	"flag"
	"fmt"
	"net"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend/vllm"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/handlers"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/scheduling"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
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

	// Validate flags
	if err := validateFlags(); err != nil {
		klog.Fatalf("flag validation failed: %v", err)
	}

	// Print all flag values
	flags := "Flags: "
	flag.VisitAll(func(f *flag.Flag) {
		flags += fmt.Sprintf("%s=%v; ", f.Name, f.Value)
	})
	klog.Info(flags)

	// Create a new manager to manage controllers
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{Scheme: scheme})
	if err != nil {
		klog.Fatalf("failed to start manager: %v", err)
	}

	// Create the data store used to cache watched resources
	datastore := backend.NewK8sDataStore()

	// Create the controllers and register them with the manager
	if err := (&backend.InferencePoolReconciler{
		Datastore: datastore,
		Scheme:    mgr.GetScheme(),
		Client:    mgr.GetClient(),
		PoolNamespacedName: types.NamespacedName{
			Name:      *poolName,
			Namespace: *poolNamespace,
		},
		Record: mgr.GetEventRecorderFor("InferencePool"),
	}).SetupWithManager(mgr); err != nil {
		klog.Fatalf("Error setting up InferencePoolReconciler: %v", err)
	}

	if err := (&backend.InferenceModelReconciler{
		Datastore: datastore,
		Scheme:    mgr.GetScheme(),
		Client:    mgr.GetClient(),
		PoolNamespacedName: types.NamespacedName{
			Name:      *poolName,
			Namespace: *poolNamespace,
		},
		Record: mgr.GetEventRecorderFor("InferenceModel"),
	}).SetupWithManager(mgr); err != nil {
		klog.Fatalf("Error setting up InferenceModelReconciler: %v", err)
	}

	if err := (&backend.EndpointSliceReconciler{
		Datastore:   datastore,
		Scheme:      mgr.GetScheme(),
		Client:      mgr.GetClient(),
		Record:      mgr.GetEventRecorderFor("endpointslice"),
		ServiceName: *serviceName,
		Zone:        *zone,
	}).SetupWithManager(mgr); err != nil {
		klog.Fatalf("Error setting up EndpointSliceReconciler: %v", err)
	}

	// Channel to handle error signals for goroutines
	errChan := make(chan error, 1)

	// Start each component in its own goroutine
	startControllerManager(mgr, errChan)
	healthSvr := startHealthServer(mgr, errChan, *grpcHealthPort)
	extProcSvr := startExternalProcessorServer(
		errChan,
		datastore,
		*grpcPort,
		*refreshPodsInterval,
		*refreshMetricsInterval,
		*targetPodHeader,
	)

	// Wait for first error from any goroutine
	err = <-errChan
	if err != nil {
		klog.Errorf("goroutine failed: %v", err)
	} else {
		klog.Infof("Manager exited gracefully")
	}

	// Gracefully shutdown components
	if healthSvr != nil {
		klog.Info("Health server shutting down...")
		healthSvr.GracefulStop()
	}
	if extProcSvr != nil {
		klog.Info("Ext-proc server shutting down...")
		extProcSvr.GracefulStop()
	}

	klog.Info("All components stopped gracefully")
}

// startControllerManager runs the controller manager in a goroutine.
func startControllerManager(mgr ctrl.Manager, errChan chan<- error) {
	go func() {
		// Blocking and will return when shutdown is complete.
		if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
			errChan <- fmt.Errorf("controller manager failed to start: %w", err)
		}
		// Manager exited gracefully
		klog.Info("Controller manager shutting down...")
		errChan <- nil
	}()
}

// startHealthServer starts the gRPC health probe server in a goroutine.
func startHealthServer(mgr ctrl.Manager, errChan chan<- error, port int) *grpc.Server {
	healthSvr := grpc.NewServer()
	healthPb.RegisterHealthServer(healthSvr, &healthServer{Client: mgr.GetClient()})

	go func() {
		healthLis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
		if err != nil {
			errChan <- fmt.Errorf("health server failed to listen: %w", err)
		}
		klog.Infof("Health server listening on port: %d", port)

		// Blocking and will return when shutdown is complete.
		if serveErr := healthSvr.Serve(healthLis); serveErr != nil && serveErr != grpc.ErrServerStopped {
			errChan <- fmt.Errorf("health server failed: %w", serveErr)
		}
	}()
	return healthSvr
}

// startExternalProcessorServer starts the Envoy external processor server in a goroutine.
func startExternalProcessorServer(
	errChan chan<- error,
	datastore *backend.K8sDatastore,
	port int,
	refreshPodsInterval, refreshMetricsInterval time.Duration,
	targetPodHeader string,
) *grpc.Server {
	extSvr := grpc.NewServer()
	go func() {
		lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
		if err != nil {
			errChan <- fmt.Errorf("ext-proc server failed to listen: %w", err)
		}
		klog.Infof("Ext-proc server listening on port: %d", port)

		// Initialize backend provider
		pp := backend.NewProvider(&vllm.PodMetricsClientImpl{}, datastore)
		if err := pp.Init(refreshPodsInterval, refreshMetricsInterval); err != nil {
			errChan <- fmt.Errorf("failed to initialize backend provider: %w", err)
		}

		// Register ext_proc handlers
		extProcPb.RegisterExternalProcessorServer(
			extSvr,
			handlers.NewServer(pp, scheduling.NewScheduler(pp), targetPodHeader, datastore),
		)

		// Blocking and will return when shutdown is complete.
		if serveErr := extSvr.Serve(lis); serveErr != nil && serveErr != grpc.ErrServerStopped {
			errChan <- fmt.Errorf("ext-proc server failed: %w", serveErr)
		}
	}()
	return extSvr
}

func validateFlags() error {
	if *poolName == "" {
		return fmt.Errorf("required %q flag not set", "poolName")
	}

	if *serviceName == "" {
		return fmt.Errorf("required %q flag not set", "serviceName")
	}

	return nil
}
