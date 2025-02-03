package server

import (
	"context"
	"errors"
	"fmt"
	"net"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/handlers"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/scheduling"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	klog "k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
)

// ExtProcServerRunner provides methods to manage an external process server.
type ExtProcServerRunner struct {
	GrpcPort               int
	TargetEndpointKey      string
	PoolName               string
	PoolNamespace          string
	ServiceName            string
	Zone                   string
	RefreshPodsInterval    time.Duration
	RefreshMetricsInterval time.Duration
	Scheme                 *runtime.Scheme
	Config                 *rest.Config
	Datastore              *backend.K8sDatastore
	manager                ctrl.Manager
}

// Default values for CLI flags in main
const (
	DefaultGrpcPort               = 9002                             // default for --grpcPort
	DefaultTargetEndpointKey      = "x-gateway-destination-endpoint" // default for --targetEndpointKey
	DefaultPoolName               = ""                               // required but no default
	DefaultPoolNamespace          = "default"                        // default for --poolNamespace
	DefaultServiceName            = ""                               // required but no default
	DefaultZone                   = ""                               // default for --zone
	DefaultRefreshPodsInterval    = 10 * time.Second                 // default for --refreshPodsInterval
	DefaultRefreshMetricsInterval = 50 * time.Millisecond            // default for --refreshMetricsInterval
)

func NewDefaultExtProcServerRunner() *ExtProcServerRunner {
	return &ExtProcServerRunner{
		GrpcPort:               DefaultGrpcPort,
		TargetEndpointKey:      DefaultTargetEndpointKey,
		PoolName:               DefaultPoolName,
		PoolNamespace:          DefaultPoolNamespace,
		ServiceName:            DefaultServiceName,
		Zone:                   DefaultZone,
		RefreshPodsInterval:    DefaultRefreshPodsInterval,
		RefreshMetricsInterval: DefaultRefreshMetricsInterval,
		// Scheme, Config, and Datastore can be assigned later.
	}
}

// Setup creates the reconcilers for pools, models, and endpointSlices and starts the manager.
func (r *ExtProcServerRunner) Setup() error {
	// Create a new manager to manage controllers
	mgr, err := ctrl.NewManager(r.Config, ctrl.Options{Scheme: r.Scheme})
	if err != nil {
		return fmt.Errorf("failed to create controller manager: %w", err)
	}
	r.manager = mgr

	// Create the controllers and register them with the manager
	if err := (&backend.InferencePoolReconciler{
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

	if err := (&backend.InferenceModelReconciler{
		Datastore: r.Datastore,
		Scheme:    mgr.GetScheme(),
		Client:    mgr.GetClient(),
		PoolNamespacedName: types.NamespacedName{
			Name:      r.PoolName,
			Namespace: r.PoolNamespace,
		},
		Record: mgr.GetEventRecorderFor("InferenceModel"),
	}).SetupWithManager(mgr); err != nil {
		return fmt.Errorf("failed setting up InferenceModelReconciler: %w", err)
	}

	if err := (&backend.EndpointSliceReconciler{
		Datastore:   r.Datastore,
		Scheme:      mgr.GetScheme(),
		Client:      mgr.GetClient(),
		Record:      mgr.GetEventRecorderFor("endpointslice"),
		ServiceName: r.ServiceName,
		Zone:        r.Zone,
	}).SetupWithManager(mgr); err != nil {
		return fmt.Errorf("failed setting up EndpointSliceReconciler: %w", err)
	}
	return nil
}

// Start starts the Envoy external processor server and blocks
// until the context is canceled or an error encountered.
func (r *ExtProcServerRunner) Start(
	ctx context.Context,
	podDatastore *backend.K8sDatastore,
	podMetricsClient backend.PodMetricsClient,
) error {
	klog.InfoS("Ext-proc server starting...")

	// Start listening.
	svr := grpc.NewServer()
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", r.GrpcPort))
	if err != nil {
		klog.ErrorS(err, "Ext-proc server failed to listen", "port", r.GrpcPort)
		return err
	}
	// The listener will be closed by the server,
	// but the function may also return earlier on error.
	defer lis.Close()

	klog.InfoS("Ext-proc server listening", "port", r.GrpcPort)

	// Initialize backend provider
	pp := backend.NewProvider(podMetricsClient, podDatastore)
	if err := pp.Init(r.RefreshPodsInterval, r.RefreshMetricsInterval); err != nil {
		klog.Fatalf("Failed to initialize backend provider: %v", err)
	}

	// Register ext_proc handlers
	extProcPb.RegisterExternalProcessorServer(
		svr,
		handlers.NewServer(pp, scheduling.NewScheduler(pp), r.TargetEndpointKey, r.Datastore),
	)

	// Terminate the server on context closed.
	// Make sure the goroutine does not leak.
	doneCh := make(chan struct{})
	defer close(doneCh)
	go func() {
		select {
		case <-ctx.Done():
			klog.InfoS("Ext-proc server shutting down...")
			svr.GracefulStop()
		case <-doneCh:
		}
	}()

	// Block until terminated.
	if err := svr.Serve(lis); err != nil && err != grpc.ErrServerStopped {
		klog.ErrorS(err, "Ext-proc server failed")
		return err
	}
	klog.InfoS("Ext-proc server terminated")
	return nil
}

func (r *ExtProcServerRunner) StartManager(ctx context.Context) error {
	if r.manager == nil {
		err := errors.New("runner manager is not set")
		klog.ErrorS(err, "Runner has no manager setup to run")
		return err
	}

	// Start the controller manager. Blocking and will return when shutdown is complete.
	klog.InfoS("Controller manager starting...")
	mgr := r.manager
	if err := mgr.Start(ctx); err != nil {
		klog.ErrorS(err, "Error starting controller manager")
		return err
	}
	klog.InfoS("Controller manager terminated")
	return nil
}
