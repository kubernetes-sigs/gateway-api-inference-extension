package server

import (
	"context"
	"errors"
	"fmt"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	klog "k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/handlers"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/internal/runnable"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/scheduling"
)

// ExtProcServerRunner provides methods to manage an external process server.
type ExtProcServerRunner struct {
	GrpcPort                         int
	TargetEndpointKey                string
	PoolName                         string
	PoolNamespace                    string
	ServiceName                      string
	Zone                             string
	RefreshPodsInterval              time.Duration
	RefreshMetricsInterval           time.Duration
	RefreshPrometheusMetricsInterval time.Duration
	Scheme                           *runtime.Scheme
	Config                           *rest.Config
	Datastore                        *backend.K8sDatastore
	Manager                          ctrl.Manager
}

// Default values for CLI flags in main
const (
	DefaultGrpcPort                         = 9002                             // default for --grpcPort
	DefaultTargetEndpointKey                = "x-gateway-destination-endpoint" // default for --targetEndpointKey
	DefaultPoolName                         = ""                               // required but no default
	DefaultPoolNamespace                    = "default"                        // default for --poolNamespace
	DefaultServiceName                      = ""                               // required but no default
	DefaultZone                             = ""                               // default for --zone
	DefaultRefreshPodsInterval              = 10 * time.Second                 // default for --refreshPodsInterval
	DefaultRefreshMetricsInterval           = 50 * time.Millisecond            // default for --refreshMetricsInterval
	DefaultRefreshPrometheusMetricsInterval = 5 * time.Second                  // default for --refreshPrometheusMetricsInterval
)

func NewDefaultExtProcServerRunner() *ExtProcServerRunner {
	return &ExtProcServerRunner{
		GrpcPort:                         DefaultGrpcPort,
		TargetEndpointKey:                DefaultTargetEndpointKey,
		PoolName:                         DefaultPoolName,
		PoolNamespace:                    DefaultPoolNamespace,
		ServiceName:                      DefaultServiceName,
		Zone:                             DefaultZone,
		RefreshPodsInterval:              DefaultRefreshPodsInterval,
		RefreshMetricsInterval:           DefaultRefreshMetricsInterval,
		RefreshPrometheusMetricsInterval: DefaultRefreshPrometheusMetricsInterval,
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
	r.Manager = mgr

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
		return fmt.Errorf("failed setting up EndpointSliceReconciler: %v", err)
	}
	return nil
}

// AsRunnable returns a Runnable that can be used to start the ext-proc gRPC server.
// The runnable implements LeaderElectionRunnable with leader election disabled.
func (r *ExtProcServerRunner) AsRunnable(
	podDatastore *backend.K8sDatastore,
	podMetricsClient backend.PodMetricsClient,
) manager.Runnable {
	return runnable.NoLeaderElection(manager.RunnableFunc(func(ctx context.Context) error {
		// Initialize backend provider
		pp := backend.NewProvider(podMetricsClient, podDatastore)
		if err := pp.Init(r.RefreshPodsInterval, r.RefreshMetricsInterval, r.RefreshPrometheusMetricsInterval); err != nil {
			klog.ErrorS(err, "Failed to initialize backend provider")
			return err
		}

		// Init the server.
		srv := grpc.NewServer()
		extProcPb.RegisterExternalProcessorServer(
			srv,
			handlers.NewServer(pp, scheduling.NewScheduler(pp), r.TargetEndpointKey, r.Datastore),
		)

		// Forward to the gRPC runnable.
		return runnable.GRPCServer("ext-proc", srv, r.GrpcPort).Start(ctx)
	}))
}

func (r *ExtProcServerRunner) StartManager(ctx context.Context) error {
	if r.Manager == nil {
		err := errors.New("runner manager is not set")
		klog.ErrorS(err, "Runner has no manager setup to run")
		return err
	}

	// Start the controller manager. Blocking and will return when shutdown is complete.
	klog.InfoS("Controller manager starting")
	if err := r.Manager.Start(ctx); err != nil {
		klog.ErrorS(err, "Error starting controller manager")
		return err
	}
	klog.InfoS("Controller manager terminated")
	return nil
}
