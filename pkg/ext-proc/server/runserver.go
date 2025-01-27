package server

import (
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
	TargetPodHeader        string
	PoolName               string
	PoolNamespace          string
	ServiceName            string
	Zone                   string
	RefreshPodsInterval    time.Duration
	RefreshMetricsInterval time.Duration
	Scheme                 *runtime.Scheme
	Config                 *rest.Config
	Datastore              *backend.K8sDatastore
	manager                *ctrl.Manager
}

// Setup creates the reconcilers for pools, models, and endpointSlices and starts the manager.
func (r *ExtProcServerRunner) Setup() {
	// Create a new manager to manage controllers
	mgr, err := ctrl.NewManager(r.Config, ctrl.Options{Scheme: r.Scheme})
	if err != nil {
		klog.Fatalf("Failed to create controller manager: %v", err)
	}
	r.manager = &mgr

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
		klog.Fatalf("Failed setting up InferencePoolReconciler: %v", err)
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
		klog.Fatalf("Failed setting up InferenceModelReconciler: %v", err)
	}

	if err := (&backend.EndpointSliceReconciler{
		Datastore:   r.Datastore,
		Scheme:      mgr.GetScheme(),
		Client:      mgr.GetClient(),
		Record:      mgr.GetEventRecorderFor("endpointslice"),
		ServiceName: r.ServiceName,
		Zone:        r.Zone,
	}).SetupWithManager(mgr); err != nil {
		klog.Fatalf("Failed setting up EndpointSliceReconciler: %v", err)
	}
}

// Start starts the Envoy external processor server in a goroutine.
func (r *ExtProcServerRunner) Start(
	podDatastore *backend.K8sDatastore,
	podMetricsClient backend.PodMetricsClient,
) *grpc.Server {
	svr := grpc.NewServer()

	go func() {
		lis, err := net.Listen("tcp", fmt.Sprintf(":%d", r.GrpcPort))
		if err != nil {
			klog.Fatalf("Ext-proc server failed to listen: %v", err)
		}
		klog.Infof("Ext-proc server listening on port: %d", r.GrpcPort)

		// Initialize backend provider
		pp := backend.NewProvider(podMetricsClient, podDatastore)
		if err := pp.Init(r.RefreshPodsInterval, r.RefreshMetricsInterval); err != nil {
			klog.Fatalf("Failed to initialize backend provider: %v", err)
		}

		// Register ext_proc handlers
		extProcPb.RegisterExternalProcessorServer(
			svr,
			handlers.NewServer(pp, scheduling.NewScheduler(pp), r.TargetPodHeader, r.Datastore),
		)

		// Blocking and will return when shutdown is complete.
		if err := svr.Serve(lis); err != nil && err != grpc.ErrServerStopped {
			klog.Fatalf("Ext-proc server failed: %v", err)
		}
		klog.Info("Ext-proc server shutting down")
	}()
	return svr
}

func (r *ExtProcServerRunner) StartManager() {
	if r.manager == nil {
		klog.Fatalf("Runner has no manager setup to run: %v", r)
	}
	// Start the controller manager. Blocking and will return when shutdown is complete.
	klog.Infof("Starting controller manager")
	mgr := *r.manager
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		klog.Fatalf("Error starting controller manager: %v", err)
	}
	klog.Info("Controller manager shutting down")
}
