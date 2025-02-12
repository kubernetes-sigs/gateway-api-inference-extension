package server

import (
	"crypto/tls"
	"fmt"
	"log"
	"net"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
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
	RefreshPodsInterval    time.Duration
	RefreshMetricsInterval time.Duration
	Scheme                 *runtime.Scheme
	Config                 *rest.Config
	Datastore              *backend.K8sDatastore
	Manager                ctrl.Manager
}

// Default values for CLI flags in main
const (
	DefaultGrpcPort               = 9002                             // default for --grpcPort
	DefaultTargetEndpointKey      = "x-gateway-destination-endpoint" // default for --targetEndpointKey
	DefaultPoolName               = ""                               // required but no default
	DefaultPoolNamespace          = "default"                        // default for --poolNamespace
	DefaultRefreshPodsInterval    = 10 * time.Second                 // default for --refreshPodsInterval
	DefaultRefreshMetricsInterval = 50 * time.Millisecond            // default for --refreshMetricsInterval
)

func NewDefaultExtProcServerRunner() *ExtProcServerRunner {
	return &ExtProcServerRunner{
		GrpcPort:               DefaultGrpcPort,
		TargetEndpointKey:      DefaultTargetEndpointKey,
		PoolName:               DefaultPoolName,
		PoolNamespace:          DefaultPoolNamespace,
		RefreshPodsInterval:    DefaultRefreshPodsInterval,
		RefreshMetricsInterval: DefaultRefreshMetricsInterval,
		// Scheme, Config, and Datastore can be assigned later.
	}
}

// Setup creates the reconcilers for pools and models and starts the manager.
func (r *ExtProcServerRunner) Setup() {
	// Create a new manager to manage controllers
	mgr, err := ctrl.NewManager(r.Config, ctrl.Options{Scheme: r.Scheme})
	if err != nil {
		klog.Fatalf("Failed to create controller manager: %v", err)
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
}

// Start starts the Envoy external processor server in a goroutine.
func (r *ExtProcServerRunner) Start(
	podMetricsClient backend.PodMetricsClient,
) *grpc.Server {

	certString := `-----BEGIN CERTIFICATE-----
MIIFlTCCA32gAwIBAgIUPUu7ZHNHXQhcOPt3ubiQZ54L8OYwDQYJKoZIhvcNAQEL
BQAwWjELMAkGA1UEBhMCVVMxEjAQBgNVBAgMCVlvdXJTdGF0ZTERMA8GA1UEBwwI
WW91ckNpdHkxEDAOBgNVBAoMB1lvdXJPcmcxEjAQBgNVBAMMCWxvY2FsaG9zdDAe
Fw0yNTAyMTEwMzUxMzBaFw0yNjAyMTEwMzUxMzBaMFoxCzAJBgNVBAYTAlVTMRIw
EAYDVQQIDAlZb3VyU3RhdGUxETAPBgNVBAcMCFlvdXJDaXR5MRAwDgYDVQQKDAdZ
b3VyT3JnMRIwEAYDVQQDDAlsb2NhbGhvc3QwggIiMA0GCSqGSIb3DQEBAQUAA4IC
DwAwggIKAoICAQC/HG2QS4dV57c2G5RultZ2Zq/7k0aHfI2ujUt2QyvOOy3RFGB4
rPV8nGDlJd+3nNUpVQG/O8s2mW0IvcI6s9uQ2lC96RtVkfX5iedIhQoxWK5tunAr
iUk9orac2lWQFC2Yq9UH8Mr7BAFUIUZQXDh4wPQ8f06Itmw9XejDQMyDkuAoMGwa
0k8a8g6QuioVhGXj2T01Vzez+OnLa5U9YDee4txxh67vAg7jrEoK2oLo009p9Do4
wgouR6HkstiXEaLq1ED3oxMgOo1A9lCEmDffZTb0Qx2vBJZbYZLBKW3L72VbkzHm
jebqjgpld/a1LhwkKeKLcv+2+arpFcmbtahxIhEJQZgN/2/k0gem0hcfbiOatqyT
S/aF7PwO/elVFETK+Lqd2QvzIjZA2Pr3y3DNeufelR3nw3NX0f0woaLfNI5RADDY
dWrgS3hYudO/uUt2bOZLFqrib1kTUdr9t8Nn4N8kagsK5N6+r4OhdNGNfa0dA0M5
E4LWQLyyQQN/sihmWF1OBgorlGJicMqYA9S5mrsj6IuRySM8PW/X2tPcnV/KiCTT
k2VwzVY4bX0KIWzlEYaehlLBZ955VlXWk9rNSsJAFKjr9AhxkWbl2r56pkNFuFHl
CR0xhlo8yvDGmw+Abs9/HmI0f69JiEY5fFULeYngxBJLcNcEOgWP5iVPewIDAQAB
o1MwUTAdBgNVHQ4EFgQUKwQNh+Ds6TqhPa2vutDMJ9NVCX8wHwYDVR0jBBgwFoAU
KwQNh+Ds6TqhPa2vutDMJ9NVCX8wDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0B
AQsFAAOCAgEApkLKyolzVIN82mmDxng9Gbjfmda2Mf5qTDGwBJJ5w0YoH2+t76bt
hFbw6NblHkmbfMsqwr65J+HYoc6Iwgc2O+WIWEiC299hFKEpUoCygifC3yUBUEPV
aaqG0lv6+ULECqDo35F6AZzPYC2D7QCPlGdjjoXbPrH3YU3w1upVrPDQzCqggo7a
3VuOLh5Stk6MgJihs44J++yWQOsMmygB6/J9l2VDtZCqPsGmzl4Wq4aWnSgSMZVg
qrGyjiIvmP60EQLmqIXnTKRmFH3pHJIghZPzRa+RVZsnyksAF4+yDsE9XdqznDmF
Jb3i1iOxQ+PbtHjdsYO470H6HY1dymHXFjtmNUyvtvh+B0VvnblYPBra+4pMRbc8
/3KRqEQNgMmwFi4xlWT4uUQCHEyF+zzSMHGOw0vkCU+6IF3gctk4s6ClMmhgPRGo
C0WGdkqNQn92H8eSBLHXWc0X6BJjgaCW66C8LLyK1z13zE1llgyRkGXF/1yCHNeE
NIG5HSWV/zWfFnwd6MAr7H1JUMFe0nz6BIiaO8ahBNTHfNhWZDLFOPcg5R1f3TPe
ZzNHVOy4c2VOv6zFkCwDon9eZb1x1zkXs5D4GYGiPCcB20+A7GpqGjuF7pODxRdV
XO43KcqhiqErycwIKLfhPt0vHbnBNYJNAGu4OWTJr4ikXR3AUOkoVYE=
-----END CERTIFICATE-----`

	keyString := `-----BEGIN PRIVATE KEY-----
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
0SJmM+295B4SHANLsbqetlEfu8xP1uc=
-----END PRIVATE KEY-----`

	// Create tls based credential.
	cert, err := createTLSCertificateFromStrings(certString, keyString)
	if err != nil {
		log.Fatalf("Error creating TLS certificate: %v", err)
	}

	creds := credentials.NewTLS(&tls.Config{
		Certificates: []tls.Certificate{cert},
	})

	svr := grpc.NewServer(grpc.Creds(creds))

	go func() {
		lis, err := net.Listen("tcp", fmt.Sprintf(":%d", r.GrpcPort))
		if err != nil {
			klog.Fatalf("Ext-proc server failed to listen: %v", err)
		}
		klog.Infof("Ext-proc server listening on port: %d", r.GrpcPort)

		// Initialize backend provider
		pp := backend.NewProvider(podMetricsClient, r.Datastore)
		if err := pp.Init(r.RefreshPodsInterval, r.RefreshMetricsInterval); err != nil {
			klog.Fatalf("Failed to initialize backend provider: %v", err)
		}

		// Register ext_proc handlers
		extProcPb.RegisterExternalProcessorServer(
			svr,
			handlers.NewServer(pp, scheduling.NewScheduler(pp), r.TargetEndpointKey, r.Datastore),
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
	if r.Manager == nil {
		klog.Fatalf("Runner has no manager setup to run: %v", r)
	}
	// Start the controller manager. Blocking and will return when shutdown is complete.
	klog.Infof("Starting controller manager")
	if err := r.Manager.Start(ctrl.SetupSignalHandler()); err != nil {
		klog.Fatalf("Error starting controller manager: %v", err)
	}
	klog.Info("Controller manager shutting down")
}

func createTLSCertificateFromStrings(certString, keyString string) (tls.Certificate, error) {
	certBytes := []byte(certString)
	keyBytes := []byte(keyString)

	cert, err := tls.X509KeyPair(certBytes, keyBytes)
	if err != nil {
		return tls.Certificate{}, err // Return empty certificate and the error
	}

	return cert, nil
}
