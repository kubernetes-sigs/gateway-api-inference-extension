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

package runner

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"

	uberzap "go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"google.golang.org/grpc"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	"sigs.k8s.io/gateway-api-inference-extension/internal/runnable"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/metrics"
	bbr "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins"
	routing "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/routing"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/server"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/profiling"
	"sigs.k8s.io/gateway-api-inference-extension/version"
)

var (
	// Flags
	grpcPort            = flag.Int("grpc-port", 9004, "The gRPC port used for communicating with Envoy proxy")
	grpcHealthPort      = flag.Int("grpc-health-port", 9005, "The port used for gRPC liveness and readiness probes")
	metricsPort         = flag.Int("metrics-port", 9090, "The metrics port")
	metricsEndpointAuth = flag.Bool("metrics-endpoint-auth", true, "Enables authentication and authorization of the metrics endpoint")
	streaming           = flag.Bool("streaming", false, "Enables streaming support for Envoy full-duplex streaming mode")
	secureServing       = flag.Bool("secure-serving", true, "Enables secure serving.")
	logVerbosity        = flag.Int("v", logutil.DEFAULT, "number for the log level verbosity")
	enablePprof         = flag.Bool("enable-pprof", true, "Enables pprof handlers. Defaults to true. Set to false to disable pprof handlers.")

	// Logging
	setupLog = ctrl.Log.WithName("setup")

	// Contains the BBR plugins specs specified via repeated flags:
	//   --plugin <type>:<name>[:<json>]
	pluginSpecs bbr.BBRPluginSpecs
)

func NewRunner() *Runner {
	return &Runner{
		bbrExecutableName: "BBR",
	}
}

// Runner is used to run bbr with its plugins
type Runner struct {
	bbrExecutableName string

	// The slice of BBR plugin instances executed by the request handler,
	// in the same order the plugin flags are provided.
	bbrPluginInstances []bbr.BBRPlugin
}

// WithExecutableName sets the name of the executable containing the runner.
// The name is used in the version log upon startup and is otherwise opaque.
func (r *Runner) WithExecutableName(exeName string) *Runner {
	r.bbrExecutableName = exeName
	return r
}

func (r *Runner) Run(ctx context.Context) error {
	setupLog.Info(r.bbrExecutableName+" build", "commit-sha", version.CommitSHA, "build-ref", version.BuildRef)
	opts := zap.Options{Development: true}
	opts.BindFlags(flag.CommandLine)

	flag.Var(&pluginSpecs, "plugin", `Repeatable. --plugin <type>:<name>[:<json>]`)
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

	ds := datastore.NewDatastore()

	metrics.Register()
	// Register metrics handler.
	// Metrics endpoint is enabled in 'config/default/kustomization.yaml'. The Metrics options configure the server.
	// More info:
	// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.19.1/pkg/metrics/server
	// - https://book.kubebuilder.io/reference/metrics.html
	metricsServerOptions := metricsserver.Options{
		BindAddress: fmt.Sprintf(":%d", *metricsPort),
		FilterProvider: func() func(c *rest.Config, httpClient *http.Client) (metricsserver.Filter, error) {
			if *metricsEndpointAuth {
				return filters.WithAuthenticationAndAuthorization
			}

			return nil
		}(),
	}
	// label "inference.networking.k8s.io/bbr-managed" = "true" is used for server-side filtering of configmaps.
	// only the configmap objects with this label will be tracked by bbr.
	cacheOptions := cache.Options{
		ByObject: map[client.Object]cache.ByObject{
			&corev1.ConfigMap{}: {
				Label: labels.SelectorFromSet(labels.Set{
					"inference.networking.k8s.io/bbr-managed": "true",
				}),
			},
		},
	}
	// Apply namespace filtering only if env var is set
	namespace := os.Getenv("NAMESPACE")
	if namespace != "" {
		cacheOptions.DefaultNamespaces = map[string]cache.Config{
			namespace: {},
		}
	}

	mgr, err := ctrl.NewManager(cfg, ctrl.Options{Cache: cacheOptions, Metrics: metricsServerOptions})
	if err != nil {
		setupLog.Error(err, "Failed to create manager", "config", cfg)
		return err
	}

	if *enablePprof {
		setupLog.Info("Setting pprof handlers")
		if err = profiling.SetupPprofHandlers(mgr); err != nil {
			setupLog.Error(err, "Failed to setup pprof handlers")
			return err
		}
	}

	// Register factories for all known in-tree BBR plugins
	r.registerInTreePlugins()

	// Construct BBR plugin instances for the in tree plugins that are (1) registered and (2) requested via the --plugin flags
	if len(pluginSpecs) == 0 {
		setupLog.Info("No BBR plugins are specified. Running BBR with the default behavior.")

		// Append a default BBRPlugin to the slice of the BBRPlugin instances using regular registered factory mechanism.
		factory := bbr.Registry[routing.DefaultPluginType]
		defaultPlugin, err := factory("", nil)
		if err != nil {
			setupLog.Error(err, "Failed to create default plugin")
			return err
		}
		r.withPlugin(defaultPlugin)
	} else {
		setupLog.Info("BBR plugins are specified. Running BBR with the specified plugins.")

		for _, s := range pluginSpecs {
			factory, ok := bbr.Registry[s.Type]
			if !ok {
				setupLog.Error(err, fmt.Sprintf("unknown plugin type %q (no factory registered)\n", s.Type))
			}
			instance, err := factory(s.Name, s.JSON)
			if err != nil {
				setupLog.Error(err, fmt.Sprintf("invalid %s#%s: %v\n", s.Type, s.Name, err))
			}
			r.withPlugin(instance)
		}
	}

	// Setup ExtProc Server Runner
	serverRunner := &runserver.ExtProcServerRunner{
		GrpcPort:        *grpcPort,
		Datastore:       ds,
		SecureServing:   *secureServing,
		Streaming:       *streaming,
		PluginInstances: r.bbrPluginInstances,
	}
	if err := serverRunner.SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "Failed to setup BBR controllers")
		return err
	}

	// Register health server.
	if err := registerHealthServer(mgr, *grpcHealthPort); err != nil {
		return err
	}

	// Register ext-proc server.
	if err := mgr.Add(serverRunner.AsRunnable(ctrl.Log.WithName("ext-proc"))); err != nil {
		setupLog.Error(err, "Failed to register ext-proc gRPC server")
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

// registerInTreePlugins registers the factory functions of all known BBR plugins
func (r *Runner) registerInTreePlugins() {
	bbr.Register(routing.DefaultPluginType, routing.DefaultPluginFactory)
}

func (r *Runner) withPlugin(p bbr.BBRPlugin) {
	r.bbrPluginInstances = append(r.bbrPluginInstances, p)
}

// registerHealthServer adds the Health gRPC server as a Runnable to the given manager.
func registerHealthServer(mgr manager.Manager, port int) error {
	srv := grpc.NewServer()
	healthPb.RegisterHealthServer(srv, &healthServer{})
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

	logutil.InitLogging(opts)
}
