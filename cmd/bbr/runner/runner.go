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
	"fmt"
	"net/http"
	"os"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/pflag"
	"google.golang.org/grpc"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	"sigs.k8s.io/gateway-api-inference-extension/internal/runnable"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins/basemodelextractor"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins/bodyfieldtoheader"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/server"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/profiling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/tracing"
	"sigs.k8s.io/gateway-api-inference-extension/version"
)

const modelField = "model"

var setupLog = ctrl.Log.WithName("setup")

func NewRunner() *Runner {
	return &Runner{
		bbrExecutableName: "BBR",
		requestPlugins:    []framework.RequestProcessor{},
		responsePlugins:   []framework.ResponseProcessor{},
		customCollectors:  []prometheus.Collector{},
	}
}

// Runner is used to run bbr with its plugins
type Runner struct {
	bbrExecutableName string
	// The slice of BBR plugin instances executed by the request handler,
	// in the same order the plugin flags are provided.
	requestPlugins []framework.RequestProcessor
	// The slice of BBR plugin instances executed by the response handler,
	// in the same order the plugin flags are provided.
	responsePlugins []framework.ResponseProcessor

	customCollectors []prometheus.Collector
}

func (r *Runner) WithCustomCollectors(collectors ...prometheus.Collector) *Runner {
	r.customCollectors = collectors
	return r
}

func (r *Runner) Run(ctx context.Context) error {
	// Setup a basic logger in case command-line argument parsing fails.
	logutil.InitSetupLogging()

	setupLog.Info(r.bbrExecutableName+" build", "commit-sha", version.CommitSHA, "build-ref", version.BuildRef)

	opts := runserver.NewOptions()
	opts.AddFlags(pflag.CommandLine)
	pflag.Parse()

	if err := opts.Complete(); err != nil {
		return err
	}
	if err := opts.Validate(); err != nil {
		setupLog.Error(err, "Failed to validate flags")
		return err
	}

	// Print all flag values.
	flags := make(map[string]any)
	pflag.VisitAll(func(f *pflag.Flag) {
		flags[f.Name] = f.Value
	})

	if opts.Tracing {
		err := tracing.InitTracing(ctx, setupLog, "gateway-api-inference-extension/bbr")
		if err != nil {
			setupLog.Error(err, "failed to initialize tracing")
			return err
		}
	}

	setupLog.Info("Flags processed", "flags", flags)

	logutil.InitLogging(&opts.ZapOptions)

	// Init runtime.
	cfg, err := ctrl.GetConfig()
	if err != nil {
		setupLog.Error(err, "Failed to get rest config")
		return err
	}

	// --- Setup Metrics Server ---
	metrics.Register(r.customCollectors...)
	metrics.RecordBBRInfo(version.CommitSHA, version.BuildRef)
	// Register metrics handler.
	// Metrics endpoint is enabled in 'config/default/kustomization.yaml'. The Metrics options configure the server.
	// More info:
	// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.19.1/pkg/metrics/server
	// - https://book.kubebuilder.io/reference/metrics.html
	metricsServerOptions := metricsserver.Options{
		BindAddress: fmt.Sprintf(":%d", opts.MetricsPort),
		FilterProvider: func() func(c *rest.Config, httpClient *http.Client) (metricsserver.Filter, error) {
			if opts.MetricsEndpointAuth {
				return filters.WithAuthenticationAndAuthorization
			}

			return nil
		}(),
	}
	cacheOptions := cache.Options{}
	// Apply namespace filtering only if env var is set.
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

	if opts.EnablePprof {
		setupLog.Info("Setting pprof handlers")
		if err = profiling.SetupPprofHandlers(mgr); err != nil {
			setupLog.Error(err, "Failed to setup pprof handlers")
			return err
		}
	}

	bbrHandle := framework.NewBbrHandle(ctx, mgr)

	// Register factories for all known in-tree BBR plugins
	r.registerInTreePlugins()

	// Construct BBR plugin instances for the in-tree plugins that are (1) registered and (2) requested via the --plugin flags
	if len(opts.PluginSpecs) == 0 {
		setupLog.Info("No BBR plugins are specified. Running BBR with the default behavior.")

		modelToHeaderPlugin, err := bodyfieldtoheader.NewBodyFieldToHeaderPlugin(modelField, bodyfieldtoheader.ModelHeader)
		if err != nil {
			setupLog.Error(err, "Failed to create plugin", "pluginType", bodyfieldtoheader.BodyFieldToHeaderPluginType)
			return err
		}
		r.requestPlugins = append(r.requestPlugins, modelToHeaderPlugin)

		// Create BaseModelToHeaderPlugin instance for extracting the "model" field into X-Gateway-Base-Model-Name
		baseModelToHeaderPlugin, err := basemodelextractor.NewBaseModelToHeaderPlugin(func() *builder.Builder {
			return ctrl.NewControllerManagedBy(mgr)
		}, mgr.GetAPIReader())
		if err != nil {
			setupLog.Error(err, "Failed to create plugin", "pluginType", basemodelextractor.BaseModelToHeaderPluginType)
			return err
		}

		r.requestPlugins = append(r.requestPlugins, baseModelToHeaderPlugin)
	} else {
		setupLog.Info("BBR plugins are specified. Running BBR with the specified plugins.")

		for _, s := range opts.PluginSpecs {
			factory, ok := framework.Registry[s.Type]
			if !ok {
				setupLog.Error(err, fmt.Sprintf("unknown plugin type %q (no factory registered)\n", s.Type))
				return err
			}
			instance, err := factory(s.Name, s.JSON, bbrHandle)
			if err != nil {
				setupLog.Error(err, fmt.Sprintf("invalid %s#%s: %v\n", s.Type, s.Name, err))
				return err
			}
			if requestProcessor, ok := instance.(framework.RequestProcessor); ok {
				r.requestPlugins = append(r.requestPlugins, requestProcessor)
			}
			if responseProcessor, ok := instance.(framework.ResponseProcessor); ok {
				r.responsePlugins = append(r.responsePlugins, responseProcessor)
			}
		}
	}

	// Setup ExtProc Server Runner.
	serverRunner := &runserver.ExtProcServerRunner{
		GrpcPort:        opts.GRPCPort,
		SecureServing:   opts.SecureServing,
		Streaming:       opts.Streaming,
		RequestPlugins:  r.requestPlugins,
		ResponsePlugins: r.responsePlugins,
	}

	// Register health server.
	if err := registerHealthServer(mgr, opts.GRPCHealthPort); err != nil {
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
	framework.Register(bodyfieldtoheader.BodyFieldToHeaderPluginType, bodyfieldtoheader.BodyFieldToHeaderPluginFactory)
	framework.Register(basemodelextractor.BaseModelToHeaderPluginType, basemodelextractor.BaseModelToHeaderPluginFactory)
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
