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
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	"sigs.k8s.io/gateway-api-inference-extension/internal/runnable"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/config/loader"
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

// WithExecutableName sets the name of the executable containing the runner.
// The name is used in the version log upon startup and is otherwise opaque.
func (r *Runner) WithExecutableName(exeName string) *Runner {
	r.bbrExecutableName = exeName
	return r
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

	// Load plugins via one of three mutually exclusive paths:
	// 1. --config-file: load from YAML config file
	// 2. --plugin flags: legacy CLI-based plugin instantiation
	// 3. Neither: use default config
	switch {
	case opts.ConfigFile != "":
		setupLog.Info("Loading BBR plugins from config file", "path", opts.ConfigFile)
		configBytes, err := os.ReadFile(opts.ConfigFile)
		if err != nil {
			setupLog.Error(err, "Failed to read config file", "path", opts.ConfigFile)
			return err
		}
		rawConfig, err := loader.LoadRawConfig(configBytes, setupLog)
		if err != nil {
			setupLog.Error(err, "Failed to parse config file")
			return err
		}
		r.requestPlugins, r.responsePlugins, err = loader.InstantiateAndConfigure(rawConfig, bbrHandle)
		if err != nil {
			setupLog.Error(err, "Failed to instantiate plugins from config")
			return err
		}
	case len(opts.PluginSpecs) > 0:
		setupLog.Info("BBR plugins are specified via --plugin flags.")

		for _, s := range opts.PluginSpecs {
			factory, ok := framework.Registry[s.Type]
			if !ok {
				err := fmt.Errorf("unknown plugin type %q (no factory registered)", s.Type)
				setupLog.Error(err, "Failed to find plugin factory")
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
	default:
		setupLog.Info("No plugins specified. Loading default configuration.")
		rawConfig, err := loader.LoadRawConfig(nil, setupLog)
		if err != nil {
			setupLog.Error(err, "Failed to load default config")
			return err
		}
		r.requestPlugins, r.responsePlugins, err = loader.InstantiateAndConfigure(rawConfig, bbrHandle)
		if err != nil {
			setupLog.Error(err, "Failed to instantiate default plugins")
			return err
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
