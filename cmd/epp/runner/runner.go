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
	"crypto/tls"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"net/http/pprof"
	"os"
	"runtime"
	"sync/atomic"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
	uberzap "go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"google.golang.org/grpc"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	"sigs.k8s.io/gateway-api-inference-extension/internal/runnable"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/config/loader"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	dlmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol"
	fccontroller "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/controller"
	fcregistry "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/registry"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics/collectors"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/saturationdetector"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/scorer"
	testfilter "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/test/filter"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/server"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/env"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	"sigs.k8s.io/gateway-api-inference-extension/version"
)

const (
	// enableExperimentalDatalayerV2 defines the environment variable used as feature flag for the pluggable data layer.
	enableExperimentalDatalayerV2 = "ENABLE_EXPERIMENTAL_DATALAYER_V2"
	// enableExperimentalFlowControlLayer defines the environment variable used as a feature flag for the pluggable flow
	// control layer.
	enableExperimentalFlowControlLayer = "ENABLE_EXPERIMENTAL_FLOW_CONTROL_LAYER"
)

// TODO: this is hardcoded for POC only. This needs to be hooked up to our text-based config story.
var flowControlConfig = flowcontrol.Config{
	Controller: fccontroller.Config{}, // Use all defaults.
	Registry: fcregistry.Config{
		// Define domain of accepted priority levels as this field is required. Use defaults for all optional fields.
		// TODO: this should not be hardcoded.
		PriorityBands: []fcregistry.PriorityBandConfig{
			{Priority: 0, PriorityName: "Default"},
		},
	},
}

var (
	grpcPort            = flag.Int("grpc-port", runserver.DefaultGrpcPort, "The gRPC port used for communicating with Envoy proxy")
	grpcHealthPort      = flag.Int("grpc-health-port", runserver.DefaultGrpcHealthPort, "The port used for gRPC liveness and readiness probes")
	metricsPort         = flag.Int("metrics-port", runserver.DefaultMetricsPort, "The metrics port")
	metricsEndpointAuth = flag.Bool("metrics-endpoint-auth", true, "Enables authentication and authorization of the metrics endpoint")
	enablePprof         = flag.Bool("enable-pprof", runserver.DefaultEnablePprof, "Enables pprof handlers. Defaults to true. Set to false to disable pprof handlers.")
	poolName            = flag.String("pool-name", runserver.DefaultPoolName, "Name of the InferencePool this Endpoint Picker is associated with.")
	poolNamespace       = flag.String("pool-namespace", "", "Namespace of the InferencePool this Endpoint Picker is associated with.")
	logVerbosity        = flag.Int("v", logging.DEFAULT, "number for the log level verbosity")
	secureServing       = flag.Bool("secure-serving", runserver.DefaultSecureServing, "Enables secure serving. Defaults to true.")
	healthChecking      = flag.Bool("health-checking", runserver.DefaultHealthChecking, "Enables health checking")
	certPath            = flag.String("cert-path", runserver.DefaultCertPath, "The path to the certificate for secure serving. The certificate and private key files "+
		"are assumed to be named tls.crt and tls.key, respectively. If not set, and secureServing is enabled, "+
		"then a self-signed certificate is used.")
	// metric flags
	totalQueuedRequestsMetric    = flag.String("total-queued-requests-metric", runserver.DefaultTotalQueuedRequestsMetric, "Prometheus metric for the number of queued requests.")
	kvCacheUsagePercentageMetric = flag.String("kv-cache-usage-percentage-metric", runserver.DefaultKvCacheUsagePercentageMetric, "Prometheus metric for the fraction of KV-cache blocks currently in use (from 0 to 1).")
	// LoRA metrics
	loraInfoMetric = flag.String("lora-info-metric", runserver.DefaultLoraInfoMetric, "Prometheus metric for the LoRA info metrics (must be in vLLM label format).")
	// Cache info  metrics
	cacheInfoMetric = flag.String("cache-info-metric", runserver.DefaultCacheInfoMetric, "Prometheus metric for the cache info metrics.")
	// metrics related flags
	refreshMetricsInterval           = flag.Duration("refresh-metrics-interval", runserver.DefaultRefreshMetricsInterval, "interval to refresh metrics")
	refreshPrometheusMetricsInterval = flag.Duration("refresh-prometheus-metrics-interval", runserver.DefaultRefreshPrometheusMetricsInterval, "interval to flush prometheus metrics")
	metricsStalenessThreshold        = flag.Duration("metrics-staleness-threshold", runserver.DefaultMetricsStalenessThreshold, "Duration after which metrics are considered stale. This is used to determine if a pod's metrics are fresh enough.")
	// configuration flags
	configFile = flag.String("config-file", runserver.DefaultConfigFile, "The path to the configuration file")
	configText = flag.String("config-text", runserver.DefaultConfigText, "The configuration specified as text, in lieu of a file")

	modelServerMetricsPort = flag.Int("model-server-metrics-port", 0, "Port to scrape metrics from pods. "+
		"Default value will be set to the InferencePool.Spec.TargetPorts[0].Number if not set.")
	modelServerMetricsPath                    = flag.String("model-server-metrics-path", "/metrics", "Path to scrape metrics from pods")
	modelServerMetricsScheme                  = flag.String("model-server-metrics-scheme", "http", "Scheme to scrape metrics from pods")
	modelServerMetricsHttpsInsecureSkipVerify = flag.Bool("model-server-metrics-https-insecure-skip-verify", true, "When using 'https' scheme for 'model-server-metrics-scheme', configure 'InsecureSkipVerify' (default to true)")
	haEnableLeaderElection                    = flag.Bool("ha-enable-leader-election", false, "Enables leader election for high availability. When enabled, readiness probes will only pass on the leader.")
	tracing                                   = flag.Bool("tracing", true, "Enables emitting traces")

	setupLog = ctrl.Log.WithName("setup")
)

// NewRunner initializes a new EPP Runner and returns its pointer.
func NewRunner() *Runner {
	return &Runner{
		requestControlConfig: requestcontrol.NewConfig(), // default requestcontrol config has empty plugin list
	}
}

// Runner is used to run epp with its plugins
type Runner struct {
	requestControlConfig *requestcontrol.Config
	schedulerConfig      *scheduling.SchedulerConfig
}

func (r *Runner) WithRequestControlConfig(requestControlConfig *requestcontrol.Config) *Runner {
	r.requestControlConfig = requestControlConfig
	return r
}

func (r *Runner) WithSchedulerConfig(schedulerConfig *scheduling.SchedulerConfig) *Runner {
	r.schedulerConfig = schedulerConfig
	return r
}

func (r *Runner) Run(ctx context.Context) error {
	opts := zap.Options{
		Development: true,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()
	initLogging(&opts)

	if *tracing {
		err := common.InitTracing(ctx, setupLog)
		if err != nil {
			return err
		}
	}

	setupLog.Info("GIE build", "commit-sha", version.CommitSHA, "build-ref", version.BuildRef)

	// Validate flags
	if err := validateFlags(); err != nil {
		setupLog.Error(err, "Failed to validate flags")
		return err
	}

	// Print all flag values
	flags := make(map[string]any)
	flag.VisitAll(func(f *flag.Flag) {
		flags[f.Name] = f.Value
	})
	setupLog.Info("Flags processed", "flags", flags)

	// --- Load Configurations from Environment Variables ---
	sdConfig := saturationdetector.LoadConfigFromEnv()

	// --- Get Kubernetes Config ---
	cfg, err := ctrl.GetConfig()
	if err != nil {
		setupLog.Error(err, "Failed to get Kubernetes rest config")
		return err
	}

	// --- Setup Datastore ---
	useDatalayerV2 := env.GetEnvBool(enableExperimentalDatalayerV2, false, setupLog)
	epf, err := r.setupMetricsCollection(setupLog, useDatalayerV2)
	if err != nil {
		return err
	}
	datastore := datastore.NewDatastore(ctx, epf, int32(*modelServerMetricsPort))

	// --- Setup Metrics Server ---
	customCollectors := []prometheus.Collector{collectors.NewInferencePoolMetricsCollector(datastore)}
	metrics.Register(customCollectors...)
	metrics.RecordInferenceExtensionInfo(version.CommitSHA, version.BuildRef)
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

	// Determine pool namespace: if --pool-namespace is non-empty, use it; else NAMESPACE env var; else default
	resolvePoolNamespace := func() string {
		if *poolNamespace != "" {
			return *poolNamespace
		}
		if nsEnv := os.Getenv("NAMESPACE"); nsEnv != "" {
			return nsEnv
		}
		return runserver.DefaultPoolNamespace
	}
	resolvedPoolNamespace := resolvePoolNamespace()
	poolNamespacedName := types.NamespacedName{
		Name:      *poolName,
		Namespace: resolvedPoolNamespace,
	}
	poolGKNN := common.GKNN{
		NamespacedName: poolNamespacedName,
	}

	isLeader := &atomic.Bool{}
	isLeader.Store(false)

	mgr, err := runserver.NewDefaultManager(poolGKNN, cfg, metricsServerOptions, *haEnableLeaderElection)
	if err != nil {
		setupLog.Error(err, "Failed to create controller manager")
		return err
	}

	if *haEnableLeaderElection {
		setupLog.Info("Leader election enabled")
		go func() {
			<-mgr.Elected()
			isLeader.Store(true)
			setupLog.Info("This instance is now the leader!")
		}()
	} else {
		// If leader election is disabled, all instances are "leaders" for readiness purposes.
		isLeader.Store(true)
	}

	if *enablePprof {
		setupLog.Info("Enabling pprof handlers")
		err = setupPprofHandlers(mgr)
		if err != nil {
			setupLog.Error(err, "Failed to setup pprof handlers")
			return err
		}
		runtime.SetMutexProfileFraction(1)
		runtime.SetBlockProfileRate(1)
	}

	err = r.parsePluginsConfiguration(ctx, datastore)
	if err != nil {
		setupLog.Error(err, "Failed to parse plugins configuration")
		return err
	}

	// --- Initialize Core EPP Components ---
	if r.schedulerConfig == nil {
		err := errors.New("scheduler config must be set either by config api or through code")
		setupLog.Error(err, "failed to create scheduler")
		return err
	}

	setupLog.Info("parsed config", "scheduler-config", r.schedulerConfig)

	scheduler := scheduling.NewSchedulerWithConfig(r.schedulerConfig)

	saturationDetector := saturationdetector.NewDetector(sdConfig, setupLog)

	// --- Admission Control Initialization ---
	enableFlowControl := env.GetEnvBool(enableExperimentalFlowControlLayer, false, setupLog)
	var admissionController requestcontrol.AdmissionController
	if enableFlowControl {
		setupLog.Info("Initializing experimental Flow Control layer")
		fcCfg, err := flowControlConfig.ValidateAndApplyDefaults()
		if err != nil {
			setupLog.Error(err, "failed to initialize Flow Control layer")
			return fmt.Errorf("invalid Flow Control config: %w", err)
		}

		registry, err := fcregistry.NewFlowRegistry(fcCfg.Registry, setupLog)
		if err != nil {
			return fmt.Errorf("failed to initialize Flow Registry: %w", err)
		}
		fc, err := fccontroller.NewFlowController(
			ctx,
			fcCfg.Controller,
			registry,
			saturationDetector,
			setupLog,
		)
		if err != nil {
			return fmt.Errorf("failed to initialize Flow Controller: %w", err)
		}
		go registry.Run(ctx)
		admissionController = requestcontrol.NewFlowControlAdmissionController(saturationDetector, fc)
	} else {
		setupLog.Info("Experimental Flow Control layer is disabled, using legacy admission control")
		admissionController = requestcontrol.NewLegacyAdmissionController(saturationDetector)
	}

	director := requestcontrol.NewDirectorWithConfig(
		datastore,
		scheduler,
		admissionController,
		r.requestControlConfig)

	// --- Setup ExtProc Server Runner ---
	serverRunner := &runserver.ExtProcServerRunner{
		GrpcPort:                         *grpcPort,
		PoolNamespacedName:               poolNamespacedName,
		PoolGKNN:                         poolGKNN,
		Datastore:                        datastore,
		SecureServing:                    *secureServing,
		HealthChecking:                   *healthChecking,
		CertPath:                         *certPath,
		RefreshPrometheusMetricsInterval: *refreshPrometheusMetricsInterval,
		MetricsStalenessThreshold:        *metricsStalenessThreshold,
		Director:                         director,
		SaturationDetector:               saturationDetector,
		UseExperimentalDatalayerV2:       useDatalayerV2, // pluggable data layer feature flag
	}
	if err := serverRunner.SetupWithManager(ctx, mgr); err != nil {
		setupLog.Error(err, "Failed to setup EPP controllers")
		return err
	}

	// --- Add Runnables to Manager ---
	// Register health server.
	if err := registerHealthServer(mgr, ctrl.Log.WithName("health"), datastore, *grpcHealthPort, isLeader, *haEnableLeaderElection); err != nil {
		return err
	}

	// Register ext-proc server.
	if err := registerExtProcServer(mgr, serverRunner, ctrl.Log.WithName("ext-proc")); err != nil {
		return err
	}

	// --- Start Manager ---
	// This blocks until a signal is received.
	setupLog.Info("Controller manager starting")
	if err := mgr.Start(ctx); err != nil {
		setupLog.Error(err, "Error starting controller manager")
		return err
	}
	setupLog.Info("Controller manager terminated")
	return nil
}

// registerInTreePlugins registers the factory functions of all known plugins
func (r *Runner) registerInTreePlugins() {
	plugins.Register(prefix.PrefixCachePluginType, prefix.PrefixCachePluginFactory)
	plugins.Register(picker.MaxScorePickerType, picker.MaxScorePickerFactory)
	plugins.Register(picker.RandomPickerType, picker.RandomPickerFactory)
	plugins.Register(picker.WeightedRandomPickerType, picker.WeightedRandomPickerFactory)
	plugins.Register(profile.SingleProfileHandlerType, profile.SingleProfileHandlerFactory)
	plugins.Register(scorer.KvCacheUtilizationScorerType, scorer.KvCacheUtilizationScorerFactory)
	plugins.Register(scorer.QueueScorerType, scorer.QueueScorerFactory)
	plugins.Register(scorer.LoraAffinityScorerType, scorer.LoraAffinityScorerFactory)
	// register filter for test purpose only (used in conformance tests)
	plugins.Register(testfilter.HeaderBasedTestingFilterType, testfilter.HeaderBasedTestingFilterFactory)
}

func (r *Runner) parsePluginsConfiguration(ctx context.Context, ds datastore.Datastore) error {
	if *configText == "" && *configFile == "" {
		return nil // configuring through code, not through file
	}

	logger := log.FromContext(ctx)

	var configBytes []byte
	if *configText != "" {
		configBytes = []byte(*configText)
	} else if *configFile != "" { // if config was specified through a file
		var err error
		configBytes, err = os.ReadFile(*configFile)
		if err != nil {
			return fmt.Errorf("failed to load config from a file '%s' - %w", *configFile, err)
		}
	}

	r.registerInTreePlugins()
	handle := plugins.NewEppHandle(ctx, ds.PodList)
	config, err := loader.LoadConfig(configBytes, handle, logger)

	if err != nil {
		return fmt.Errorf("failed to load the configuration - %w", err)
	}

	r.schedulerConfig = config.SchedulerConfig

	// Add requestControl plugins
	r.requestControlConfig.AddPlugins(handle.GetAllPlugins()...)

	logger.Info("loaded configuration from file/text successfully")
	return nil
}

func (r *Runner) setupMetricsCollection(setupLog logr.Logger, useExperimentalDatalayer bool) (datalayer.EndpointFactory, error) {
	if useExperimentalDatalayer {
		return setupDatalayer()
	}

	if len(datalayer.GetSources()) != 0 {
		setupLog.Info("data sources registered but pluggable datalayer is disabled")
	}
	return setupMetricsV1(setupLog)
}

func setupMetricsV1(setupLog logr.Logger) (datalayer.EndpointFactory, error) {
	mapping, err := backendmetrics.NewMetricMapping(
		*totalQueuedRequestsMetric,
		*kvCacheUsagePercentageMetric,
		*loraInfoMetric,
		*cacheInfoMetric,
	)
	if err != nil {
		setupLog.Error(err, "Failed to create metric mapping from flags.")
		return nil, err
	}
	verifyMetricMapping(*mapping, setupLog)

	var metricsHttpClient *http.Client
	if *modelServerMetricsScheme == "https" {
		metricsHttpClient = &http.Client{
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{
					InsecureSkipVerify: *modelServerMetricsHttpsInsecureSkipVerify,
				},
			},
		}
	} else {
		metricsHttpClient = http.DefaultClient
	}

	pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.PodMetricsClientImpl{
		MetricMapping:            mapping,
		ModelServerMetricsPath:   *modelServerMetricsPath,
		ModelServerMetricsScheme: *modelServerMetricsScheme,
		Client:                   metricsHttpClient,
	},
		*refreshMetricsInterval)
	return pmf, nil
}

func setupDatalayer() (datalayer.EndpointFactory, error) {
	// create and register a metrics data source and extractor. In the future,
	// data sources and extractors might be configured via a file. Once done,
	// this (and registering the sources with the endpoint factory) should
	// be moved accordingly.
	source := dlmetrics.NewDataSource(*modelServerMetricsScheme,
		*modelServerMetricsPath,
		*modelServerMetricsHttpsInsecureSkipVerify,
		nil)
	extractor, err := dlmetrics.NewExtractor(*totalQueuedRequestsMetric,
		*kvCacheUsagePercentageMetric,
		*loraInfoMetric, *cacheInfoMetric)

	if err != nil {
		return nil, err
	}
	if err := source.AddExtractor(extractor); err != nil {
		return nil, err
	}
	if err := datalayer.RegisterSource(source); err != nil {
		return nil, err
	}

	factory := datalayer.NewEndpointFactory(datalayer.GetSources(), *refreshMetricsInterval)
	return factory, nil
}

func initLogging(opts *zap.Options) {
	// Unless -zap-log-level is explicitly set, use -v
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

	logger := zap.New(zap.UseFlagOptions(opts), zap.RawZapOpts(uberzap.AddCaller()))
	ctrl.SetLogger(logger)
}

// registerExtProcServer adds the ExtProcServerRunner as a Runnable to the manager.
func registerExtProcServer(mgr manager.Manager, runner *runserver.ExtProcServerRunner, logger logr.Logger) error {
	if err := mgr.Add(runner.AsRunnable(logger)); err != nil {
		setupLog.Error(err, "Failed to register ext-proc gRPC server runnable")
		return err
	}
	setupLog.Info("ExtProc server runner added to manager.")
	return nil
}

// registerHealthServer adds the Health gRPC server as a Runnable to the given manager.
func registerHealthServer(mgr manager.Manager, logger logr.Logger, ds datastore.Datastore, port int, isLeader *atomic.Bool, leaderElectionEnabled bool) error {
	srv := grpc.NewServer()
	healthPb.RegisterHealthServer(srv, &healthServer{
		logger:                logger,
		datastore:             ds,
		isLeader:              isLeader,
		leaderElectionEnabled: leaderElectionEnabled,
	})
	if err := mgr.Add(
		runnable.NoLeaderElection(runnable.GRPCServer("health", srv, port))); err != nil {
		setupLog.Error(err, "Failed to register health server")
		return err
	}
	return nil
}

func validateFlags() error {
	if *poolName == "" {
		return fmt.Errorf("required %q flag not set", "poolName")
	}
	if *configText != "" && *configFile != "" {
		return fmt.Errorf("both the %q and %q flags can not be set at the same time", "configText", "configFile")
	}
	if *modelServerMetricsScheme != "http" && *modelServerMetricsScheme != "https" {
		return fmt.Errorf("unexpected %q value for %q flag, it can only be set to 'http' or 'https'", *modelServerMetricsScheme, "model-server-metrics-scheme")
	}

	return nil
}

func verifyMetricMapping(mapping backendmetrics.MetricMapping, logger logr.Logger) {
	if mapping.TotalQueuedRequests == nil {
		logger.Info("Not scraping metric: TotalQueuedRequests")
	}
	if mapping.KVCacheUtilization == nil {
		logger.Info("Not scraping metric: KVCacheUtilization")
	}
	if mapping.LoraRequestInfo == nil {
		logger.Info("Not scraping metric: LoraRequestInfo")
	}
	if mapping.CacheConfigInfo == nil {
		logger.Info("Not scraping metric: CacheConfigInfo")
	}
}

// setupPprofHandlers only implements the pre-defined profiles:
// https://cs.opensource.google/go/go/+/refs/tags/go1.24.4:src/runtime/pprof/pprof.go;l=108
func setupPprofHandlers(mgr ctrl.Manager) error {
	var err error
	profiles := []string{
		"heap",
		"goroutine",
		"allocs",
		"threadcreate",
		"block",
		"mutex",
	}
	for _, p := range profiles {
		err = mgr.AddMetricsServerExtraHandler("/debug/pprof/"+p, pprof.Handler(p))
		if err != nil {
			return err
		}
	}
	return nil
}
