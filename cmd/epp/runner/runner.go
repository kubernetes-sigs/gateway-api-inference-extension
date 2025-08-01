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

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
	uberzap "go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"google.golang.org/grpc"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics/collectors"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/saturationdetector"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/filter"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/scorer"
	testfilter "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/test/filter"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/server"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	"sigs.k8s.io/gateway-api-inference-extension/version"
)

var (
	grpcPort = flag.Int(
		"grpc-port",
		runserver.DefaultGrpcPort,
		"The gRPC port used for communicating with Envoy proxy")
	grpcHealthPort = flag.Int(
		"grpc-health-port",
		runserver.DefaultGrpcHealthPort,
		"The port used for gRPC liveness and readiness probes")
	metricsPort = flag.Int(
		"metrics-port",
		runserver.DefaultMetricsPort,
		"The metrics port")
	enablePprof = flag.Bool(
		"enable-pprof",
		runserver.DefaultEnablePprof,
		"Enables pprof handlers. Defaults to true. Set to false to disable pprof handlers.")
	poolName = flag.String(
		"pool-name",
		runserver.DefaultPoolName,
		"Name of the InferencePool this Endpoint Picker is associated with.")
	poolGroup = flag.String(
		"pool-group",
		runserver.DefaultPoolGroup,
		"group of the InferencePool this Endpoint Picker is associated with.")
	poolNamespace = flag.String(
		"pool-namespace",
		runserver.DefaultPoolNamespace,
		"Namespace of the InferencePool this Endpoint Picker is associated with.")
	logVerbosity = flag.Int(
		"v",
		logging.DEFAULT,
		"number for the log level verbosity")
	secureServing = flag.Bool(
		"secure-serving",
		runserver.DefaultSecureServing,
		"Enables secure serving. Defaults to true.")
	healthChecking = flag.Bool(
		"health-checking",
		runserver.DefaultHealthChecking,
		"Enables health checking")
	certPath = flag.String(
		"cert-path",
		runserver.DefaultCertPath,
		"The path to the certificate for secure serving. The certificate and private key files "+
			"are assumed to be named tls.crt and tls.key, respectively. If not set, and secureServing is enabled, "+
			"then a self-signed certificate is used.")
	// header/metadata flags
	destinationEndpointHintKey = flag.String(
		"destination-endpoint-hint-key",
		runserver.DefaultDestinationEndpointHintKey,
		"Header and response metadata key used by Envoy to route to the appropriate pod. This must match Envoy configuration.")
	destinationEndpointHintMetadataNamespace = flag.String(
		"destination-endpoint-hint-metadata-namespace",
		runserver.DefaultDestinationEndpointHintMetadataNamespace,
		"The key for the outer namespace struct in the metadata field of the extproc response that is used to wrap the"+
			"target endpoint. If not set, then an outer namespace struct should not be created.")
	fairnessIDHeaderKey = flag.String(
		"fairness-id-header-key",
		runserver.DefaultFairnessIDHeaderKey,
		"The header key used to pass the fairness ID to be used in Flow Control.")
	// metric flags
	totalQueuedRequestsMetric = flag.String(
		"total-queued-requests-metric",
		runserver.DefaultTotalQueuedRequestsMetric,
		"Prometheus metric for the number of queued requests.")
	kvCacheUsagePercentageMetric = flag.String(
		"kv-cache-usage-percentage-metric",
		runserver.DefaultKvCacheUsagePercentageMetric,
		"Prometheus metric for the fraction of KV-cache blocks currently in use (from 0 to 1).")
	// LoRA metrics
	loraInfoMetric = flag.String(
		"lora-info-metric",
		runserver.DefaultLoraInfoMetric,
		"Prometheus metric for the LoRA info metrics (must be in vLLM label format).")

	// metrics related flags
	refreshMetricsInterval = flag.Duration(
		"refresh-metrics-interval",
		runserver.DefaultRefreshMetricsInterval,
		"interval to refresh metrics")
	refreshPrometheusMetricsInterval = flag.Duration(
		"refresh-prometheus-metrics-interval",
		runserver.DefaultRefreshPrometheusMetricsInterval,
		"interval to flush prometheus metrics")
	metricsStalenessThreshold = flag.Duration("metrics-staleness-threshold",
		runserver.DefaultMetricsStalenessThreshold,
		"Duration after which metrics are considered stale. This is used to determine if a pod's metrics are fresh enough.")
	// configuration flags
	configFile = flag.String(
		"config-file",
		runserver.DefaultConfigFile,
		"The path to the configuration file")
	configText = flag.String(
		"config-text",
		runserver.DefaultConfigText,
		"The configuration specified as text, in lieu of a file")

	modelServerMetricsPort = flag.Int("model-server-metrics-port", 0, "Port to scrape metrics from pods. "+
		"Default value will be set to InferencePool.Spec.TargetPortNumber if not set.")
	modelServerMetricsPath                    = flag.String("model-server-metrics-path", "/metrics", "Path to scrape metrics from pods")
	modelServerMetricsScheme                  = flag.String("model-server-metrics-scheme", "http", "Scheme to scrape metrics from pods")
	modelServerMetricsHttpsInsecureSkipVerify = flag.Bool("model-server-metrics-https-insecure-skip-verify", true, "When using 'https' scheme for 'model-server-metrics-scheme', configure 'InsecureSkipVerify' (default to true)")

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

func bindEnvToFlags() {
	// map[ENV_VAR]flagName   – add more as needed
	for env, flg := range map[string]string{
		"GRPC_PORT":                                       "grpc-port",
		"GRPC_HEALTH_PORT":                                "grpc-health-port",
		"MODEL_SERVER_METRICS_PORT":                       "model-server-metrics-port",
		"MODEL_SERVER_METRICS_PATH":                       "model-server-metrics-path",
		"MODEL_SERVER_METRICS_SCHEME":                     "model-server-metrics-scheme",
		"MODEL_SERVER_METRICS_HTTPS_INSECURE_SKIP_VERIFY": "model-server-metrics-https-insecure-skip-verify",
		"DESTINATION_ENDPOINT_HINT_KEY":                   "destination-endpoint-hint-key",
		"POOL_NAME":                                       "pool-name",
		"POOL_NAMESPACE":                                  "pool-namespace",
		// durations & bools work too; flag.Set expects the *string* form
		"REFRESH_METRICS_INTERVAL": "refresh-metrics-interval",
		"SECURE_SERVING":           "secure-serving",
	} {
		if v := os.Getenv(env); v != "" {
			// ignore error; Parse() will catch invalid values later
			_ = flag.Set(flg, v)
		}
	}
}

func (r *Runner) Run(ctx context.Context) error {
	// Defaults already baked into flag declarations
	// Load env vars as "soft" overrides
	bindEnvToFlags()

	opts := zap.Options{
		Development: true,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()
	initLogging(&opts)

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
	mapping, err := backendmetrics.NewMetricMapping(
		*totalQueuedRequestsMetric,
		*kvCacheUsagePercentageMetric,
		*loraInfoMetric,
	)
	if err != nil {
		setupLog.Error(err, "Failed to create metric mapping from flags.")
		return err
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
		ModelServerMetricsPort:   int32(*modelServerMetricsPort),
		ModelServerMetricsPath:   *modelServerMetricsPath,
		ModelServerMetricsScheme: *modelServerMetricsScheme,
		Client:                   metricsHttpClient,
	},
		*refreshMetricsInterval, *metricsStalenessThreshold)

	datastore := datastore.NewDatastore(ctx, pmf)

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
		BindAddress:    fmt.Sprintf(":%d", *metricsPort),
		FilterProvider: filters.WithAuthenticationAndAuthorization,
	}

	poolNamespacedName := types.NamespacedName{
		Name:      *poolName,
		Namespace: *poolNamespace,
	}
	poolGroupKind := schema.GroupKind{
		Group: *poolGroup,
		Kind:  "InferencePool",
	}
	poolGKNN := common.GKNN{
		NamespacedName: poolNamespacedName,
		GroupKind:      poolGroupKind,
	}
	mgr, err := runserver.NewDefaultManager(poolGKNN, cfg, metricsServerOptions)
	if err != nil {
		setupLog.Error(err, "Failed to create controller manager")
		return err
	}

	if *enablePprof {
		setupLog.Info("Enabling pprof handlers")
		err = setupPprofHandlers(mgr)
		if err != nil {
			setupLog.Error(err, "Failed to setup pprof handlers")
			return err
		}
	}

	err = r.parsePluginsConfiguration(ctx)
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

	saturationDetector := saturationdetector.NewDetector(sdConfig, datastore, setupLog)

	director := requestcontrol.NewDirectorWithConfig(datastore, scheduler, saturationDetector, r.requestControlConfig)

	// --- Setup ExtProc Server Runner ---
	serverRunner := &runserver.ExtProcServerRunner{
		GrpcPort:                                 *grpcPort,
		DestinationEndpointHintMetadataNamespace: *destinationEndpointHintMetadataNamespace,
		DestinationEndpointHintKey:               *destinationEndpointHintKey,
		FairnessIDHeaderKey:                      *fairnessIDHeaderKey,
		PoolNamespacedName:                       poolNamespacedName,
		PoolGKNN:                                 poolGKNN,
		Datastore:                                datastore,
		SecureServing:                            *secureServing,
		HealthChecking:                           *healthChecking,
		CertPath:                                 *certPath,
		RefreshPrometheusMetricsInterval:         *refreshPrometheusMetricsInterval,
		MetricsStalenessThreshold:                *metricsStalenessThreshold,
		Director:                                 director,
		SaturationDetector:                       saturationDetector,
	}
	if err := serverRunner.SetupWithManager(ctx, mgr); err != nil {
		setupLog.Error(err, "Failed to setup EPP controllers")
		return err
	}

	// --- Add Runnables to Manager ---
	// Register health server.
	if err := registerHealthServer(mgr, ctrl.Log.WithName("health"), datastore, *grpcHealthPort); err != nil {
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
	plugins.Register(filter.DecisionTreeFilterType, filter.DecisionTreeFilterFactory)
	plugins.Register(filter.LeastKVCacheFilterType, filter.LeastKVCacheFilterFactory)
	plugins.Register(filter.LeastQueueFilterType, filter.LeastQueueFilterFactory)
	plugins.Register(filter.LoraAffinityFilterType, filter.LoraAffinityFilterFactory)
	plugins.Register(filter.LowQueueFilterType, filter.LowQueueFilterFactory)
	plugins.Register(prefix.PrefixCachePluginType, prefix.PrefixCachePluginFactory)
	plugins.Register(picker.MaxScorePickerType, picker.MaxScorePickerFactory)
	plugins.Register(picker.RandomPickerType, picker.RandomPickerFactory)
	plugins.Register(profile.SingleProfileHandlerType, profile.SingleProfileHandlerFactory)
	plugins.Register(scorer.KvCacheUtilizationScorerType, scorer.KvCacheUtilizationScorerFactory)
	plugins.Register(scorer.QueueScorerType, scorer.QueueScorerFactory)
	plugins.Register(scorer.LoraAffinityScorerType, scorer.LoraAffinityScorerFactory)
	// register filter for test purpose only (used in conformance tests)
	plugins.Register(testfilter.HeaderBasedTestingFilterType, testfilter.HeaderBasedTestingFilterFactory)
}

func (r *Runner) parsePluginsConfiguration(ctx context.Context) error {
	if *configText == "" && *configFile == "" {
		return nil // configuring through code, not through file
	}

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
	handle := plugins.NewEppHandle(ctx)
	config, err := loader.LoadConfig(configBytes, handle)
	if err != nil {
		return fmt.Errorf("failed to load the configuration - %w", err)
	}

	r.schedulerConfig, err = loader.LoadSchedulerConfig(config.SchedulingProfiles, handle)
	if err != nil {
		return fmt.Errorf("failed to create Scheduler configuration - %w", err)
	}

	// Add requestControl plugins
	r.requestControlConfig.AddPlugins(handle.GetAllPlugins()...)

	log.FromContext(ctx).Info("loaded configuration from file/text successfully")
	return nil
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
func registerHealthServer(mgr manager.Manager, logger logr.Logger, ds datastore.Datastore, port int) error {
	srv := grpc.NewServer()
	healthPb.RegisterHealthServer(srv, &healthServer{
		logger:    logger,
		datastore: ds,
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
