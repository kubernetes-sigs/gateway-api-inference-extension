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
	"flag"
	"fmt"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
	uberzap "go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"google.golang.org/grpc"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	"sigs.k8s.io/gateway-api-inference-extension/api/config/v1alpha1"
	conformance_epp "sigs.k8s.io/gateway-api-inference-extension/conformance/testing-epp"
	"sigs.k8s.io/gateway-api-inference-extension/internal/runnable"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/common/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics/collectors"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/saturationdetector"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/scorer"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/server"
	envutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/env"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

var (
	grpcPort = flag.Int(
		"grpcPort",
		runserver.DefaultGrpcPort,
		"The gRPC port used for communicating with Envoy proxy")
	grpcHealthPort = flag.Int(
		"grpcHealthPort",
		9003,
		"The port used for gRPC liveness and readiness probes")
	metricsPort = flag.Int(
		"metricsPort", 9090, "The metrics port")
	destinationEndpointHintKey = flag.String(
		"destinationEndpointHintKey",
		runserver.DefaultDestinationEndpointHintKey,
		"Header and response metadata key used by Envoy to route to the appropriate pod. This must match Envoy configuration.")
	destinationEndpointHintMetadataNamespace = flag.String(
		"DestinationEndpointHintMetadataNamespace",
		runserver.DefaultDestinationEndpointHintMetadataNamespace,
		"The key for the outer namespace struct in the metadata field of the extproc response that is used to wrap the"+
			"target endpoint. If not set, then an outer namespace struct should not be created.")
	poolName = flag.String(
		"poolName",
		runserver.DefaultPoolName,
		"Name of the InferencePool this Endpoint Picker is associated with.")
	poolNamespace = flag.String(
		"poolNamespace",
		runserver.DefaultPoolNamespace,
		"Namespace of the InferencePool this Endpoint Picker is associated with.")
	refreshMetricsInterval = flag.Duration(
		"refreshMetricsInterval",
		runserver.DefaultRefreshMetricsInterval,
		"interval to refresh metrics")
	refreshPrometheusMetricsInterval = flag.Duration(
		"refreshPrometheusMetricsInterval",
		runserver.DefaultRefreshPrometheusMetricsInterval,
		"interval to flush prometheus metrics")
	logVerbosity  = flag.Int("v", logging.DEFAULT, "number for the log level verbosity")
	secureServing = flag.Bool(
		"secureServing", runserver.DefaultSecureServing, "Enables secure serving. Defaults to true.")
	certPath = flag.String(
		"certPath", "", "The path to the certificate for secure serving. The certificate and private key files "+
			"are assumed to be named tls.crt and tls.key, respectively. If not set, and secureServing is enabled, "+
			"then a self-signed certificate is used.")
	// metric flags
	totalQueuedRequestsMetric = flag.String("totalQueuedRequestsMetric",
		"vllm:num_requests_waiting",
		"Prometheus metric for the number of queued requests.")
	kvCacheUsagePercentageMetric = flag.String("kvCacheUsagePercentageMetric",
		"vllm:gpu_cache_usage_perc",
		"Prometheus metric for the fraction of KV-cache blocks currently in use (from 0 to 1).")
	// LoRA metrics
	loraInfoMetric = flag.String("loraInfoMetric",
		"vllm:lora_requests_info",
		"Prometheus metric for the LoRA info metrics (must be in vLLM label format).")
	// configuration flags
	configFile = flag.String("configFile", "", "The path to the configuration file")
	configText = flag.String("configText", "", "The configuration specified as text, in lieu of a file")

	setupLog = ctrl.Log.WithName("setup")

	// Environment variables
	schedulerV2                       = envutil.GetEnvBool("EXPERIMENTAL_USE_SCHEDULER_V2", false, setupLog)
	prefixCacheScheduling             = envutil.GetEnvBool("ENABLE_PREFIX_CACHE_SCHEDULING", false, setupLog)
	reqHeaderBasedSchedulerForTesting = envutil.GetEnvBool("ENABLE_REQ_HEADER_BASED_SCHEDULER_FOR_TESTING", false, setupLog)
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

func (r *Runner) Run() error {
	opts := zap.Options{
		Development: true,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()
	initLogging(&opts)

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
	pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.PodMetricsClientImpl{MetricMapping: mapping}, *refreshMetricsInterval)
	ctx := ctrl.SetupSignalHandler()
	datastore := datastore.NewDatastore(ctx, pmf)

	// --- Setup Metrics Server ---
	customCollectors := []prometheus.Collector{collectors.NewInferencePoolMetricsCollector(datastore)}
	metrics.Register(customCollectors...)
	metrics.RecordInferenceExtensionInfo()
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
	mgr, err := runserver.NewDefaultManager(poolNamespacedName, cfg, metricsServerOptions)
	if err != nil {
		setupLog.Error(err, "Failed to create controller manager")
		return err
	}

	var theConfig *v1alpha1.EndpointPickerConfig
	var instantiatedPlugins map[string]plugins.Plugin

	if len(*configText) != 0 || len(*configFile) != 0 {
		theConfig, err = config.LoadConfig([]byte(*configText), *configFile, setupLog)
		if err != nil {
			setupLog.Error(err, "Failed to load the configuration")
			return err
		}

		instantiatedPlugins, err = config.LoadPluginReferences(theConfig, setupLog)
		if err != nil {
			setupLog.Error(err, "Failed to instantiate the plugins")
			return err
		}

		r.schedulerConfig, err = scheduling.LoadSchedulerConfig(theConfig.SchedulingProfiles, instantiatedPlugins, setupLog)
		if err != nil {
			setupLog.Error(err, "Failed to create Scheduler configuration")
			return err
		}
	}

	// --- Initialize Core EPP Components ---
	scheduler, err := r.initializeScheduler(datastore)
	if err != nil {
		setupLog.Error(err, "Failed to create scheduler")
		return err
	}

	saturationDetector := saturationdetector.NewDetector(sdConfig, datastore, ctrl.Log)

	director := requestcontrol.NewDirectorWithConfig(datastore, scheduler, saturationDetector, r.requestControlConfig)

	// --- Setup ExtProc Server Runner ---
	serverRunner := &runserver.ExtProcServerRunner{
		GrpcPort:                                 *grpcPort,
		DestinationEndpointHintMetadataNamespace: *destinationEndpointHintMetadataNamespace,
		DestinationEndpointHintKey:               *destinationEndpointHintKey,
		PoolNamespacedName:                       poolNamespacedName,
		Datastore:                                datastore,
		SecureServing:                            *secureServing,
		CertPath:                                 *certPath,
		RefreshPrometheusMetricsInterval:         *refreshPrometheusMetricsInterval,
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

func (r *Runner) initializeScheduler(datastore datastore.Datastore) (*scheduling.Scheduler, error) {
	if r.schedulerConfig != nil {
		return scheduling.NewSchedulerWithConfig(datastore, r.schedulerConfig), nil
	}

	// otherwise, no one configured from outside scheduler config. use existing configuration
	scheduler := scheduling.NewScheduler(datastore)
	if schedulerV2 {
		queueScorerWeight := envutil.GetEnvInt("QUEUE_SCORE_WEIGHT", scorer.DefaultQueueScorerWeight, setupLog)
		kvCacheScorerWeight := envutil.GetEnvInt("KV_CACHE_SCORE_WEIGHT", scorer.DefaultKVCacheScorerWeight, setupLog)

		schedulerProfile := framework.NewSchedulerProfile().
			WithScorers(framework.NewWeightedScorer(scorer.NewQueueScorer(), queueScorerWeight),
				framework.NewWeightedScorer(scorer.NewKVCacheScorer(), kvCacheScorerWeight)).
			WithPicker(picker.NewMaxScorePicker())

		if prefixCacheScheduling {
			prefixScorerWeight := envutil.GetEnvInt("PREFIX_CACHE_SCORE_WEIGHT", prefix.DefaultScorerWeight, setupLog)
			if err := schedulerProfile.AddPlugins(framework.NewWeightedScorer(prefix.New(loadPrefixCacheConfig()), prefixScorerWeight)); err != nil {
				return nil, fmt.Errorf("Failed to register scheduler plugins - %w", err)
			}
		}

		schedulerConfig := scheduling.NewSchedulerConfig(profile.NewSingleProfileHandler(), map[string]*framework.SchedulerProfile{"schedulerv2": schedulerProfile})
		scheduler = scheduling.NewSchedulerWithConfig(datastore, schedulerConfig)
	}

	if reqHeaderBasedSchedulerForTesting {
		scheduler = conformance_epp.NewReqHeaderBasedScheduler(datastore)
	}

	return scheduler, nil
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

func loadPrefixCacheConfig() prefix.Config {
	baseLogger := log.Log.WithName("env-config")

	return prefix.Config{
		HashBlockSize:          envutil.GetEnvInt("PREFIX_CACHE_HASH_BLOCK_SIZE", prefix.DefaultHashBlockSize, baseLogger),
		MaxPrefixBlocksToMatch: envutil.GetEnvInt("PREFIX_CACHE_MAX_PREFIX_BLOCKS", prefix.DefaultMaxPrefixBlocks, baseLogger),
		LRUIndexerCapacity:     envutil.GetEnvInt("PREFIX_CACHE_LRU_CAPACITY", prefix.DefaultLRUIndexerCapacity, baseLogger),
	}
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
