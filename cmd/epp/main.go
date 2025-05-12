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

package main

import (
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"strconv"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	uberzap "go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"google.golang.org/grpc"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	"k8s.io/component-base/metrics/legacyregistry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	"sigs.k8s.io/gateway-api-inference-extension/internal/runnable"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics/collectors"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/filter"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/scorer"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/server"
	envutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/env"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	defaultMetricsEndpoint = "/metrics"
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

	setupLog = ctrl.Log.WithName("setup")

	// Environment variables
	schedulerV2           = envutil.GetEnvString("EXPERIMENTAL_USE_SCHEDULER_V2", "false", setupLog)
	prefixCacheScheduling = envutil.GetEnvString("ENABLE_PREFIX_CACHE_SCHEDULING", "false", setupLog)
)

func loadPrefixCacheConfig() prefix.Config {
	baseLogger := log.Log.WithName("env-config")

	return prefix.Config{
		HashBlockSize:          envutil.GetEnvInt("PREFIX_CACHE_HASH_BLOCK_SIZE", prefix.DefaultHashBlockSize, baseLogger),
		MaxPrefixBlocksToMatch: envutil.GetEnvInt("PREFIX_CACHE_MAX_PREFIX_BLOCKS", prefix.DefaultMaxPrefixBlocks, baseLogger),
		LRUIndexerCapacity:     envutil.GetEnvInt("PREFIX_CACHE_LRU_CAPACITY", prefix.DefaultLRUIndexerCapacity, baseLogger),
	}
}

func main() {
	if err := run(); err != nil {
		os.Exit(1)
	}
}

func run() error {
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

	// Init runtime.
	cfg, err := ctrl.GetConfig()
	if err != nil {
		setupLog.Error(err, "Failed to get rest config")
		return err
	}

	poolNamespacedName := types.NamespacedName{
		Name:      *poolName,
		Namespace: *poolNamespace,
	}
	mgr, err := runserver.NewDefaultManager(poolNamespacedName, cfg)
	if err != nil {
		setupLog.Error(err, "Failed to create controller manager")
		return err
	}

	// Set up mapper for metric scraping.
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
	// Setup runner.
	ctx := ctrl.SetupSignalHandler()

	datastore := datastore.NewDatastore(ctx, pmf)

	scheduler := scheduling.NewScheduler(datastore)
	if schedulerV2 == "true" {
		queueScorerWeight := envutil.GetEnvInt("QUEUE_SCORE_WEIGHT", scorer.DefaultQueueScorerWeight, setupLog)
		kvCacheScorerWeight := envutil.GetEnvInt("KV_CACHE_SCORE_WEIGHT", scorer.DefaultKVCacheScorerWeight, setupLog)
		scorers := map[plugins.Scorer]int{
			&scorer.QueueScorer{}:   queueScorerWeight,
			&scorer.KVCacheScorer{}: kvCacheScorerWeight,
		}
		schedConfigOpts := []scheduling.ConfigOption{}
		if prefixCacheScheduling == "true" {
			prefixScorerWeight := envutil.GetEnvInt("PREFIX_CACHE_SCORE_WEIGHT", prefix.DefaultScorerWeight, setupLog)
			schedConfigOpts = append(schedConfigOpts, scheduling.AddPrefixPlugin(loadPrefixCacheConfig(), prefixScorerWeight))
		}
		schedulerConfig := scheduling.NewSchedulerConfig(
			[]plugins.PreSchedule{},
			[]plugins.Filter{filter.NewSheddableCapacityFilter()},
			scorers,
			picker.NewMaxScorePicker(),
			[]plugins.PostSchedule{},
			[]plugins.PostResponse{},
			schedConfigOpts...)
		scheduler = scheduling.NewSchedulerWithConfig(datastore, schedulerConfig)
	}
	serverRunner := &runserver.ExtProcServerRunner{
		GrpcPort:                                 *grpcPort,
		DestinationEndpointHintMetadataNamespace: *destinationEndpointHintMetadataNamespace,
		DestinationEndpointHintKey:               *destinationEndpointHintKey,
		PoolNamespacedName:                       poolNamespacedName,
		Datastore:                                datastore,
		SecureServing:                            *secureServing,
		CertPath:                                 *certPath,
		RefreshPrometheusMetricsInterval:         *refreshPrometheusMetricsInterval,
		Scheduler:                                scheduler,
	}
	if err := serverRunner.SetupWithManager(ctx, mgr); err != nil {
		setupLog.Error(err, "Failed to setup ext-proc controllers")
		return err
	}

	// Register health server.
	if err := registerHealthServer(mgr, ctrl.Log.WithName("health"), datastore, *grpcHealthPort); err != nil {
		return err
	}

	// Register ext-proc server.
	if err := mgr.Add(serverRunner.AsRunnable(ctrl.Log.WithName("ext-proc"))); err != nil {
		setupLog.Error(err, "Failed to register ext-proc gRPC server")
		return err
	}

	// Register metrics handler.
	if err := registerMetricsHandler(mgr, *metricsPort, cfg, datastore); err != nil {
		return err
	}

	// Start the manager. This blocks until a signal is received.
	setupLog.Info("Controller manager starting")
	if err := mgr.Start(ctx); err != nil {
		setupLog.Error(err, "Error starting controller manager")
		return err
	}
	setupLog.Info("Controller manager terminated")
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

// registerMetricsHandler adds the metrics HTTP handler as a Runnable to the given manager.
func registerMetricsHandler(mgr manager.Manager, port int, cfg *rest.Config, ds datastore.Datastore) error {
	metrics.Register()
	legacyregistry.CustomMustRegister(collectors.NewInferencePoolMetricsCollector(ds))

	metrics.RecordInferenceExtensionInfo()

	// Init HTTP server.
	h, err := metricsHandlerWithAuthenticationAndAuthorization(cfg)
	if err != nil {
		return err
	}

	mux := http.NewServeMux()
	mux.Handle(defaultMetricsEndpoint, h)

	srv := &http.Server{
		Addr:    net.JoinHostPort("", strconv.Itoa(port)),
		Handler: mux,
	}

	if err := mgr.Add(&manager.Server{
		Name:   "metrics",
		Server: srv,
	}); err != nil {
		setupLog.Error(err, "Failed to register metrics HTTP handler")
		return err
	}
	return nil
}

func metricsHandlerWithAuthenticationAndAuthorization(cfg *rest.Config) (http.Handler, error) {
	h := promhttp.HandlerFor(
		legacyregistry.DefaultGatherer,
		promhttp.HandlerOpts{},
	)
	httpClient, err := rest.HTTPClientFor(cfg)
	if err != nil {
		setupLog.Error(err, "Failed to create http client for metrics auth")
		return nil, err
	}

	filter, err := filters.WithAuthenticationAndAuthorization(cfg, httpClient)
	if err != nil {
		setupLog.Error(err, "Failed to create metrics filter for auth")
		return nil, err
	}
	metricsLogger := ctrl.Log.WithName("metrics").WithValues("path", defaultMetricsEndpoint)
	metricsAuthHandler, err := filter(metricsLogger, h)
	if err != nil {
		setupLog.Error(err, "Failed to create metrics auth handler")
		return nil, err
	}
	return metricsAuthHandler, nil
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
