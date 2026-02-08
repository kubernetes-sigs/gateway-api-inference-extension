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
	"regexp"
	"runtime"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/pflag"
	uberzap "go.uber.org/zap"
	"google.golang.org/grpc"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/internal/runnable"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/config/loader"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	dlendpoints "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer/endpoints"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	fccontroller "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/controller"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/fairness"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/ordering"
	fcregistry "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/registry"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	dlmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/source/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/requestcontrol/requestattributereporter"
	testresponsereceived "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/requestcontrol/test/responsereceived"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/scorer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/scorer/predictedlatency"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	testfilter "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/test/filter"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics/collectors"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/saturationdetector/framework/plugins/utilizationdetector"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/server"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/env"
	"sigs.k8s.io/gateway-api-inference-extension/version"
)

const (
	// enableExperimentalDatalayerV2 defines the environment variable used as feature flag for the pluggable data layer.
	// DEPRECATION NOTICE - this env var will be removed in the next version as we switch to configuring the EPP using FeatureGates in the config file.
	enableExperimentalDatalayerV2 = "ENABLE_EXPERIMENTAL_DATALAYER_V2"
	// enableExperimentalFlowControlLayer defines the environment variable used as a feature flag for the pluggable flow
	// control layer.
	// DEPRECATION NOTICE - this env var will be removed in the next version as we switch to configuring the EPP using FeatureGates in the config file.
	enableExperimentalFlowControlLayer = "ENABLE_EXPERIMENTAL_FLOW_CONTROL_LAYER"

	// Saturation Detector deprecated configuration environment variables
	// DEPRECATION NOTICE - these env vars will be removed in the next version as we switch to configuring the EPP using the config file.
	EnvSdQueueDepthThreshold       = "SD_QUEUE_DEPTH_THRESHOLD"
	EnvSdKVCacheUtilThreshold      = "SD_KV_CACHE_UTIL_THRESHOLD"
	EnvSdMetricsStalenessThreshold = "SD_METRICS_STALENESS_THRESHOLD"
)

var (
	setupLog = ctrl.Log.WithName("setup")
)

// NewRunner initializes a new EPP Runner and returns its pointer.
func NewRunner() *Runner {
	return &Runner{
		eppExecutableName:    "GIE",
		requestControlConfig: requestcontrol.NewConfig(), // default requestcontrol config has empty plugin list
		customCollectors:     []prometheus.Collector{},
	}
}

// Runner is used to run epp with its plugins
type Runner struct {
	eppExecutableName    string // the EPP executable name
	featureGates         map[string]bool
	requestControlConfig *requestcontrol.Config
	schedulerConfig      *scheduling.SchedulerConfig
	customCollectors     []prometheus.Collector
}

// WithExecutableName sets the name of the executable containing the runner.
// The name is used in the version log upon startup and is otherwise opaque.
func (r *Runner) WithExecutableName(exeName string) *Runner {
	r.eppExecutableName = exeName
	return r
}

func (r *Runner) WithRequestControlConfig(requestControlConfig *requestcontrol.Config) *Runner {
	r.requestControlConfig = requestControlConfig
	return r
}

func (r *Runner) WithSchedulerConfig(schedulerConfig *scheduling.SchedulerConfig) *Runner {
	r.schedulerConfig = schedulerConfig
	return r
}

func (r *Runner) WithCustomCollectors(collectors ...prometheus.Collector) *Runner {
	r.customCollectors = collectors
	return r
}

func (r *Runner) Run(ctx context.Context) error {
	setupLog.Info(r.eppExecutableName+" build", "commit-sha", version.CommitSHA, "build-ref", version.BuildRef)

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

	// Print all flag values
	flags := make(map[string]any)
	flag.VisitAll(func(f *flag.Flag) {
		flags[f.Name] = f.Value
	})
	setupLog.Info("Flags processed", "flags", flags)

	initLogging(&opts.ZapOptions)

	if opts.Tracing {
		err := common.InitTracing(ctx, setupLog)
		if err != nil {
			return err
		}
	}

	// --- Get Kubernetes Config ---
	cfg, err := ctrl.GetConfig()
	if err != nil {
		setupLog.Error(err, "Failed to get Kubernetes rest config")
		return err
	}

	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	if err != nil {
		setupLog.Error(err, "Failed to parse configuration")
		return err
	}

	// --- Setup Datastore ---
	epf, err := r.setupMetricsCollection(r.featureGates[datalayer.ExperimentalDatalayerFeatureGate], opts)
	if err != nil {
		return err
	}

	gknn, err := extractGKNN(opts.PoolName, opts.PoolGroup, opts.PoolNamespace, opts.EndpointSelector)
	if err != nil {
		setupLog.Error(err, "Failed to extract GKNN")
		return err
	}

	startCrdReconcilers := opts.EndpointSelector == "" // If endpointSelector is empty, it means it's not in the standalone mode. Then we should start the inferencePool and other CRD Reconciler.
	controllerCfg := runserver.NewControllerConfig(startCrdReconcilers)

	ds, err := setupDatastore(ctx, epf, int32(opts.ModelServerMetricsPort), startCrdReconcilers,
		opts.PoolName, opts.PoolNamespace, opts.EndpointSelector, opts.EndpointTargetPorts)
	if err != nil {
		setupLog.Error(err, "Failed to setup datastore")
		return err
	}
	eppConfig, err := r.parseConfigurationPhaseTwo(ctx, rawConfig, ds)
	if err != nil {
		setupLog.Error(err, "Failed to parse configuration")
		return err
	}

	// --- Setup Metrics Server ---
	r.customCollectors = append(r.customCollectors, collectors.NewInferencePoolMetricsCollector(ds))
	metrics.Register(r.customCollectors...)
	metrics.RecordInferenceExtensionInfo(version.CommitSHA, version.BuildRef)
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

	isLeader := &atomic.Bool{}
	isLeader.Store(false)

	mgr, err := runserver.NewDefaultManager(controllerCfg, *gknn, cfg, metricsServerOptions, opts.EnableLeaderElection)
	if err != nil {
		setupLog.Error(err, "Failed to create controller manager")
		return err
	}

	if opts.EnableLeaderElection {
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

	if opts.EnablePprof {
		if err = setupPprofHandlers(mgr); err != nil {
			setupLog.Error(err, "Failed to setup pprof handlers")
			return err
		}
	}

	// --- Initialize Core EPP Components ---
	if r.schedulerConfig == nil {
		err := errors.New("scheduler config must be set either by config api or through code")
		setupLog.Error(err, "failed to create scheduler")
		return err
	}

	setupLog.Info("parsed config", "scheduler-config", r.schedulerConfig)

	scheduler := scheduling.NewSchedulerWithConfig(r.schedulerConfig)

	datalayerMetricsEnabled := r.featureGates[datalayer.ExperimentalDatalayerFeatureGate]
	if err := r.setupDataLayer(datalayerMetricsEnabled, eppConfig.DataConfig, epf, ds); err != nil {
		setupLog.Error(err, "failed to initialize data layer")
		return err
	}

	saturationDetector := utilizationdetector.NewDetector(eppConfig.SaturationDetectorConfig, setupLog)

	// --- Admission Control Initialization ---
	var admissionController requestcontrol.AdmissionController
	var locator contracts.PodLocator
	locator = requestcontrol.NewDatastorePodLocator(ds, requestcontrol.WithDisableEndpointSubsetFilter(opts.DisableEndpointSubsetFilter))
	if r.featureGates[flowcontrol.FeatureGate] {
		locator = requestcontrol.NewCachedPodLocator(ctx, locator, time.Millisecond*50)
		setupLog.Info("Initializing experimental Flow Control layer")
		registry, err := fcregistry.NewFlowRegistry(eppConfig.FlowControlConfig.Registry, setupLog)
		if err != nil {
			return fmt.Errorf("failed to initialize Flow Registry: %w", err)
		}
		fc, err := fccontroller.NewFlowController(
			ctx,
			eppConfig.FlowControlConfig.Controller,
			registry, saturationDetector,
			locator,
		)
		if err != nil {
			return fmt.Errorf("failed to initialize Flow Controller: %w", err)
		}
		go registry.Run(ctx)
		admissionController = requestcontrol.NewFlowControlAdmissionController(fc, opts.PoolName)
	} else {
		setupLog.Info("Experimental Flow Control layer is disabled, using legacy admission control")
		admissionController = requestcontrol.NewLegacyAdmissionController(saturationDetector, locator)
	}

	director := requestcontrol.NewDirectorWithConfig(ds, scheduler, admissionController, locator, r.requestControlConfig)

	// --- Setup ExtProc Server Runner ---
	serverRunner := &runserver.ExtProcServerRunner{
		GrpcPort:                         opts.GRPCPort,
		GKNN:                             *gknn,
		Datastore:                        ds,
		ControllerCfg:                    controllerCfg,
		SecureServing:                    opts.SecureServing,
		HealthChecking:                   opts.HealthChecking,
		CertPath:                         opts.CertPath,
		EnableCertReload:                 opts.EnableCertReload,
		RefreshPrometheusMetricsInterval: opts.RefreshPrometheusMetricsInterval,
		MetricsStalenessThreshold:        opts.MetricsStalenessThreshold,
		Director:                         director,
		SaturationDetector:               saturationDetector,
		UseExperimentalDatalayerV2:       r.featureGates[datalayer.ExperimentalDatalayerFeatureGate], // pluggable data layer feature flag
	}
	if err := serverRunner.SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "Failed to setup EPP controllers")
		return err
	}

	// --- Add Runnables to Manager ---
	// Register health server.
	if err := registerHealthServer(mgr, ctrl.Log.WithName("health"), ds, opts.GRPCHealthPort, isLeader, opts.EnableLeaderElection); err != nil {
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

func setupDatastore(ctx context.Context, epFactory datalayer.EndpointFactory, modelServerMetricsPort int32,
	startCrdReconcilers bool, namespace, name, endpointSelector string, endpointTargetPorts []int) (datastore.Datastore, error) {
	if startCrdReconcilers {
		return datastore.NewDatastore(ctx, epFactory, modelServerMetricsPort), nil
	} else {
		endpointPool := datalayer.NewEndpointPool(namespace, name)
		labelsMap, err := labels.ConvertSelectorToLabelsMap(endpointSelector)
		if err != nil {
			setupLog.Error(err, "Failed to parse flag %q with error: %w", "endpoint-selector", err)
			return nil, err
		}
		endpointPool.Selector = labelsMap
		endpointPool.TargetPorts = append(endpointPool.TargetPorts, endpointTargetPorts...)

		endpointPoolOption := datastore.WithEndpointPool(endpointPool)
		return datastore.NewDatastore(ctx, epFactory, modelServerMetricsPort, endpointPoolOption), nil
	}
}

// registerInTreePlugins registers the factory functions of all known plugins
func (r *Runner) registerInTreePlugins() {
	fwkplugin.Register(prefix.PrefixCachePluginType, prefix.PrefixCachePluginFactory)
	fwkplugin.Register(picker.MaxScorePickerType, picker.MaxScorePickerFactory)
	fwkplugin.Register(picker.RandomPickerType, picker.RandomPickerFactory)
	fwkplugin.Register(picker.WeightedRandomPickerType, picker.WeightedRandomPickerFactory)
	fwkplugin.Register(profile.SingleProfileHandlerType, profile.SingleProfileHandlerFactory)
	fwkplugin.Register(scorer.KvCacheUtilizationScorerType, scorer.KvCacheUtilizationScorerFactory)
	fwkplugin.Register(scorer.QueueScorerType, scorer.QueueScorerFactory)
	fwkplugin.Register(scorer.RunningRequestsSizeScorerType, scorer.RunningRequestsSizeScorerFactory)
	fwkplugin.Register(scorer.LoraAffinityScorerType, scorer.LoraAffinityScorerFactory)
	// Flow Control plugins
	fwkplugin.Register(fairness.GlobalStrictFairnessPolicyType, fairness.GlobalStrictFairnessPolicyFactory)
	fwkplugin.Register(fairness.RoundRobinFairnessPolicyType, fairness.RoundRobinFairnessPolicyFactory)
	fwkplugin.Register(ordering.FCFSOrderingPolicyType, ordering.FCFSOrderingPolicyFactory)
	fwkplugin.Register(ordering.EDFOrderingPolicyType, ordering.EDFOrderingPolicyFactory)
	// Latency predictor plugins
	fwkplugin.Register(predictedlatency.PredictedLatencyPluginType, predictedlatency.PredictedLatencyFactory)
	// register filter for test purpose only (used in conformance tests)
	fwkplugin.Register(testfilter.HeaderBasedTestingFilterType, testfilter.HeaderBasedTestingFilterFactory)
	// register response received plugin for test purpose only (used in conformance tests)
	fwkplugin.Register(testresponsereceived.DestinationEndpointServedVerifierType, testresponsereceived.DestinationEndpointServedVerifierFactory)
	// register datalayer metrics collection plugins
	fwkplugin.Register(dlmetrics.MetricsDataSourceType, dlmetrics.MetricsDataSourceFactory)
	fwkplugin.Register(dlmetrics.MetricsExtractorType, dlmetrics.ModelServerExtractorFactory)
	// register datalayer endpoint notification plugin
	fwkplugin.Register(dlendpoints.EndpointsDataSourceType, dlendpoints.EndpointDataSourceFactory)
	// register the request attribute reporter plugin
	fwkplugin.Register(requestattributereporter.RequestAttributeReporterType, requestattributereporter.RequestAttributeReporterPluginFactory)
}

func (r *Runner) parseConfigurationPhaseOne(ctx context.Context, opts *runserver.Options) (*configapi.EndpointPickerConfig, error) {
	if opts.ConfigText == "" && opts.ConfigFile == "" {
		return nil, nil // configuring through code, not through file
	}

	logger := log.FromContext(ctx)

	var configBytes []byte
	if opts.ConfigText != "" {
		configBytes = []byte(opts.ConfigText)
	} else if opts.ConfigFile != "" { // if config was specified through a file
		var err error
		configBytes, err = os.ReadFile(opts.ConfigFile)
		if err != nil {
			return nil, fmt.Errorf("failed to load config from a file '%s' - %w", opts.ConfigFile, err)
		}
	}

	loader.RegisterFeatureGate(datalayer.ExperimentalDatalayerFeatureGate)
	loader.RegisterFeatureGate(flowcontrol.FeatureGate)
	loader.RegisterFeatureGate(datalayer.PrepareDataPluginsFeatureGate)

	r.registerInTreePlugins()

	rawConfig, featureGates, err := loader.LoadRawConfig(configBytes, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to parse config - %w", err)
	}

	r.featureGates = featureGates

	return rawConfig, nil
}

// Return a function that can be used in the EPP Handle to list pod names.
func makePodListFunc(ds datastore.Datastore) func() []types.NamespacedName {
	return func() []types.NamespacedName {
		pods := ds.PodList(datastore.AllPodsPredicate)
		names := make([]types.NamespacedName, 0, len(pods))

		for _, p := range pods {
			names = append(names, p.GetMetadata().NamespacedName)
		}
		return names
	}
}

func (r *Runner) parseConfigurationPhaseTwo(ctx context.Context, rawConfig *configapi.EndpointPickerConfig, ds datastore.Datastore) (*config.Config, error) {
	logger := log.FromContext(ctx)

	applyDeprecatedEnvFeatureGate(enableExperimentalDatalayerV2, "Data Layer V2", datalayer.ExperimentalDatalayerFeatureGate, rawConfig)
	applyDeprecatedEnvFeatureGate(enableExperimentalFlowControlLayer, "Flow Control layer", flowcontrol.FeatureGate, rawConfig)

	handle := fwkplugin.NewEppHandle(ctx, makePodListFunc(ds))
	cfg, err := loader.InstantiateAndConfigure(rawConfig, handle, logger)

	if err != nil {
		return nil, fmt.Errorf("failed to load the configuration - %w", err)
	}

	r.schedulerConfig = cfg.SchedulerConfig

	// Add requestControl plugins
	r.requestControlConfig.AddPlugins(handle.GetAllPlugins()...)

	// Sort prepare data plugins in DAG order (topological sort). Also check prepare data plugins for cycles.
	if r.requestControlConfig.PrepareDataPluginGraph() != nil {
		return nil, errors.New("failed to load the configuration - prepare data plugins have cyclic dependencies")
	}
	// TODO(#1970): Remove feature gate check once prepare data plugins are stable.
	if !r.featureGates[datalayer.PrepareDataPluginsFeatureGate] {
		// If the feature gate is disabled, clear any prepare data plugins so they are not used.
		r.requestControlConfig.WithPrepareDataPlugins()
	}

	r.applyDeprecatedSaturationConfig(cfg)

	logger.Info("loaded configuration from file/text successfully")
	return cfg, nil
}

func applyDeprecatedEnvFeatureGate(envVar, featureName, featureGate string, rawConfig *configapi.EndpointPickerConfig) {
	if _, ok := os.LookupEnv(envVar); ok {
		setupLog.Info(fmt.Sprintf("Enabling the experimental %s using environment variables is deprecated and will be removed in next version", featureName))
		if env.GetEnvBool(envVar, false, setupLog) {
			if rawConfig.FeatureGates == nil {
				rawConfig.FeatureGates = make(configapi.FeatureGates, 0)
			}
			rawConfig.FeatureGates = append(rawConfig.FeatureGates, featureGate)
		}
	}
}

func (r *Runner) applyDeprecatedSaturationConfig(cfg *config.Config) {
	if _, ok := os.LookupEnv(EnvSdQueueDepthThreshold); ok {
		setupLog.Info("Configuring Saturation Detector using environment variables is deprecated and will be removed in next version")
		cfg.SaturationDetectorConfig.QueueDepthThreshold =
			env.GetEnvInt(EnvSdQueueDepthThreshold, utilizationdetector.DefaultQueueDepthThreshold, setupLog)
		if cfg.SaturationDetectorConfig.QueueDepthThreshold <= 0 {
			cfg.SaturationDetectorConfig.QueueDepthThreshold = utilizationdetector.DefaultQueueDepthThreshold
		}
	}
	if _, ok := os.LookupEnv(EnvSdKVCacheUtilThreshold); ok {
		setupLog.Info("Configuring Saturation Detector using environment variables is deprecated and will be removed in next version")
		cfg.SaturationDetectorConfig.KVCacheUtilThreshold = env.GetEnvFloat(EnvSdKVCacheUtilThreshold, utilizationdetector.DefaultKVCacheUtilThreshold, setupLog)
		if cfg.SaturationDetectorConfig.KVCacheUtilThreshold <= 0 || cfg.SaturationDetectorConfig.KVCacheUtilThreshold >= 1 {
			cfg.SaturationDetectorConfig.KVCacheUtilThreshold = utilizationdetector.DefaultKVCacheUtilThreshold
		}
	}
	if _, ok := os.LookupEnv(EnvSdMetricsStalenessThreshold); ok {
		setupLog.Info("Configuring Saturation Detector using environment variables is deprecated and will be removed in next version")
		cfg.SaturationDetectorConfig.MetricsStalenessThreshold = env.GetEnvDuration(EnvSdMetricsStalenessThreshold, utilizationdetector.DefaultMetricsStalenessThreshold, setupLog)
		if cfg.SaturationDetectorConfig.MetricsStalenessThreshold <= 0 {
			cfg.SaturationDetectorConfig.MetricsStalenessThreshold = utilizationdetector.DefaultMetricsStalenessThreshold
		}
	}
}

func (r *Runner) setupDataLayer(enableNewMetrics bool, cfg *datalayer.Config, epf datalayer.EndpointFactory, ds fwkdl.StoreNotifier) error {
	disallowedMetricsExtractor := ""
	if !enableNewMetrics { // using backend.PodMetrics, disallow datalayer's metrics data source/extractor
		disallowedMetricsExtractor = dlmetrics.MetricsExtractorType
	}

	if err := datalayer.WithConfig(cfg, disallowedMetricsExtractor); err != nil {
		return err
	}

	sources := datalayer.GetSources()
	if enableNewMetrics && len(sources) == 0 {
		err := errors.New("data layer enabled but no data sources configured")
		return err
	}

	// Separate collection sources (timer-based) from notification sources
	collectors := []fwkdl.DataSource{}

	for _, src := range sources {
		setupLog.Info("data layer configuration", "source", src.TypedName().String(), "extractors", src.Extractors())

		// Check if this is a notification source and attach to datastore
		if notifySrc, ok := src.(fwkdl.NotificationDataSource); ok {
			if err := notifySrc.AttachToStore(ds); err != nil {
				return fmt.Errorf("failed to attach notification source %s: %w", src.TypedName().String(), err)
			}
		} else if collectSrc, ok := src.(fwkdl.DataSource); ok {
			// This is a collection source, add to list for EndpointFactory
			collectors = append(collectors, collectSrc)
		}
	}

	// Only pass collection sources to EndpointFactory for timer-based collection
	epf.SetSources(collectors)

	return nil
}

func (r *Runner) setupMetricsCollection(enableNewMetrics bool, opts *runserver.Options) (datalayer.EndpointFactory, error) {
	if enableNewMetrics {
		return datalayer.NewEndpointFactory(nil, opts.RefreshMetricsInterval), nil
	}
	return setupMetricsV1(opts)
}

func setupMetricsV1(opts *runserver.Options) (datalayer.EndpointFactory, error) {
	mapping, err := backendmetrics.NewMetricMapping(
		opts.TotalQueuedRequestsMetric,
		opts.TotalRunningRequestsMetric,
		opts.KVCacheUsagePercentageMetric,
		opts.LoRAInfoMetric,
		opts.CacheInfoMetric,
	)
	if err != nil {
		setupLog.Error(err, "Failed to create metric mapping from flags.")
		return nil, err
	}
	verifyMetricMapping(*mapping)

	var metricsHttpClient *http.Client
	if opts.ModelServerMetricsScheme == "https" {
		metricsHttpClient = &http.Client{
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{
					InsecureSkipVerify: opts.ModelServerMetricsHTTPSInsecure,
				},
			},
		}
	} else {
		metricsHttpClient = http.DefaultClient
	}

	pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.PodMetricsClientImpl{
		MetricMapping:            mapping,
		ModelServerMetricsPath:   opts.ModelServerMetricsPath,
		ModelServerMetricsScheme: opts.ModelServerMetricsScheme,
		Client:                   metricsHttpClient,
	},
		opts.RefreshMetricsInterval)
	return pmf, nil
}

func initLogging(opts *zap.Options) {
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

func verifyMetricMapping(mapping backendmetrics.MetricMapping) {
	if mapping.TotalQueuedRequests == nil {
		setupLog.Info("Not scraping metric: TotalQueuedRequests")
	}
	if mapping.KVCacheUtilization == nil {
		setupLog.Info("Not scraping metric: KVCacheUtilization")
	}
	if mapping.LoraRequestInfo == nil {
		setupLog.Info("Not scraping metric: LoraRequestInfo")
	}
	if mapping.CacheConfigInfo == nil {
		setupLog.Info("Not scraping metric: CacheConfigInfo")
	}
}

// setupPprofHandlers only implements the pre-defined profiles:
// https://cs.opensource.google/go/go/+/refs/tags/go1.24.4:src/runtime/pprof/pprof.go;l=108
func setupPprofHandlers(mgr ctrl.Manager) error {
	setupLog.Info("Enabling pprof handlers")
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

	runtime.SetMutexProfileFraction(1)
	runtime.SetBlockProfileRate(1)

	return nil
}

func extractDeploymentName(podName string) (string, error) {
	regex := regexp.MustCompile(`^(.+)-[a-z0-9]+-[a-z0-9]+$`)

	matches := regex.FindStringSubmatch(podName)
	if len(matches) == 2 {
		return matches[1], nil
	}
	return "", fmt.Errorf("failed to parse deployment name from pod name %s", podName)
}

func extractGKNN(poolName, poolGroup, poolNamespace, endpointSelector string) (*common.GKNN, error) {
	if poolName != "" {
		// Determine pool namespace: if --pool-namespace is non-empty, use it; else NAMESPACE env var; else default
		resolvedPoolNamespace := resolvePoolNamespace(poolNamespace)
		poolNamespacedName := types.NamespacedName{
			Name:      poolName,
			Namespace: resolvedPoolNamespace,
		}
		poolGroupKind := schema.GroupKind{
			Group: poolGroup,
			Kind:  "InferencePool",
		}
		return &common.GKNN{
			NamespacedName: poolNamespacedName,
			GroupKind:      poolGroupKind,
		}, nil
	}

	if endpointSelector != "" {
		// Determine EPP namespace: NAMESPACE env var; else default
		resolvedPoolNamespace := resolvePoolNamespace(poolNamespace)
		// Determine EPP name: POD_NAME env var
		eppPodNameEnv := os.Getenv("POD_NAME")
		if eppPodNameEnv == "" {
			return nil, errors.New("failed to get environment variable POD_NAME")

		}
		eppName, err := extractDeploymentName(eppPodNameEnv)
		if err != nil {
			return nil, err
		}
		return &common.GKNN{
			NamespacedName: types.NamespacedName{Namespace: resolvedPoolNamespace, Name: eppName},
			GroupKind:      schema.GroupKind{Kind: "Deployment", Group: "apps"},
		}, nil
	}
	return nil, errors.New("can't construct gknn as both pool-name and endpoint-selector are missing")
}

func resolvePoolNamespace(poolNamespace string) string {
	if poolNamespace != "" {
		return poolNamespace
	}
	if nsEnv := os.Getenv("NAMESPACE"); nsEnv != "" {
		return nsEnv
	}
	return runserver.DefaultPoolNamespace
}
