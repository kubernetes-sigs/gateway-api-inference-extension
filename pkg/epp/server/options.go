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

package server

import (
	"errors"
	"flag"
	"fmt"
	"time"

	"github.com/spf13/pflag"
	uberzap "go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	DefaultGrpcPort      = 9002
	DefaultPoolNamespace = "default" // default when pool namespace is empty (CLI flag default is empty)
	ZapLogLevelFlagName  = "zap-log-level"
)

// Options contains configuration values necessary to create and run the EPP.
type Options struct {
	//
	// ext_proc configuration.
	//
	GRPCPort             int  // gRPC port used for communicating with Envoy proxy. (TODO: uint16?)
	EnableLeaderElection bool // Enables leader election for high availability
	//
	// InferencePool.
	//
	PoolGroup     string // Kubernetes resource group of the InferencePool this Endpoint Picker is associated with.
	PoolNamespace string // Namespace of the InferencePool this Endpoint Picker is associated with.
	PoolName      string // Name of the InferencePool this Endpoint Picker is associated with.
	//
	// Endpoints (in lieu of using an InferencePool for service discovery).
	//
	EndpointSelector    string // Selector to filter model server pods on, only 'key=value' pairs are supported. (TODO: k8s.Selector, pflag.StringSlice?)
	EndpointTargetPorts []int  // Target ports of model server pods.
	//
	// MSP metrics scraping.
	//
	ModelServerMetricsScheme         string        // Protocol scheme used in scraping metrics from endpoints.
	ModelServerMetricsPath           string        // URL path used in scraping metrics from endpoints.
	ModelServerMetricsPort           int           // Port to scrape metrics from endpoints. (TODO: Deprecated, uint16)
	ModelServerMetricsHTTPSInsecure  bool          // Disable certificate verification when using 'https' scheme for 'model-server-metrics-scheme'.
	RefreshMetricsInterval           time.Duration // Interval to refresh metrics.
	RefreshPrometheusMetricsInterval time.Duration // Interval to flush Prometheus metrics.
	MetricsStalenessThreshold        time.Duration // Duration after which metrics are considered stale.
	TotalQueuedRequestsMetric        string        // Prometheus metric specification for the number of queued requests.
	TotalRunningRequestsMetric       string        // Prometheus metric specification for the number of running requests.
	KVCacheUsagePercentageMetric     string        // Prometheus metric specification for the fraction of KV-cache blocks currently in use.
	LoRAInfoMetric                   string        // Prometheus metric specification for the LoRA info metrics.
	CacheInfoMetric                  string        // Prometheus metric specification for the cache info metrics.
	//
	// Diagnostics.
	//
	LogVerbosity        int         // Number for the log level verbosity.
	ZapOptions          zap.Options // Zap logging options
	Tracing             bool        // Enables emitting traces.
	HealthChecking      bool        // Enables health checking.
	MetricsPort         int         // The metrics port exposed by EPP. (TODO: uint16)
	GRPCHealthPort      int         // The port used for gRPC liveness and readiness probes. (TODO: uint16)
	EnablePprof         bool        // Enables pprof handlers.
	CertPath            string      // The path to the certificate for secure serving.
	EnableCertReload    bool        // Enables certificate reloading of the certificates specified in --cert-path.
	SecureServing       bool        // Enables secure serving.
	MetricsEndpointAuth bool        // Enables authentication and authorization of the metrics endpoint.
	//
	// Configuration.
	//
	ConfigFile string // The path to the configuration file.
	ConfigText string // The configuration specified as text, in lieu of a file.

	// internal
	fs *pflag.FlagSet // FlagSet used in AddFlags() and consulted in Complete()
}

// NewOptions returns a new Options struct initialized with the default values.
func NewOptions() *Options {
	return &Options{ // "zero" values are no explicitly set
		GRPCPort:                         DefaultGrpcPort,
		PoolGroup:                        "inference.networking.k8s.io",
		EndpointTargetPorts:              []int{},
		ModelServerMetricsScheme:         "http",
		ModelServerMetricsPath:           "/metrics",
		ModelServerMetricsHTTPSInsecure:  true,
		RefreshMetricsInterval:           50 * time.Millisecond,
		RefreshPrometheusMetricsInterval: 5 * time.Second,
		MetricsStalenessThreshold:        2 * time.Second,
		TotalQueuedRequestsMetric:        "vllm:num_requests_waiting",
		TotalRunningRequestsMetric:       "vllm:num_requests_running",
		KVCacheUsagePercentageMetric:     "vllm:kv_cache_usage_perc",
		LoRAInfoMetric:                   "vllm:lora_requests_info",
		CacheInfoMetric:                  "vllm:cache_config_info",
		LogVerbosity:                     logging.DEFAULT,
		ZapOptions:                       zap.Options{Development: true},
		Tracing:                          true,
		MetricsPort:                      9090,
		GRPCHealthPort:                   9003,
		EnablePprof:                      true,
		SecureServing:                    true,
		MetricsEndpointAuth:              true,
	}
}

func (opts *Options) AddFlags(fs *pflag.FlagSet) {
	if fs == nil {
		fs = pflag.CommandLine
	}
	opts.fs = fs

	fs.IntVar(&opts.GRPCPort, "grpc-port", opts.GRPCPort, "gRPC port used for communicating with Envoy proxy.")
	fs.BoolVar(&opts.EnableLeaderElection, "ha-enable-leader-election", opts.EnableLeaderElection,
		"Enables leader election for high availability. When enabled, readiness probes will only pass on the leader.")
	fs.StringVar(&opts.PoolGroup, "pool-group", opts.PoolGroup,
		"Kubernetes resource group of the InferencePool this Endpoint Picker is associated with.")
	fs.StringVar(&opts.PoolNamespace, "pool-namespace", opts.PoolNamespace,
		"Namespace of the InferencePool this Endpoint Picker is associated with.")
	fs.StringVar(&opts.PoolName, "pool-name", opts.PoolName, "Name of the InferencePool this Endpoint Picker is associated with.")
	fs.StringVar(&opts.EndpointSelector, "endpoint-selector", opts.EndpointSelector,
		"Selector to filter model server pods on, only 'key=value' pairs are supported. "+
			"Format: a comma-separated list of key=value pairs without whitespace (e.g., 'app=vllm-llama3-8b-instruct,env=prod').")
	fs.IntSliceVar(&opts.EndpointTargetPorts, "endpoint-target-ports", opts.EndpointTargetPorts, "Target ports of model server pods. "+
		"Format: a comma-separated list of numbers without whitespace (e.g., '3000,3001,3002').")
	fs.StringVar(&opts.ModelServerMetricsScheme, "model-server-metrics-scheme", opts.ModelServerMetricsScheme,
		"Protocol scheme used in scraping metrics from endpoints.")
	fs.StringVar(&opts.ModelServerMetricsPath, "model-server-metrics-path", opts.ModelServerMetricsPath,
		"URL path used in scraping metrics from endpoints.")
	fs.IntVar(&opts.ModelServerMetricsPort, "model-server-metrics-port", opts.ModelServerMetricsPort,
		"Port to scrape metrics from endpoints. Set to the InferencePool.Spec.TargetPorts[0].Number if not defined.")
	_ = fs.MarkDeprecated("model-server-metrics-port", "This flag is deprecated and will be removed in a future release.")
	fs.BoolVar(&opts.ModelServerMetricsHTTPSInsecure, "model-server-metrics-https-insecure-skip-verify", opts.ModelServerMetricsHTTPSInsecure,
		"Disable certificate verification when using 'https' scheme for 'model-server-metrics-scheme'.")
	fs.DurationVar(&opts.RefreshMetricsInterval, "refresh-metrics-interval", opts.RefreshMetricsInterval, "Interval to refresh metrics.")
	fs.DurationVar(&opts.RefreshPrometheusMetricsInterval, "refresh-prometheus-metrics-interval", opts.RefreshPrometheusMetricsInterval,
		"Interval to flush Prometheus metrics.")
	fs.DurationVar(&opts.MetricsStalenessThreshold, "metrics-staleness-threshold", opts.MetricsStalenessThreshold,
		"Duration after which metrics are considered stale. This is used to determine if an endpoint's metrics are fresh enough.")
	fs.StringVar(&opts.TotalQueuedRequestsMetric, "total-queued-requests-metric", opts.TotalQueuedRequestsMetric,
		"Prometheus metric for the number of queued requests.")
	fs.StringVar(&opts.TotalRunningRequestsMetric, "total-running-requests-metric", opts.TotalRunningRequestsMetric,
		"Prometheus metric for the number of running requests.")
	fs.StringVar(&opts.KVCacheUsagePercentageMetric, "kv-cache-usage-percentage-metric", opts.KVCacheUsagePercentageMetric,
		"Prometheus metric for the fraction of KV-cache blocks currently in use (from 0 to 1).")
	fs.StringVar(&opts.LoRAInfoMetric, "lora-info-metric", opts.LoRAInfoMetric,
		"Prometheus metric for the LoRA info metrics (must be in vLLM label format).")
	fs.StringVar(&opts.CacheInfoMetric, "cache-info-metric", opts.CacheInfoMetric, "Prometheus metric for the cache info metrics.")
	fs.IntVarP(&opts.LogVerbosity, "v", "v", opts.LogVerbosity, "Number for the log level verbosity.") // allow both --v and -v
	gofs := flag.NewFlagSet("zap", flag.ExitOnError)
	opts.ZapOptions.BindFlags(gofs) // zap expects a standard Go FlagSet and pflag.FlagSet is not compatible.
	fs.AddGoFlagSet(gofs)
	fs.BoolVar(&opts.Tracing, "tracing", opts.Tracing, "Enables emitting traces.")
	fs.BoolVar(&opts.HealthChecking, "health-checking", opts.HealthChecking, "Enables health checking.")
	fs.IntVar(&opts.MetricsPort, "metrics-port", opts.MetricsPort, "The metrics port exposed by EPP.")
	fs.IntVar(&opts.GRPCHealthPort, "grpc-health-port", opts.GRPCHealthPort,
		"The port used for gRPC liveness and readiness probes.")
	fs.BoolVar(&opts.EnablePprof, "enable-pprof", opts.EnablePprof,
		"Enables pprof handlers. Defaults to true. Set to false to disable pprof handlers.")
	fs.StringVar(&opts.CertPath, "cert-path", opts.CertPath,
		"The path to the certificate for secure serving. The certificate and private key files "+
			"are assumed to be named tls.crt and tls.key, respectively. If not set, and secureServing is enabled, "+
			"then a self-signed certificate is used.")
	fs.BoolVar(&opts.EnableCertReload, "enable-cert-reload", opts.EnableCertReload,
		"Enables certificate reloading of the certificates specified in --cert-path.")
	fs.BoolVar(&opts.SecureServing, "secure-serving", opts.SecureServing, "Enables secure serving.")
	fs.BoolVar(&opts.MetricsEndpointAuth, "metrics-endpoint-auth", opts.MetricsEndpointAuth,
		"Enables authentication and authorization of the metrics endpoint.")
	fs.StringVar(&opts.ConfigFile, "config-file", opts.ConfigFile, "The path to the configuration file.")
	fs.StringVar(&opts.ConfigText, "config-text", opts.ConfigText, "The configuration specified as text, in lieu of a file.")
}

func (opts *Options) Complete() error {
	// TODO: postprocessing or command line arguments. For example, convert EndpointSelector
	// from raw string to k8s.LabelSelector, load ConfigFile into ConfigText, etc.

	opts.EndpointTargetPorts = removeDuplicatePorts(opts.EndpointTargetPorts)

	// ensure zap log level is set - explicitly by user or from "-v"
	zapLogLevelFlag := opts.fs.Lookup(ZapLogLevelFlagName)
	if zapLogLevelFlag != nil && !zapLogLevelFlag.Changed { // not set explicitly
		lvl := -1 * (opts.LogVerbosity) // See https://pkg.go.dev/sigs.k8s.io/controller-runtime/pkg/log/zap#Options.Level
		opts.ZapOptions.Level = uberzap.NewAtomicLevelAt(zapcore.Level(int8(lvl)))
		zapLogLevelFlag.Changed = true
	}
	return nil
}

func (opts *Options) Validate() error {
	if (opts.PoolName != "" && opts.EndpointSelector != "") || (opts.PoolName == "" && opts.EndpointSelector == "") {
		return errors.New("either pool-name or endpoint-selector must be set")
	}
	if opts.EndpointSelector != "" {
		if len(opts.EndpointTargetPorts) == 0 || len(opts.EndpointTargetPorts) > 8 {
			return fmt.Errorf("flag %q should have length from 1 to 8", "endpoint-target-ports")
		}
		for _, port := range opts.EndpointTargetPorts { // valid port range
			if port < 0 || port > 65535 {
				return fmt.Errorf("invalid port number %d in %q", port, "endpoint-target-ports")
			}
		}
	}

	if opts.ConfigText != "" && opts.ConfigFile != "" {
		return fmt.Errorf("both the %q and %q flags can not be set at the same time", "configText", "configFile")
	}
	if opts.ModelServerMetricsScheme != "http" && opts.ModelServerMetricsScheme != "https" {
		return fmt.Errorf("unexpected %q value for %q flag, it can only be set to 'http' or 'https'",
			opts.ModelServerMetricsScheme, "model-server-metrics-scheme")
	}

	return nil
}

func removeDuplicatePorts(ports []int) []int {
	seen := sets.NewInt()
	unique := make([]int, 0, len(ports))

	for _, val := range ports {
		if !seen.Has(val) {
			unique = append(unique, val)
			seen.Insert(val)
		}
	}
	return unique
}
