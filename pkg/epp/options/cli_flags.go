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

package options

import (
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

var (
	//
	// ext_proc configuration
	//
	GRPCPort = Flag{
		Name:     "grpc-port",
		DefValue: 9002,
		Usage:    "gRPC port used for communicating with Envoy proxy.",
	}
	EnableLeaderElection = Flag{
		Name:     "ha-enable-leader-election",
		DefValue: false,
		Usage:    "Enables leader election for high availability. When enabled, readiness probes will only pass on the leader.",
	}

	//
	// InferencePool
	//
	PoolGroup = Flag{
		Name:     "pool-group",
		DefValue: "inference.networking.k8s.io",
		Usage:    "Kubernetes resource group of the InferencePool this Endpoint Picker is associated with.",
	}
	PoolNamespace = Flag{
		Name:     "pool-namespace",
		DefValue: "default",
		Usage:    "Namespace of the InferencePool this Endpoint Picker is associated with.",
	}
	PoolName = Flag{
		Name:     "pool-name",
		DefValue: "",
		Usage:    "Name of the InferencePool this Endpoint Picker is associated with.",
	}

	//
	// Endpoints (in lieu of using an InferencePool)
	//
	EndpointSelector = Flag{
		Name:     "endpoint-selector",
		DefValue: "",
		Usage: "Selector to filter model server pods on, only 'key=value' pairs are supported. " +
			"Format: a comma-separated list of key=value pairs without whitespace (e.g., 'app=vllm-llama3-8b-instruct,env=prod').",
	}
	EndpointTargetPorts = Flag{
		Name:     "endpoint-target-ports",
		DefValue: "",
		Usage: "Target ports of model server pods. " +
			"Format: a comma-separated list of numbers without whitespace (e.g., '3000,3001,3002').",
	}

	//
	// MSP metrics scraping
	//
	ModelServerMetricsScheme = Flag{
		Name:     "model-server-metrics-scheme",
		DefValue: "http",
		Usage:    "Protocol scheme used in scraping metrics from endpoints.",
	}
	ModelServerMetricsPath = Flag{
		Name:     "model-server-metrics-path",
		DefValue: "/metrics",
		Usage:    "URL path used in scraping metrics from endpoints.",
	}
	ModelServerMetricsPort = Flag{
		Name:       "model-server-metrics-port",
		DefValue:   0,
		Usage:      "Port to scrape metrics from endpoints. Set to the InferencePool.Spec.TargetPorts[0].Number if not defined.",
		Deprecated: true, // no replacement, to be removed
	}
	ModelServerMetricsHTTPSInsecure = Flag{
		Name:     "model-server-metrics-https-insecure-skip-verify",
		DefValue: true,
		Usage:    "Disable certificate verification when using 'https' scheme for 'model-server-metrics-scheme'.",
	}
	RefreshMetricsInterval = Flag{
		Name:     "refresh-metrics-interval",
		DefValue: 50 * time.Millisecond,
		Usage:    "Interval to refresh metrics.",
	}
	RefreshPrometheusMetricsInterval = Flag{
		Name:     "refresh-prometheus-metrics-interval",
		DefValue: 5 * time.Second,
		Usage:    "Interval to flush Prometheus metrics.",
	}
	MetricsStalenessThreshold = Flag{
		Name:     "metrics-staleness-threshold",
		DefValue: 2 * time.Second,
		Usage:    "Duration after which metrics are considered stale. This is used to determine if an endpoint's metrics are fresh enough.",
	}
	TotalQueuedRequestsMetric = Flag{
		Name:     "total-queued-requests-metric",
		DefValue: "vllm:num_requests_waiting",
		Usage:    "Prometheus metric for the number of queued requests.",
	}
	TotalRunningRequestsMetric = Flag{
		Name:     "total-running-requests-metric",
		DefValue: "vllm:num_requests_running",
		Usage:    "Prometheus metric for the number of running requests.",
	}
	KVCacheUsagePercentageMetric = Flag{
		Name:     "kv-cache-usage-percentage-metric",
		DefValue: "vllm:kv_cache_usage_perc",
		Usage:    "Prometheus metric for the fraction of KV-cache blocks currently in use (from 0 to 1).",
	}
	LoRAInfoMetric = Flag{
		Name:     "lora-info-metric",
		DefValue: "vllm:lora_requests_info",
		Usage:    "Prometheus metric for the LoRA info metrics (must be in vLLM label format).",
	}
	CacheInfoMetric = Flag{
		Name:     "cache-info-metric",
		DefValue: "vllm:cache_config_info",
		Usage:    "Prometheus metric for the cache info metrics.",
	}

	//
	// Diagnostics
	//
	LogVerbosity = Flag{
		Name:     "v",
		DefValue: logging.DEFAULT,
		Usage:    "Number for the log level verbosity.",
	}
	Tracing = Flag{
		Name:     "tracing",
		DefValue: true,
		Usage:    "Enables emitting traces.",
	}
	HealthChecking = Flag{
		Name:     "health-checking",
		DefValue: false,
		Usage:    "Enables health checking.",
	}
	MetricsPort = Flag{
		Name:     "metrics-port",
		DefValue: 9090,
		Usage:    "The metrics port exposed by EPP.",
	}
	GRPCHealthPort = Flag{
		Name:     "grpc-health-port",
		DefValue: 9003,
		Usage:    "The port used for gRPC liveness and readiness probes.",
	}
	EnablePprof = Flag{
		Name:     "enable-pprof",
		DefValue: true,
		Usage:    "Enables pprof handlers. Defaults to true. Set to false to disable pprof handlers.",
	}
	CertPath = Flag{
		Name:     "cert-path",
		DefValue: "",
		Usage: "The path to the certificate for secure serving. The certificate and private key files " +
			"are assumed to be named tls.crt and tls.key, respectively. If not set, and secureServing is enabled, " +
			"then a self-signed certificate is used.",
	}
	EnableCertReload = Flag{
		Name:     "enable-cert-reload",
		DefValue: false,
		Usage:    "Enables certificate reloading of the certificates specified in --cert-path.",
	}
	SecureServing = Flag{
		Name:     "secure-serving",
		DefValue: true,
		Usage:    "Enables secure serving.",
	}
	MetricsEndpointAuth = Flag{
		Name:     "metrics-endpoint-auth",
		DefValue: true,
		Usage:    "Enables authentication and authorization of the metrics endpoint.",
	}

	//
	// Configuration
	//
	ConfigFile = Flag{
		Name:     "config-file",
		DefValue: "",
		Usage:    "The path to the configuration file.",
	}
	ConfigText = Flag{
		Name:     "config-text",
		DefValue: "",
		Usage:    "The configuration specified as text, in lieu of a file.",
	}
)
