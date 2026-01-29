/*
Copyright 2026 The Kubernetes Authors.

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

package metrics

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
	flag "github.com/spf13/pflag"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer/http"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	MetricsDataSourceType = "metrics-data-source"
	MetricsExtractorType  = "model-server-protocol-metrics"
)

// Configuration parameters for metrics data source and extractor.
type (
	// Data source configuration parameters
	metricsDatasourceParams struct {
		// Scheme defines the protocol scheme used in metrics retrieval (e.g., "http").
		Scheme string `json:"scheme"`
		// Path defines the URL path used in metrics retrieval (e.g., "/metrics").
		Path string `json:"path"`
		// InsecureSkipVerify defines whether model server certificate should be verified or not.
		InsecureSkipVerify bool `json:"insecureSkipVerify"`
	}

	// engineConfigParams holds metric specifications for a specific engine type.
	engineConfigParams struct {
		// Name is the engine type identifier.
		Name string `json:"name"`
		// Default marks this engine as the default fallback when Pod has no engine label.
		// Only one engine should be marked as default.
		Default bool `json:"default"`
		// QueuedRequestsSpec defines the metric specification string for retrieving queued request count.
		QueuedRequestsSpec string `json:"queuedRequestsSpec"`
		// RunningRequestsSpec defines the metric specification string for retrieving running requests count.
		RunningRequestsSpec string `json:"runningRequestsSpec"`
		// KVUsageSpec defines the metric specification string for retrieving KV cache usage.
		KVUsageSpec string `json:"kvUsageSpec"`
		// LoRASpec defines the metric specification string for retrieving LoRA availability.
		LoRASpec string `json:"loraSpec"`
		// CacheInfoSpec defines the metrics specification string for retrieving KV cache configuration.
		CacheInfoSpec string `json:"cacheInfoSpec"`
	}

	// Extractor configuration parameters
	modelServerExtractorParams struct {
		// EngineLabelKey is the Pod label key used to identify the engine type.
		// Defaults to "inference.networking.k8s.io/engine-type".
		EngineLabelKey string `json:"engineLabelKey"`
		// EngineConfigs defines metric specifications for specific engine types.
		// If not specified, default vLLM and SGLang configurations are used.
		EngineConfigs []engineConfigParams `json:"engineConfigs"`
	}
)

// Default engine configurations for vLLM and SGLang.
var defaultEngineConfigs = []engineConfigParams{
	{
		Name:                "vllm",
		Default:             true,
		QueuedRequestsSpec:  "vllm:num_requests_waiting",
		RunningRequestsSpec: "vllm:num_requests_running",
		KVUsageSpec:         "vllm:kv_cache_usage_perc",
		LoRASpec:            "vllm:lora_requests_info",
		CacheInfoSpec:       "vllm:cache_config_info",
	},
	{
		Name:                "sglang",
		Default:             false,
		QueuedRequestsSpec:  "sglang:num_queue_reqs",
		RunningRequestsSpec: "sglang:num_running_reqs",
		KVUsageSpec:         "sglang:token_usage",
		LoRASpec:            "",
		CacheInfoSpec:       "",
	},
}

// MetricsDataSourceFactory is a factory function used to instantiate data layer's
// metrics data source plugins specified in a configuration.
func MetricsDataSourceFactory(name string, parameters json.RawMessage, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg, err := defaultDataSourceConfigParams()
	if err != nil {
		return nil, err
	}

	if parameters != nil { // overlay the defaults with configured values
		if err := json.Unmarshal(parameters, cfg); err != nil {
			return nil, err
		}
	}

	ds := http.NewHTTPDataSource(cfg.Scheme, cfg.Path, cfg.InsecureSkipVerify, MetricsDataSourceType,
		name, parseMetrics, PrometheusMetricType)
	return ds, nil
}

// ModelServerExtractorFactory is a factory function used to instantiate data layer's metrics
// Extractor plugins specified in a configuration.
func ModelServerExtractorFactory(name string, parameters json.RawMessage, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := defaultExtractorConfigParams()

	if parameters != nil { // overlay the defaults with configured values
		if err := json.Unmarshal(parameters, cfg); err != nil {
			return nil, err
		}
	}

	registry := NewMappingRegistry()

	// Validate and register engine configurations
	defaultCount := 0
	for _, engineConfig := range cfg.EngineConfigs {
		if engineConfig.Name == "" {
			return nil, errors.New("engine config name cannot be empty")
		}

		mapping, err := NewMapping(
			engineConfig.QueuedRequestsSpec,
			engineConfig.RunningRequestsSpec,
			engineConfig.KVUsageSpec,
			engineConfig.LoRASpec,
			engineConfig.CacheInfoSpec,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create mapping for engine %q: %w", engineConfig.Name, err)
		}

		// Register by engine name
		if err := registry.Register(engineConfig.Name, mapping); err != nil {
			return nil, fmt.Errorf("failed to register engine mapping for %q: %w", engineConfig.Name, err)
		}

		// Also register as default if marked
		if engineConfig.Default {
			defaultCount++
			if defaultCount > 1 {
				return nil, errors.New("only one engine can be marked as default")
			}
			if err := registry.Register(DefaultEngineType, mapping); err != nil {
				return nil, fmt.Errorf("failed to register default mapping: %w", err)
			}
		}
	}

	// Ensure at least one default is registered
	if defaultCount == 0 {
		return nil, errors.New("at least one engine must be marked as default (isDefault: true)")
	}

	extractor, err := NewModelServerExtractor(registry, cfg.EngineLabelKey)
	if err != nil {
		return nil, err
	}
	extractor.typedName.Name = name
	return extractor, nil
}

// Names of CLI flags in main
//
// TODO:
//
//  1. Consider having a cli package with all flag names and constants?
//     Can't use values from runserver as this creates an import cycle with datalayer.
//     Given that relevant issues/PRs have been closed so may be able to remove the cycle?
//     Comment from runserver package (regarding TestPodMetricsClient *backendmetrics.FakePodMetricsClient)
//     This should only be used in tests. We won't need this once we do not inject metrics in the tests.
//     TODO:(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/432) Cleanup
//
//  2. Deprecation notice on these flags being moved to the configuration file
const (
	modelServerMetricsPathFlag               = "model-server-metrics-path"
	modelServerMetricsSchemeFlag             = "model-server-metrics-scheme"
	modelServerMetricsInsecureSkipVerifyFlag = "model-server-metrics-https-insecure-skip-verify"
)

// return the default configuration state. The defaults are populated from
// existing command line flags.
func defaultDataSourceConfigParams() (*metricsDatasourceParams, error) {
	cfg := &metricsDatasourceParams{}

	scheme, err := fromStringFlag(modelServerMetricsSchemeFlag)
	if err != nil {
		return nil, err
	}
	cfg.Scheme = scheme

	path, err := fromStringFlag(modelServerMetricsPathFlag)
	if err != nil {
		return nil, err
	}
	cfg.Path = path

	insecure, err := fromBoolFlag(modelServerMetricsInsecureSkipVerifyFlag)
	if err != nil {
		return nil, err
	}
	cfg.InsecureSkipVerify = insecure

	return cfg, nil
}

func defaultExtractorConfigParams() *modelServerExtractorParams {
	return &modelServerExtractorParams{
		EngineLabelKey: DefaultEngineTypeLabelKey,
		EngineConfigs:  defaultEngineConfigs,
	}
}

func fromStringFlag(name string) (string, error) {
	f := flag.Lookup(name)
	if f == nil {
		return "", fmt.Errorf("flag not found: %s", name)
	}
	return f.Value.String(), nil
}

func fromBoolFlag(name string) (bool, error) {
	f := flag.Lookup(name)
	if f == nil {
		return false, fmt.Errorf("flag not found: %s", name)
	}
	return f.Value.String() == "true", nil
}

func parseMetrics(data io.Reader) (any, error) {
	parser := expfmt.NewTextParser(model.LegacyValidation)
	return parser.TextToMetricFamilies(data)
}
