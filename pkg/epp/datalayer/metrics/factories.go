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

package metrics

import (
	"encoding/json"
	"fmt"
	"strconv"

	flag "github.com/spf13/pflag"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
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
		Scheme string // `json:"scheme"`
		// Path defines the URL path used in metrics retrieval (e.g., "/metrics").
		Path string // `json:"path"`
		// InsecureSkipVerify defines whether model server certificate should be verified or not.
		InsecureSkipVerify bool // `json:"insecureSkipVerify"`
	}

	// Extractor configuration parameters
	modelServerExtractorParams struct {
		// QueueRequestsSpec defines the metric specification string for retrieving queued request count.
		QueueRequestsSpec string // `json:"queuedRequestsSpec"`
		// RunningRequestsSpec defines the metric specification string for retrieving running requests count.
		RunningRequestsSpec string // `json:"runningRequestsSpec"`
		// KVUsage defines the metric specification string for retrieving KV cache usage.
		KVUsageSpec string // `json:"kvUsageSpec"`
		// LoRASpec defines the metric specification string for retrieving LoRA availability.
		LoRASpec string // `json:"loraSpec"`
		// CacheInfoSpec defines the metrics specification string for retrieving KV cache configuration.
		CacheInfoSpec string // `json:"cacheInfoSpec"`
	}
)

// MetricsDataSourceFactory is a factory function used to instantiate data layer's
// metrics data source plugins specified in a configuration.
func MetricsDataSourceFactory(name string, parameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	cfg, err := defaultDataSourceConfigParams()
	if err != nil {
		return nil, err
	}

	if parameters != nil { // overlay the defaults with configured values
		if err := json.Unmarshal(parameters, cfg); err != nil {
			return nil, err
		}
	}

	ds := NewMetricsDataSource(cfg.Scheme, cfg.Path, cfg.InsecureSkipVerify)
	ds.typedName.Name = name
	return ds, nil
}

// ModelServerExtractorFactory is a factory function used to instantiate data layer's metrics
// Extractor plugins specified in a configuration.
func ModelServerExtractorFactory(name string, parameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	cfg, err := defaultExtractorConfigParams()
	if err != nil {
		return nil, err
	}

	if parameters != nil { // overlay the defaults with configured values
		if err := json.Unmarshal(parameters, cfg); err != nil {
			return nil, err
		}
	}

	extractor, err := NewModelServerExtractor(cfg.QueueRequestsSpec, cfg.RunningRequestsSpec, cfg.KVUsageSpec,
		cfg.LoRASpec, cfg.CacheInfoSpec)
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
	totalQueuedRequestsMetricSpecFlag        = "total-queued-requests-metric"
	totalRunningRequestsMetricSpecFlag       = "total-running-requests-metric"
	kvCacheUsagePercentageMetricSpecFlags    = "kv-cache-usage-percentage-metric"
	loraInfoMetricSpecFlag                   = "lora-info-metric"
	cacheInfoMetricSpecFlag                  = "cache-info-metric"
	modelServerMetricsPathFlag               = "model-server-metrics-path"
	modelServerMetricsSchemeFlag             = "model-server-metrics-scheme"
	modelServerMetricsInsecureSkipVerifyFlag = "model-server-metrics-https-insecure-skip-verify"
)

// return the default configuration state. The defaults are populated from
// existing command line flags.
func defaultDataSourceConfigParams() (*metricsDatasourceParams, error) {
	var err error
	cfg := &metricsDatasourceParams{}

	if cfg.Scheme, err = fromStringFlag(modelServerMetricsSchemeFlag); err != nil {
		return nil, err
	}
	if cfg.Path, err = fromStringFlag(modelServerMetricsPathFlag); err != nil {
		return nil, err
	}
	if cfg.InsecureSkipVerify, err = fromBoolFlag(modelServerMetricsInsecureSkipVerifyFlag); err != nil {
		return nil, err
	}
	return cfg, nil
}

func defaultExtractorConfigParams() (*modelServerExtractorParams, error) {
	var err error
	cfg := &modelServerExtractorParams{}

	if cfg.QueueRequestsSpec, err = fromStringFlag(totalQueuedRequestsMetricSpecFlag); err != nil {
		return nil, err
	}
	if cfg.RunningRequestsSpec, err = fromStringFlag(totalRunningRequestsMetricSpecFlag); err != nil {
		return nil, err
	}
	if cfg.KVUsageSpec, err = fromStringFlag(kvCacheUsagePercentageMetricSpecFlags); err != nil {
		return nil, err
	}
	if cfg.LoRASpec, err = fromStringFlag(loraInfoMetricSpecFlag); err != nil {
		return nil, err
	}
	if cfg.CacheInfoSpec, err = fromStringFlag(cacheInfoMetricSpecFlag); err != nil {
		return nil, err
	}

	return cfg, nil
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
	b, err := strconv.ParseBool(f.Value.String())
	if err != nil {
		return false, fmt.Errorf("invalid bool flag %q: %w", name, err)
	}
	return b, nil
}
