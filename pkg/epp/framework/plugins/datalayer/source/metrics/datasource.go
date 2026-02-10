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
	"fmt"
	"io"
	"strconv"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
	flag "github.com/spf13/pflag"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/source/http"
)

const MetricsDataSourceType = "metrics-data-source"

// Data source configuration parameters
type metricsDatasourceParams struct {
	// Scheme defines the protocol scheme used in metrics retrieval (e.g., "http").
	Scheme string `json:"scheme"`
	// Path defines the URL path used in metrics retrieval (e.g., "/metrics").
	Path string `json:"path"`
	// InsecureSkipVerify defines whether model server certificate should be verified or not.
	InsecureSkipVerify bool `json:"insecureSkipVerify"`
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

	return http.NewHTTPDataSource(cfg.Scheme, cfg.Path, cfg.InsecureSkipVerify, MetricsDataSourceType,
		name, parseMetrics, PrometheusMetricType)
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

func parseMetrics(data io.Reader) (any, error) {
	parser := expfmt.NewTextParser(model.LegacyValidation)
	return parser.TextToMetricFamilies(data)
}
