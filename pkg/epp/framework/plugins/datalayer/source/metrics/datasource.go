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
	"github.com/spf13/pflag"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/source/http"
)

const MetricsDataSourceType = "metrics-data-source"

// Default values for the metrics data source configuration.
const (
	defaultMetricsScheme             = "http"
	defaultMetricsPath               = "/metrics"
	defaultMetricsInsecureSkipVerify = true
)

// metricsDatasourceParams holds the configuration parameters for the metrics data source plugin.
// These values can be specified in the EndpointPickerConfig under the plugin's `parameters` field.
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

// These flags are registered in options.go (server package) and marked as deprecated there.
// They are kept for one release cycle to give users time to migrate their configuration
// to the EndpointPickerConfig `parameters` field (metricsDatasourceParams).
// They will be removed in a future release.
//
// TODO: Remove these constants and defaultDataSourceConfigParams() once the deprecated flags
// are removed from options.go.
// Note: these flag names are duplicated here (rather than imported from the server package)
// to avoid an import cycle between the datalayer plugin and the server/runserver packages.
const (
	modelServerMetricsPathFlag               = "model-server-metrics-path"
	modelServerMetricsSchemeFlag             = "model-server-metrics-scheme"
	modelServerMetricsInsecureSkipVerifyFlag = "model-server-metrics-https-insecure-skip-verify"
)

// DataSource parameters values - Priority (lowest → highest):
//  1. Built-in defaults (defaultMetricsScheme / defaultMetricsPath / defaultMetricsInsecureSkipVerify)
//  2. Deprecated CLI flag value (when the flag is registered and has been set by the operator)
//  3. Explicit plugin `parameters` in EndpointPickerConfig
func defaultDataSourceConfigParams() (*metricsDatasourceParams, error) {
	cfg := &metricsDatasourceParams{
		Scheme:             defaultMetricsScheme,
		Path:               defaultMetricsPath,
		InsecureSkipVerify: defaultMetricsInsecureSkipVerify,
	}

	if scheme, ok := fromStringFlag(modelServerMetricsSchemeFlag); ok {
		cfg.Scheme = scheme
	}

	if path, ok := fromStringFlag(modelServerMetricsPathFlag); ok {
		cfg.Path = path
	}

	if insecure, ok, err := fromBoolFlag(modelServerMetricsInsecureSkipVerifyFlag); err != nil {
		return nil, err
	} else if ok {
		cfg.InsecureSkipVerify = insecure
	}

	return cfg, nil
}

// fromStringFlag returns the value of a registered pflag string flag.
// The second return value is false when the flag is not registered; no error is returned in that case.
func fromStringFlag(name string) (string, bool) {
	f := pflag.Lookup(name)
	if f == nil {
		return "", false
	}
	return f.Value.String(), true
}

// fromBoolFlag returns the value of a registered pflag bool flag.
// The second return value is false when the flag is not registered; no error is returned in that case.
// An error is returned only when the flag exists but its value cannot be parsed as a bool.
func fromBoolFlag(name string) (bool, bool, error) {
	f := pflag.Lookup(name)
	if f == nil {
		return false, false, nil
	}
	b, err := strconv.ParseBool(f.Value.String())
	if err != nil {
		return false, false, fmt.Errorf("invalid bool flag %q: %w", name, err)
	}
	return b, true, nil
}

func parseMetrics(data io.Reader) (any, error) {
	parser := expfmt.NewTextParser(model.LegacyValidation)
	return parser.TextToMetricFamilies(data)
}
