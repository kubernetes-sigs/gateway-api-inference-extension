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
	"reflect"
	"testing"

	"github.com/spf13/pflag"
)

func resetFlags() {
	pflag.CommandLine = pflag.NewFlagSet("test", pflag.ContinueOnError)

	pflag.String(modelServerMetricsSchemeFlag, "", "")
	pflag.String(modelServerMetricsPathFlag, "", "")
	pflag.Bool(modelServerMetricsInsecureSkipVerifyFlag, false, "")
}

func TestDataSourceConfigParams_UsesBuiltInDefaults(t *testing.T) {
	resetFlags()

	cfg, err := defaultDataSourceConfigParams()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := &metricsDatasourceParams{
		Scheme:             defaultMetricsScheme,
		Path:               defaultMetricsPath,
		InsecureSkipVerify: defaultMetricsInsecureSkipVerify,
	}

	if !reflect.DeepEqual(cfg, expected) {
		t.Fatalf("expected %+v, got %+v", expected, cfg)
	}
}

func TestDataSourceConfigParams_CLIOverridesDefaults(t *testing.T) {
	resetFlags()

	_ = pflag.CommandLine.Set(modelServerMetricsSchemeFlag, "https")
	_ = pflag.CommandLine.Set(modelServerMetricsPathFlag, "/custom")
	_ = pflag.CommandLine.Set(modelServerMetricsInsecureSkipVerifyFlag, "false")

	cfg, err := defaultDataSourceConfigParams()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := &metricsDatasourceParams{
		Scheme:             "https",
		Path:               "/custom",
		InsecureSkipVerify: false,
	}

	if !reflect.DeepEqual(cfg, expected) {
		t.Fatalf("expected %+v, got %+v", expected, cfg)
	}
}

func TestDataSourceConfigParams_ConfigOverridesCLI(t *testing.T) {
	resetFlags()

	// CLI values
	_ = pflag.CommandLine.Set(modelServerMetricsSchemeFlag, "https")
	_ = pflag.CommandLine.Set(modelServerMetricsPathFlag, "/cli")
	_ = pflag.CommandLine.Set(modelServerMetricsInsecureSkipVerifyFlag, "false")

	// First resolve defaults + CLI
	cfg, err := defaultDataSourceConfigParams()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Config JSON overrides
	parameters := metricsDatasourceParams{
		Scheme:             "grpc",
		Path:               "/config",
		InsecureSkipVerify: true,
	}

	raw, _ := json.Marshal(parameters)

	if err := json.Unmarshal(raw, cfg); err != nil {
		t.Fatalf("unexpected error unmarshalling: %v", err)
	}

	expected := &metricsDatasourceParams{
		Scheme:             "grpc",
		Path:               "/config",
		InsecureSkipVerify: true,
	}

	if !reflect.DeepEqual(cfg, expected) {
		t.Fatalf("expected %+v, got %+v", expected, cfg)
	}
}

func TestDataSourceConfigParams_PartialConfigOverride(t *testing.T) {
	resetFlags()

	_ = pflag.CommandLine.Set(modelServerMetricsSchemeFlag, "https")
	_ = pflag.CommandLine.Set(modelServerMetricsPathFlag, "/cli")

	cfg, err := defaultDataSourceConfigParams()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Only override Path via config
	parameters := map[string]any{
		"path": "/config",
	}

	raw, _ := json.Marshal(parameters)

	if err := json.Unmarshal(raw, cfg); err != nil {
		t.Fatalf("unexpected error unmarshalling: %v", err)
	}

	expected := &metricsDatasourceParams{
		Scheme:             "https",   // from CLI
		Path:               "/config", // from config
		InsecureSkipVerify: defaultMetricsInsecureSkipVerify,
	}

	if !reflect.DeepEqual(cfg, expected) {
		t.Fatalf("expected %+v, got %+v", expected, cfg)
	}
}

func TestDataSourceConfigParams_BoolCLIOnlyWhenChanged(t *testing.T) {
	resetFlags()

	// Do NOT set insecure flag — ensure default remains
	// And not "false" (which is the CLI default)
	cfg, err := defaultDataSourceConfigParams()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cfg.InsecureSkipVerify != defaultMetricsInsecureSkipVerify {
		t.Fatalf("expected default bool %v, got %v",
			defaultMetricsInsecureSkipVerify,
			cfg.InsecureSkipVerify)
	}
}
