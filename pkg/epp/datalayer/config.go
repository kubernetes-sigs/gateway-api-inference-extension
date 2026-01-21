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

package datalayer

import (
	"fmt"
)

// Config defines the configuration of EPP data layer, as the set of DataSources and
// Extractors defined on them.
type Config struct {
	Sources []DataSourceConfig // the data sources configured in the data layer
}

// DataSourceConfig defines the configuration of a specific DataSource
type DataSourceConfig struct {
	Plugin     DataSource  // the data source plugin instance
	Extractors []Extractor // extractors defined for the data source
}

// WithConfig sets up the data layer based on the provided configuration.
// @TODO: To allow running new data sources with backend.PodMetrics collection,
// the code validates that the new and the existing metrics collection are not both
// configured. This is done by passing in the new metrics extractor as "disallowed"
// when the new metrics collection is not enabled in the runner (otherwise it passes
// an empty string).
// Note that it is still possible to configure the new metrics data source with different
// extractors beyond model-server-protocol.
// Referring the "type" directly here would create an import cycle.
// This should be removed once PodMetrics is deprecated.
func WithConfig(cfg *Config, disallowedExtractorType string) error {
	for _, srcCfg := range cfg.Sources {
		for _, extractor := range srcCfg.Extractors {
			if disallowedExtractorType != "" && extractor.TypedName().Type == disallowedExtractorType {
				return fmt.Errorf("disallowed Extractor %s is configured for source %s",
					extractor.TypedName().String(), srcCfg.Plugin.TypedName().String())
			}
			if err := srcCfg.Plugin.AddExtractor(extractor); err != nil {
				return fmt.Errorf("failed to add Extractor %s to DataSource %s: %w", extractor.TypedName(),
					srcCfg.Plugin.TypedName(), err)
			}
		}
		if err := RegisterSource(srcCfg.Plugin); err != nil {
			return fmt.Errorf("failed to register DataSource %s: %w", srcCfg.Plugin.TypedName(), err)
		}
	}
	return nil
}
