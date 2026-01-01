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

	"github.com/go-logr/logr"
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
func WithConfig(cfg *Config, log logr.Logger) error {
	for _, srcCfg := range cfg.Sources {
		for _, extractor := range srcCfg.Extractors {
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
