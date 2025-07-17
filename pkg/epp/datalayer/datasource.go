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
	"errors"
	"fmt"
	"reflect"
	"sync"
)

// DataSource is an interface required from all data layer data collection
// sources.
type DataSource interface {
	// Name returns the name of this datasource.
	Name() string

	// AddExtractor adds an extractor to the data source.
	// The extractor will be called whenever the Collector might
	// have some new raw information regarding an endpoint.
	// The Extractor's expected input type should be validated against
	// the data source's output type upon registration.
	AddExtractor(extractor Extractor) error

	// Collect is triggered by the data layer framework to fetch potentially new
	// data for an endpoint. It passes retrieved data to registered Extractors.
	Collect(ep Endpoint)
}

// Extractor is used to convert raw data into relevant data layer information
// for an endpoint. They are called by data sources whenever new data might be
// be available. Multiple Extractors can be registered with a source. Extractors
// are expected to save their output with an endpoint so it becomes accessible
// to consumers in other subsystem of the inference gateway (e.g., when making
// scheduling decisions).
type Extractor interface {
	// Name returns the name of the extractor.
	Name() string

	// ExpectedType defines the type expected by the extractor. It must match
	// the output type of the data source where the extractor is registered.
	ExpectedInputType() reflect.Type

	// Extract transforms the data source output into a concrete attribute that
	// is stored on the given endpoint.
	Extract(data any, ep Endpoint)
}

var (
	// defaultDataSources is the system default data source registry.
	defaultDataSources = DataSourceRegistry{}
)

// DataSourceRegistry stores named data sources and makes them
// accessible to other subsystems in the inference gateway.
type DataSourceRegistry struct {
	sources sync.Map
}

// Register adds a source to the registry.
func (dsr *DataSourceRegistry) Register(src DataSource) error {
	if src == nil {
		return errors.New("unable to register a nil data source")
	}

	if _, found := dsr.sources.Load(src.Name()); found {
		return fmt.Errorf("unable to register duplicate data source: %s", src.Name())
	}
	dsr.sources.Store(src.Name(), src)
	return nil
}

// GetNamedSource returns the named data source, if found.
func (dsr *DataSourceRegistry) GetNamedSource(name string) (DataSource, bool) {
	if name == "" {
		return nil, false
	}

	if val, found := dsr.sources.Load(name); found {
		if ds, ok := val.(DataSource); ok {
			return ds, true
		} // ignore type assertion failures and fall through
	}
	return nil, false
}

// GetSources returns all sources registered.
func (dsr *DataSourceRegistry) GetSources() []DataSource {
	sources := []DataSource{}
	dsr.sources.Range(func(_, val any) bool {
		if ds, ok := val.(DataSource); ok {
			sources = append(sources, ds)
		}
		return true // continue iteration
	})
	return sources
}

// RegisterSource adds the data source to the default registry.
func RegisterSource(src DataSource) error {
	return defaultDataSources.Register(src)
}

// GetNamedSource returns the named source from the default registry,
// if found.
func GetNamedSource(name string) (DataSource, bool) {
	return defaultDataSources.GetNamedSource(name)
}

// GetSources returns all sources in the default registry.
func GetSources() []DataSource {
	return defaultDataSources.GetSources()
}

// ValidateExtractorType checks if an extractor can handle
// the collector's output.
func ValidateExtractorType(collectorOutputType, extractorInputType reflect.Type) error {
	if collectorOutputType == extractorInputType {
		return nil
	}

	// extractor accepts anything (i.e., interface{})
	if extractorInputType.Kind() == reflect.Interface && extractorInputType.NumMethod() == 0 {
		return nil
	}

	// check if collector output implements extractor input interface
	if collectorOutputType.Implements(extractorInputType) {
		return nil
	}

	return fmt.Errorf("extractor input type %v cannot handle collector output type %v",
		extractorInputType, collectorOutputType)
}
