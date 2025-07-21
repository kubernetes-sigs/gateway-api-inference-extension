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

// DataSource provides raw data to registered Extractors.
type DataSource interface {
	Name() string
	// AddExtractor adds an extractor to the data source. Multiple
	// Extractors can be registered.
	// The extractor will be called whenever the DataSource might
	// have some new raw information regarding an endpoint.
	// The Extractor's expected input type should be validated against
	// the data source's output type upon registration.
	AddExtractor(extractor Extractor) error
	// Collect is triggered by the data layer framework to fetch potentially new
	// data for an endpoint. Collect calls registered Extractors to convert the
	// raw data into structured attributes.
	Collect(ep Endpoint)
}

// Extractor transforms raw data into structured attributes.
type Extractor interface {
	Name() string
	// ExpectedType defines the type expected by the extractor.
	ExpectedInputType() reflect.Type
	// Extract transforms the raw data source output into a concrete structured
	// attribute, stored on the given endpoint.
	Extract(data any, ep Endpoint)
}

var defaultDataSources = DataSourceRegistry{}

// DataSourceRegistry stores named data sources.
type DataSourceRegistry struct {
	sources sync.Map
}

// Register adds a new DataSource to the registry.
func (dsr *DataSourceRegistry) Register(src DataSource) error {
	if src == nil {
		return errors.New("unable to register a nil data source")
	}
	if _, loaded := dsr.sources.LoadOrStore(src.Name(), src); loaded {
		return fmt.Errorf("unable to register duplicate data source: %s", src.Name())
	}
	return nil
}

// GetNamedSource fetches a source by name.
func (dsr *DataSourceRegistry) GetNamedSource(name string) (DataSource, bool) {
	if val, ok := dsr.sources.Load(name); ok {
		if ds, ok := val.(DataSource); ok {
			return ds, true
		}
	}
	return nil, false
}

// GetSources returns all registered sources.
func (dsr *DataSourceRegistry) GetSources() []DataSource {
	var result []DataSource
	dsr.sources.Range(func(_, val any) bool {
		if ds, ok := val.(DataSource); ok {
			result = append(result, ds)
		}
		return true
	})
	return result
}

// --- default registry accessors ---

func RegisterSource(src DataSource) error {
	return defaultDataSources.Register(src)
}

func GetNamedSource(name string) (DataSource, bool) {
	return defaultDataSources.GetNamedSource(name)
}

func GetSources() []DataSource {
	return defaultDataSources.GetSources()
}

// ValidateExtractorType checks if an extractor can handle
// the DataSource's output. It should be called by a DataSource
// when an extractor is added.
func ValidateExtractorType(output, input reflect.Type) error {
	if output == nil || input == nil {
		return fmt.Errorf("extractor input type or data source output type can't be nil")
	}
	if output == input ||
		(input.Kind() == reflect.Interface && input.NumMethod() == 0) ||
		(input.Kind() == reflect.Interface && output.Implements(input)) {
		return nil
	}
	return fmt.Errorf("extractor input type %v cannot handle data source output type %v",
		input, output)
}
