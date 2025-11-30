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
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
)

// DataSource provides raw data to registered Extractors.
type DataSource interface {
	plugins.Plugin
	// Extractors returns a list of registered Extractor names.
	Extractors() []string
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
	Collect(ctx context.Context, ep Endpoint) error
}

// Extractor transforms raw data into structured attributes.
type Extractor interface {
	plugins.Plugin
	// ExpectedType defines the type expected by the extractor.
	ExpectedInputType() reflect.Type
	// Extract transforms the raw data source output into a concrete structured
	// attribute, stored on the given endpoint.
	Extract(ctx context.Context, data any, ep Endpoint) error
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
	if _, loaded := dsr.sources.LoadOrStore(src.TypedName().Name, src); loaded {
		return fmt.Errorf("unable to register duplicate data source: %s", src.TypedName().String())
	}
	return nil
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

// RegisterSource adds a new data source to the default registry.
func RegisterSource(src DataSource) error {
	return defaultDataSources.Register(src)
}

// GetSources returns the list of data sources registered in the default registry.
func GetSources() []DataSource {
	return defaultDataSources.GetSources()
}

// ValidateExtractorType checks if an extractor can handle
// the DataSource's output. It should be called by a DataSource
// when an extractor is added.
func ValidateExtractorType(collectorOutput, extractorInput reflect.Type) error {
	if collectorOutput == nil || extractorInput == nil {
		return errors.New("extractor input type or data source output type can't be nil")
	}
	if collectorOutput == extractorInput ||
		(extractorInput.Kind() == reflect.Interface && extractorInput.NumMethod() == 0) ||
		(extractorInput.Kind() == reflect.Interface && collectorOutput.Implements(extractorInput)) {
		return nil
	}
	return fmt.Errorf("extractor input type %v cannot handle data source output type %v",
		extractorInput, collectorOutput)
}
