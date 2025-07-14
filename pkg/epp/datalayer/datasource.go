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
)

// DataSource is an interface required from all datalayer data collection
// sources.
type DataSource interface {
	// Name returns the name of this datasource.
	Name() string

	// Start begins the collection process.
	Start(ctx context.Context) error

	// Stop stops the collection process.
	Stop()

	// AddExtractor adds an extractor to the data source.
	// The extractor will be called whenever the data source might
	// have some new raw information regarding an endpoint.
	// The Extractor's expected input type should be validated against
	// the data source output type upon registration.
	AddExtractor(extractor Extractor) error

	// TODO: the following is useful for a data source that operates on
	// endpoints and might not be relevant for "global/system" collectors which
	// might not need the concept of an endpoint. This can be split, if needed,
	// to a separate interface in the future.

	// AddEndpoint adds an endpoint to collect from.
	AddEndpoint(ep Endpoint) error

	// RemoveEndpoint removes an endpoint from collection.
	RemoveEndpoint(ep Endpoint) error
}

type Extractor interface {
	// Name returns the name of the extractor.
	Name() string

	// ExpectedType defines the type expected by the extractor. It must match
	// the DataSource.OutputType() the extractor registers for.
	ExpectedType() reflect.Type

	// Extract transforms the data source output into a concrete attribute that
	// is stored on the given endpoint.
	Extract(data any, ep Endpoint)
}

var (
	// DefaultDataSources is the system default data source registry.
	DefaultDataSources = DataSourceRegistry{}
)

// DataSourceRegistry stores named data sources and makes them
// accessible to GIE subsystems.
type DataSourceRegistry struct {
	mu      sync.RWMutex
	sources map[string]DataSource
}

// Register adds a source to the registry.
func (dsr *DataSourceRegistry) Register(src DataSource) error {
	if src == nil {
		return errors.New("unable to register a nil data source")
	}

	dsr.mu.Lock()
	defer dsr.mu.Unlock()

	if _, found := dsr.sources[src.Name()]; found {
		return fmt.Errorf("unable to register duplicate data source: %s", src.Name())
	}
	dsr.sources[src.Name()] = src
	return nil
}

// GetNamedSource returns the named data source, if found.
func (dsr *DataSourceRegistry) GetNamedSource(name string) (DataSource, bool) {
	if name == "" {
		return nil, false
	}

	dsr.mu.RLock()
	defer dsr.mu.RUnlock()
	if ds, found := dsr.sources[name]; found {
		return ds, true
	}
	return nil, false
}

// AddEndpoint adds a new endpoint to all registered sources.
// Endpoints are not tracked and DataSources are only notified of
// endpoints added after the data source has been registered.
//
// TODO: track endpoints and update on later source registrations? It seems safe
// to assume that all sources are registered before endpoints are
// discovered and added to the system.
func (dsr *DataSourceRegistry) AddEndpoint(ep Endpoint) error {
	if ep == nil {
		return nil
	}

	dsr.mu.RLock()
	defer dsr.mu.RUnlock()
	errs := []error{}

	for _, ds := range dsr.sources {
		if err := ds.AddEndpoint(ep); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// RemoveEndpoint removes an endpoint from all registered sources.
// A source may be called to remove an endpoint it has not added - this
// is should not result in an error.
func (dsr *DataSourceRegistry) RemoveEndpoint(ep Endpoint) error {
	if ep == nil {
		return nil
	}

	dsr.mu.RLock()
	defer dsr.mu.RUnlock()
	errs := []error{}

	for _, ds := range dsr.sources {
		if err := ds.RemoveEndpoint(ep); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// RegisterSource adds the data source to the default registry.
func RegisterSource(src DataSource) error {
	return DefaultDataSources.Register(src)
}

// GetNamedSource returns the named source from the default registry,
// if found.
func GetNamedSource(name string) (DataSource, bool) {
	return DefaultDataSources.GetNamedSource(name)
}

// AddEndpoint adds an endpoint to all sources in the default source registry.
func AddEndpoint(ep Endpoint) error {
	return DefaultDataSources.AddEndpoint(ep)
}

// RemoveEndpoint removes an endpoint from all sources in the default source
// registry.
func RemoveEndpoint(ep Endpoint) error {
	return DefaultDataSources.RemoveEndpoint(ep)
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
