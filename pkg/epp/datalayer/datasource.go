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

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

var defaultDataSources = DataSourceRegistry{}

// DataSourceRegistry stores named data sources.
type DataSourceRegistry struct {
	sources sync.Map
}

// Register adds a new DataSource to the registry. If the source implements
// NotificationSource, GVK uniqueness is enforced.
func (dsr *DataSourceRegistry) Register(src fwkdl.DataSource) error {
	if src == nil {
		return errors.New("unable to register a nil data source")
	}

	if ns, ok := src.(fwkdl.NotificationSource); ok { // check GVK uniqueness
		gvk := ns.GVK()
		var duplicateGVK bool
		dsr.sources.Range(func(_, val any) bool {
			if existing, ok := val.(fwkdl.NotificationSource); ok {
				if existing.GVK() == gvk {
					duplicateGVK = true
					return false
				}
			}
			return true
		})
		if duplicateGVK {
			return fmt.Errorf("duplicate notification source for GVK %s", gvk)
		}
	}

	if _, loaded := dsr.sources.LoadOrStore(src.TypedName().Name, src); loaded {
		return fmt.Errorf("unable to register duplicate data source: %s", src.TypedName().String())
	}
	return nil
}

// GetSources returns all registered sources.
func (dsr *DataSourceRegistry) GetSources() []fwkdl.DataSource {
	var result []fwkdl.DataSource
	dsr.sources.Range(func(_, val any) bool {
		if ds, ok := val.(fwkdl.DataSource); ok {
			result = append(result, ds)
		}
		return true
	})
	return result
}

// --- default registry accessors ---

// RegisterSource adds a new data source to the default registry.
func RegisterSource(src fwkdl.DataSource) error {
	return defaultDataSources.Register(src)
}

// GetSources returns the list of data sources registered in the default registry.
func GetSources() []fwkdl.DataSource {
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
