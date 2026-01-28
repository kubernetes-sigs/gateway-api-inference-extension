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
	"reflect"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// DataSource provides raw data to registered Extractors.
type DataSource interface {
	plugin.Plugin
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
	plugin.Plugin
	// ExpectedType defines the type expected by the extractor.
	ExpectedInputType() reflect.Type
	// Extract transforms the raw data source output into a concrete structured
	// attribute, stored on the given endpoint.
	Extract(ctx context.Context, data any, ep Endpoint) error
}
