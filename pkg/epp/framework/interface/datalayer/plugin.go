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

var (
	ExtractorType             = reflect.TypeOf((*Extractor)(nil)).Elem()
	NotificationExtractorType = reflect.TypeOf((*NotificationExtractor)(nil)).Elem()
	NotificationEventType     = reflect.TypeOf(NotificationEvent{})
)

// DataSource provides raw data to registered Extractors.
// For poll-based sources, use PollingDataSource.
// For event-driven sources, use NotificationSource.
type DataSource interface {
	plugin.Plugin
	// Extractors returns a list of registered Extractor names.
	Extractors() []string
	// OutputType returns the type of data this DataSource produces.
	// Used for validating extractor compatibility.
	OutputType() reflect.Type
	// ExtractorType returns the type of Extractor this DataSource expects.
	// For poll-based sources, this is the base Extractor interface.
	// For notification sources, this is the NotificationExtractor interface.
	ExtractorType() reflect.Type
	// AddExtractor adds an extractor to the data source. Multiple
	// Extractors can be registered.
	// The extractor will be called whenever the DataSource might
	// have some new raw information regarding an endpoint.
	// The Extractor's expected input type should be validated against
	// the data source's output type upon registration.
	AddExtractor(extractor Extractor) error
}

// PollingDataSource is a poll-based DataSource that fetches data at regular intervals.
type PollingDataSource interface {
	DataSource
	// Poll is triggered by the data layer framework to fetch potentially new
	// data for an endpoint. Poll calls registered Extractors to convert the
	// raw data into structured attributes.
	Poll(ctx context.Context, ep Endpoint) error
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

// ValidatingDataSource is an optional interface that DataSources can implement
// to perform additional custom validation when adding extractors.
type ValidatingDataSource interface {
	// ValidateExtractor allows the DataSource to perform additional validation
	// beyond the standard type compatibility checks. Return an error if validation fails.
	ValidateExtractor(extractor Extractor) error
}
