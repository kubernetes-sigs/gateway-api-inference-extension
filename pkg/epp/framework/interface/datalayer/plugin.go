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

// EventType represents the type of endpoint event.
type EventType int

const (
	EventAddedOrModified EventType = iota
	EventDeleted
)

// ChangeNotification represents a data store change event.
type ChangeNotification struct {
	Type EventType
	Data any // Additional data from the data source (e.g., Pod).
}

// DataSourceBase defines common functionality for all data sources.
type DataSourceBase interface {
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
}

// DataSource collects data on timer ticks.
type DataSource interface {
	DataSourceBase
	// Collect is triggered by the data layer framework to fetch potentially new
	// data for an endpoint. Collect calls registered Extractors to convert the
	// raw data into structured attributes.
	Collect(ctx context.Context, ep Endpoint) error
}

// StoreNotifier defines method used to register a NotificationDataSource with the
// data store.
type StoreNotifier interface {
	// RegisterObserver registers an observer for change notification events.
	// Returns an error if the observer's expected data type doesn't match what
	// the store provides or the notifications on the requested object type are not supported.
	RegisterObserver(objType reflect.Type, observer NotificationDataSource) error
}

// NotificationDataSource receives change notifications from the data store.
type NotificationDataSource interface {
	DataSourceBase
	// ExpectedEventDataType defines the type expected by the source in notification events.
	ExpectedEventDataType() reflect.Type
	// AttachToStore registers the data source with the data store.
	AttachToStore(notifier StoreNotifier) error
	// Notify is called when a relevant change is made in the data store.
	// Endpoint is optional and set for endpoint-specific notifications.
	Notify(ctx context.Context, event ChangeNotification, ep Endpoint) error
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
