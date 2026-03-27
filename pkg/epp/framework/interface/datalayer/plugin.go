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

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

var (
	ExtractorType             = reflect.TypeFor[Extractor]()
	NotificationExtractorType = reflect.TypeFor[NotificationExtractor]()
	NotificationEventType     = reflect.TypeFor[NotificationEvent]()
)

// DataSource provides raw data to registered Extractors.
// For poll-based sources, use PollingDataSource.
// For event-driven sources, use NotificationSource.
type DataSource interface {
	plugin.Plugin
	// OutputType returns the type of data this DataSource produces.
	// Used for validating extractor compatibility.
	OutputType() reflect.Type
	// ExtractorType returns the type of Extractor this DataSource expects.
	// For poll-based sources, this is the base Extractor interface.
	// For notification sources, this is the NotificationExtractor interface.
	ExtractorType() reflect.Type
}

// PollingDataSource is a poll-based DataSource that fetches data at regular intervals.
type PollingDataSource interface {
	DataSource
	// Poll fetches data for an endpoint and returns it.
	// The Runtime handles calling extractors with the returned data.
	Poll(ctx context.Context, ep Endpoint) (any, error)
}

// Extractor transforms raw data into structured attributes.
type Extractor interface {
	plugin.Plugin
	// ExpectedInputType defines the type expected by the extractor.
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

// EventType identifies the type of mutation that triggered the notification.
type EventType int

const (
	// EventAddOrUpdate is fired when a k8s object is created or updated.
	EventAddOrUpdate EventType = iota
	// EventDelete is fired when a k8s object is deleted.
	EventDelete
)

// NotificationEvent carries the event type and the affected object.
// Object is deep-copied by the framework core before delivery.
type NotificationEvent struct {
	// Type is the mutation type.
	Type EventType
	// Object is the current state of the object (for add/update) or the
	// last known state (for delete). Note that for delete notifications
	// only the object's name and namespace can be relied on.
	Object *unstructured.Unstructured
}

// NotificationSource is an event-driven DataSource for a single k8s GVK.
// The framework core owns the k8s notification mechanisms (e.g., watches,
// caches, informers) and calls the source's Notify on events.
type NotificationSource interface {
	DataSource
	// GVK returns the GroupVersionKind this source watches.
	GVK() schema.GroupVersionKind
	// Notify is called by the framework core when a mutation event fires.
	// The event object is already deep-copied.
	// Returns the event (possibly modified) for Runtime to dispatch to extractors.
	// Returns nil event to signal Runtime to skip extractor dispatch.
	// TODO: ahy accept event but return *event?
	Notify(ctx context.Context, event NotificationEvent) (*NotificationEvent, error)
}

// NotificationExtractor processes k8s object events pushed from a
// NotificationSource.
type NotificationExtractor interface {
	Extractor
	// GVK returns the GroupVersionKind this extractor handles.
	GVK() schema.GroupVersionKind
	// ExtractNotification processes a notification event. Called synchronously
	// by the source in event order.
	ExtractNotification(ctx context.Context, event NotificationEvent) error
}
