/*
Copyright 2026 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

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
// The source then dispatches to its extractors.
type NotificationSource interface {
	DataSource
	// GVK returns the GroupVersionKind this source watches.
	GVK() schema.GroupVersionKind
	// Notify is called by the framework core when a mutation event fires.
	// The event object is already deep-copied.
	Notify(ctx context.Context, event NotificationEvent) error
}

// NotificationExtractor processes k8s object events pushed from a
// NotificationSource. It embeds Extractor so it can be stored via
// DataSource.AddExtractor. The Extractor.Extract method is never called
// on the notification path — ExtractNotification is used instead.
type NotificationExtractor interface {
	Extractor
	// ExtractNotification processes a notification event. Called synchronously
	// by the source in event order.
	ExtractNotification(ctx context.Context, event NotificationEvent) error
}
