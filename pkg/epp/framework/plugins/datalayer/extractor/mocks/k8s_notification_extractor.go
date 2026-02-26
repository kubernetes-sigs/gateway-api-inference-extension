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

package mocks

import (
	"context"
	"reflect"
	"sync"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// NotificationExtractor implements both Extractor and NotificationExtractor for testing.
// It records all events it receives and provides helper methods for test assertions.
type NotificationExtractor struct {
	name       string
	events     []fwkdl.NotificationEvent
	mu         sync.Mutex
	extractErr error
}

// NewNotificationExtractor creates a new mock extractor with the given name.
func NewNotificationExtractor(name string) *NotificationExtractor {
	return &NotificationExtractor{
		name: name,
	}
}

// WithExtractError configures the extractor to return an error on ExtractNotification.
func (m *NotificationExtractor) WithExtractError(err error) *NotificationExtractor {
	m.extractErr = err
	return m
}

func (m *NotificationExtractor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "mock-extractor", Name: m.name}
}

func (m *NotificationExtractor) ExpectedInputType() reflect.Type {
	return reflect.TypeOf(unstructured.Unstructured{})
}

// Extract is the Extractor interface method — no-op for notification extractors.
func (m *NotificationExtractor) Extract(_ context.Context, _ any, _ fwkdl.Endpoint) error {
	return nil
}

// ExtractNotification is the NotificationExtractor method — records the event.
func (m *NotificationExtractor) ExtractNotification(_ context.Context, event fwkdl.NotificationEvent) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.events = append(m.events, event)
	return m.extractErr
}

// GetEvents returns a copy of all recorded events.
func (m *NotificationExtractor) GetEvents() []fwkdl.NotificationEvent {
	m.mu.Lock()
	defer m.mu.Unlock()
	result := make([]fwkdl.NotificationEvent, len(m.events))
	copy(result, m.events)
	return result
}

// Reset clears all recorded events.
func (m *NotificationExtractor) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.events = nil
}
