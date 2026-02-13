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

package notifications

import (
	"context"
	"reflect"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// MockNotificationExtractor implements both Extractor and NotificationExtractor for testing.
// It records all events it receives and provides helper methods for test assertions.
type MockNotificationExtractor struct {
	name       string
	events     []fwkdl.NotificationEvent
	mu         sync.Mutex
	extractErr error
}

// NewMockNotificationExtractor creates a new mock extractor with the given name.
func NewMockNotificationExtractor(name string) *MockNotificationExtractor {
	return &MockNotificationExtractor{
		name: name,
	}
}

// WithExtractError configures the extractor to return an error on ExtractNotification.
func (m *MockNotificationExtractor) WithExtractError(err error) *MockNotificationExtractor {
	m.extractErr = err
	return m
}

func (m *MockNotificationExtractor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "mock-extractor", Name: m.name}
}

func (m *MockNotificationExtractor) ExpectedInputType() reflect.Type {
	return reflect.TypeOf(unstructured.Unstructured{})
}

// Extract is the Extractor interface method — no-op for notification extractors.
func (m *MockNotificationExtractor) Extract(_ context.Context, _ any, _ fwkdl.Endpoint) error {
	return nil
}

// ExtractNotification is the NotificationExtractor method — records the event.
func (m *MockNotificationExtractor) ExtractNotification(_ context.Context, event fwkdl.NotificationEvent) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.events = append(m.events, event)
	return m.extractErr
}

// GetEvents returns a copy of all recorded events.
func (m *MockNotificationExtractor) GetEvents() []fwkdl.NotificationEvent {
	m.mu.Lock()
	defer m.mu.Unlock()
	result := make([]fwkdl.NotificationEvent, len(m.events))
	copy(result, m.events)
	return result
}

// WaitForEvents waits until at least count events have been received or timeout expires.
// Returns true if the count was reached, false if timeout occurred.
func (m *MockNotificationExtractor) WaitForEvents(count int, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		m.mu.Lock()
		current := len(m.events)
		m.mu.Unlock()
		if current >= count {
			return true
		}
		time.Sleep(10 * time.Millisecond)
	}
	return false
}

// Reset clears all recorded events.
func (m *MockNotificationExtractor) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.events = nil
}
