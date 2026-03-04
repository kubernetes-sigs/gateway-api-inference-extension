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

	"k8s.io/apimachinery/pkg/runtime/schema"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// NotificationExtractor implements both Extractor and NotificationExtractor for testing.
// It records all events it receives and provides helper methods for test assertions.
type NotificationExtractor struct {
	name       string
	gvk        schema.GroupVersionKind
	events     []fwkdl.NotificationEvent
	mu         sync.Mutex
	extractErr error
}

// NewNotificationExtractor creates a new mock extractor with the given name.
func NewNotificationExtractor(name string) *NotificationExtractor {
	return &NotificationExtractor{
		name: name,
		gvk:  schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
	}
}

// WithGVK sets the GVK for the mock extractor.
func (m *NotificationExtractor) WithGVK(gvk schema.GroupVersionKind) *NotificationExtractor {
	m.gvk = gvk
	return m
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
	return reflect.TypeOf(fwkdl.NotificationEvent{})
}

// Extract is the Extractor interface method — no-op for notification extractors.
func (m *NotificationExtractor) Extract(_ context.Context, _ any, _ fwkdl.Endpoint) error {
	return nil
}

func (m *NotificationExtractor) GVK() schema.GroupVersionKind {
	return m.gvk
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

// Extractor is a generic mock for Extractor interface.
// It records call counts and optionally updates endpoint metrics.
type Extractor struct {
	name            string
	inputType       reflect.Type
	callCount       int
	mu              sync.Mutex
	extractErr      error
	updateMetrics   bool
	metricsToUpdate *fwkdl.Metrics
}

// NewExtractor creates a new mock extractor with the given name and input type.
func NewExtractor(name string, inputType reflect.Type) *Extractor {
	return &Extractor{
		name:      name,
		inputType: inputType,
	}
}

// NewPollingExtractor creates a mock extractor for polling data sources (Metrics type).
func NewPollingExtractor(name string) *Extractor {
	return NewExtractor(name, reflect.TypeOf(fwkdl.Metrics{}))
}

// WithExtractError configures the extractor to return an error.
func (m *Extractor) WithExtractError(err error) *Extractor {
	m.extractErr = err
	return m
}

// WithMetricsUpdate configures the extractor to update endpoint metrics.
func (m *Extractor) WithMetricsUpdate(metrics *fwkdl.Metrics) *Extractor {
	m.updateMetrics = true
	m.metricsToUpdate = metrics
	return m
}

func (m *Extractor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "mock-extractor", Name: m.name}
}

func (m *Extractor) ExpectedInputType() reflect.Type {
	return m.inputType
}

func (m *Extractor) Extract(_ context.Context, data any, ep fwkdl.Endpoint) error {
	m.mu.Lock()
	m.callCount++
	m.mu.Unlock()

	if m.updateMetrics && m.metricsToUpdate != nil {
		ep.UpdateMetrics(m.metricsToUpdate)
	}

	return m.extractErr
}

// CallCount returns the number of times Extract was called.
func (m *Extractor) CallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.callCount
}

// GetCallCount returns call count (thread-safe).
func (m *Extractor) GetCallCount() int {
	return m.CallCount()
}

// Reset clears the call count.
func (m *Extractor) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount = 0
}
