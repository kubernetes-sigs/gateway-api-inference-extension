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

package mocks

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"time"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// EndpointExtractCall records a single call to Extract for endpoint change notifications.
type EndpointExtractCall struct {
	ChangeNotification fwkdl.ChangeNotification
	Endpoint           fwkdl.Endpoint
	Timestamp          time.Time
	Error              error
}

// MockEndpointExtractor is a mock implementation of fwkdl.Extractor for testing
// endpoint change notification extractors. It expects ChangeNotification as input.
type MockEndpointExtractor struct {
	name         string
	extractError error
	extractCalls []EndpointExtractCall
	mu           sync.Mutex
}

// Verify MockEndpointExtractor implements fwkdl.Extractor.
var _ fwkdl.Extractor = (*MockEndpointExtractor)(nil)

// NewMockEndpointExtractor creates a new mock endpoint extractor that expects
// ChangeNotification as input type.
func NewMockEndpointExtractor(name string) *MockEndpointExtractor {
	return &MockEndpointExtractor{
		name:         name,
		extractCalls: []EndpointExtractCall{},
	}
}

// TypedName returns the type and name of the extractor.
func (m *MockEndpointExtractor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{
		Type: "mock-endpoint-extractor",
		Name: m.name,
	}
}

// ExpectedInputType returns ChangeNotification type.
func (m *MockEndpointExtractor) ExpectedInputType() reflect.Type {
	return reflect.TypeOf(fwkdl.ChangeNotification{})
}

// Extract records the call and returns the configured error (if any).
func (m *MockEndpointExtractor) Extract(ctx context.Context, data any, ep fwkdl.Endpoint) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	notification, ok := data.(fwkdl.ChangeNotification)
	if !ok {
		return fmt.Errorf("expected ChangeNotification, got %T", data)
	}

	call := EndpointExtractCall{
		ChangeNotification: notification,
		Endpoint:           ep,
		Timestamp:          time.Now(),
		Error:              m.extractError,
	}
	m.extractCalls = append(m.extractCalls, call)

	return m.extractError
}

// SetExtractError configures the error to return from Extract.
func (m *MockEndpointExtractor) SetExtractError(err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.extractError = err
}

// GetExtractCalls returns all recorded Extract calls.
func (m *MockEndpointExtractor) GetExtractCalls() []EndpointExtractCall {
	m.mu.Lock()
	defer m.mu.Unlock()
	// Return a copy to avoid race conditions
	calls := make([]EndpointExtractCall, len(m.extractCalls))
	copy(calls, m.extractCalls)
	return calls
}

// GetExtractCallCount returns the number of times Extract was called.
func (m *MockEndpointExtractor) GetExtractCallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.extractCalls)
}

// Reset clears all recorded calls and errors.
func (m *MockEndpointExtractor) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.extractCalls = []EndpointExtractCall{}
	m.extractError = nil
}
